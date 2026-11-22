//! Real GLES2/3 backend for WebGL via dlopen'd libEGL + libGLESv2.
//!
//! Chrome always answers a WebGL context (SwiftShader when no GPU exists) and
//! fingerprint scripts hash rendered output; a shimmed context whose draws are
//! no-ops reads all zero. Here each canvas gets one surfaceless EGL context
//! drawing into its own RGBA8 FBO, so state, queries and `readPixels` are the
//! real thing.
//!
//! Library resolution order:
//!   1. `OBSCURA_GL_EGL` / `OBSCURA_GL_GLES` env overrides (e.g. a Chrome for
//!      Testing directory carrying ANGLE + SwiftShader).
//!   2. System `libEGL.so.1` / `libGLESv2.so.2` (mesa).
//! Initialization failure degrades to `op_gl_create = -1`, preserving the
//! previous null-context behavior on hosts without GL.

#![allow(non_snake_case, non_camel_case_types, clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

type EGLDisplay = *mut c_void;
type EGLConfig = *mut c_void;
type EGLContext = *mut c_void;
type EGLSurface = *mut c_void;
type EGLBoolean = c_uint;
type EGLint = c_int;
type EGLenum = c_uint;
type EGLAttrib = isize;

type GLboolean = u8;
type GLenum = c_uint;
type GLbitfield = c_uint;
type GLuint = c_uint;
type GLint = c_int;
type GLsizei = c_int;
type GLchar = c_char;
type GLubyte = u8;
type GLfloat = f32;
type GLsizeiptr = isize;
type GLintptr = isize;

const EGL_OPENGL_ES_API: EGLenum = 0x30A0;
const EGL_NONE: EGLint = 0x3038;
const EGL_SURFACE_TYPE: EGLint = 0x3033;
const EGL_PBUFFER_BIT: EGLint = 0x01;
const EGL_RENDERABLE_TYPE: EGLint = 0x3040;
const EGL_OPENGL_ES2_BIT: EGLint = 0x04;
const EGL_OPENGL_ES3_BIT: EGLint = 0x40;
const EGL_RED_SIZE: EGLint = 0x3024;
const EGL_GREEN_SIZE: EGLint = 0x3023;
const EGL_BLUE_SIZE: EGLint = 0x3022;
const EGL_ALPHA_SIZE: EGLint = 0x3021;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;
const EGL_CONTEXT_CLIENT_VERSION: EGLint = 0x3098;
const EGL_PLATFORM_SURFACELESS_MESA: EGLenum = 0x3202;

const GL_EXTENSIONS: GLenum = 0x1F03;
const GL_COLOR_ATTACHMENT0: GLenum = 0x8CE0;
const GL_FRAMEBUFFER: GLenum = 0x8D40;
const GL_RENDERBUFFER: GLenum = 0x8D41;
const GL_RGBA8: GLenum = 0x8058;

struct Libs {
    egl: *mut c_void,
    gles: *mut c_void,
}
unsafe impl Send for Libs {}
unsafe impl Sync for Libs {}

struct Egl {
    libs: Libs,
    display: EGLDisplay,
    config: EGLConfig,
}
unsafe impl Send for Egl {}
unsafe impl Sync for Egl {}

static EGL: OnceLock<Option<Egl>> = OnceLock::new();

unsafe fn sym(lib: *mut c_void, name: &'static [u8]) -> *mut c_void {
    libc::dlsym(lib, name.as_ptr().cast())
}

// Resolve a GLES symbol per call site; dlsym is cheap next to any GL call and
// it keeps a ~60-function surface free of a pointer table to keep in sync.
macro_rules! egl_call {
    ($lib:expr, $n:literal, ($($t:ty),*) -> $r:ty, ($($a:expr),*)) => {{
        let f: unsafe extern "C" fn($($t),*) -> $r =
            std::mem::transmute(sym($lib, concat!($n, "\0").as_bytes()));
        f($($a),*)
    }};
}
macro_rules! gl {
    ($libs:expr, $n:literal, ($($t:ty),*) -> $r:ty, ($($a:expr),*)) => {
        egl_call!($libs.gles, $n, ($($t),*) -> $r, ($($a),*))
    };
}
macro_rules! egl {
    ($libs:expr, $n:literal, ($($t:ty),*) -> $r:ty, ($($a:expr),*)) => {
        egl_call!($libs.egl, $n, ($($t),*) -> $r, ($($a),*))
    };
}

fn dlopen_path(env: &str, fallback: &str) -> *mut c_void {
    let path = std::env::var(env).unwrap_or_else(|_| fallback.to_string());
    let c = std::ffi::CString::new(path).unwrap();
    unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL) }
}

fn egl_state() -> Option<&'static Egl> {
    EGL
        .get_or_init(|| unsafe {
            let libs = Libs {
                egl: dlopen_path("OBSCURA_GL_EGL", "libEGL.so.1"),
                gles: dlopen_path("OBSCURA_GL_GLES", "libGLESv2.so.2"),
            };
            if libs.egl.is_null() || libs.gles.is_null() {
                return None;
            }
            // A surfaceless display needs eglGetPlatformDisplayEXT with the
            // MESA platform; fall back to the default display for drivers
            // (ANGLE) that take a default display but not that platform.
            let mut display: EGLDisplay = std::ptr::null_mut();
            let get_platform: Option<unsafe extern "C" fn(EGLenum, *mut c_void, *const EGLAttrib) -> EGLDisplay> = {
                let p = egl!(libs, "eglGetProcAddress", (*const c_char) -> *mut c_void,
                    (b"eglGetPlatformDisplayEXT\0".as_ptr().cast()));
                if p.is_null() { None } else { Some(std::mem::transmute(p)) }
            };
            if let Some(gpd) = get_platform {
                display = gpd(EGL_PLATFORM_SURFACELESS_MESA, std::ptr::null_mut(), std::ptr::null());
            }
            if display.is_null() {
                display = egl!(libs, "eglGetDisplay", (*mut c_void) -> EGLDisplay,
                    (std::ptr::null_mut()));
            }
            if display.is_null() {
                return None;
            }
            let mut major: EGLint = 0;
            let mut minor: EGLint = 0;
            if egl!(libs, "eglInitialize", (EGLDisplay, *mut EGLint, *mut EGLint) -> EGLBoolean,
                (display, &mut major, &mut minor)) == 0
            {
                return None;
            }
            egl!(libs, "eglBindAPI", (EGLenum) -> EGLBoolean, (EGL_OPENGL_ES_API));
            let attribs = [
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT,
                EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                EGL_NONE,
            ];
            let mut config: EGLConfig = std::ptr::null_mut();
            let mut count: EGLint = 0;
            if egl!(libs, "eglChooseConfig", (EGLDisplay, *const EGLint, *mut EGLConfig, EGLint, *mut EGLint) -> EGLBoolean,
                (display, attribs.as_ptr(), &mut config, 1, &mut count)) == 0
                || count == 0
            {
                return None;
            }
            Some(Egl { libs, display, config })
        })
        .as_ref()
}

pub struct GlCtx {
    context: EGLContext,
    surface: EGLSurface,
    default_fbo: GLuint,
    renderbuffer: GLuint,
}

#[derive(Default)]
pub struct GlContexts {
    map: HashMap<i32, GlCtx>,
    next: i32,
}

impl GlContexts {
    pub fn create(&mut self, width: i32, height: i32, es3: bool) -> i32 {
        let Some(egl) = egl_state() else { return -1 };
        let libs = &egl.libs;
        unsafe {
            let surf_attribs = [EGL_WIDTH, width.max(1), EGL_HEIGHT, height.max(1), EGL_NONE];
            let surface = egl!(libs, "eglCreatePbufferSurface",
                (EGLDisplay, EGLConfig, *const EGLint) -> EGLSurface,
                (egl.display, egl.config, surf_attribs.as_ptr()));
            if surface.is_null() {
                return -1;
            }
            let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, if es3 { 3 } else { 2 }, EGL_NONE];
            let context = egl!(libs, "eglCreateContext",
                (EGLDisplay, EGLConfig, EGLContext, *const EGLint) -> EGLContext,
                (egl.display, egl.config, std::ptr::null_mut(), ctx_attribs.as_ptr()));
            if context.is_null() {
                egl!(libs, "eglDestroySurface", (EGLDisplay, EGLSurface) -> EGLBoolean,
                    (egl.display, surface));
                return -1;
            }
            if egl!(libs, "eglMakeCurrent",
                (EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean,
                (egl.display, surface, surface, context)) == 0
            {
                egl!(libs, "eglDestroyContext", (EGLDisplay, EGLContext) -> EGLBoolean,
                    (egl.display, context));
                egl!(libs, "eglDestroySurface", (EGLDisplay, EGLSurface) -> EGLBoolean,
                    (egl.display, surface));
                return -1;
            }
            // The canvas backing store: a default FBO + RGBA renderbuffer that
            // `bindFramebuffer(0)` maps back onto.
            let mut fbo: GLuint = 0;
            let mut rbo: GLuint = 0;
            gl!(libs, "glGenFramebuffers", (GLsizei, *mut GLuint) -> (), (1, &mut fbo));
            gl!(libs, "glBindFramebuffer", (GLenum, GLuint) -> (), (GL_FRAMEBUFFER, fbo));
            gl!(libs, "glGenRenderbuffers", (GLsizei, *mut GLuint) -> (), (1, &mut rbo));
            gl!(libs, "glBindRenderbuffer", (GLenum, GLuint) -> (), (GL_RENDERBUFFER, rbo));
            gl!(libs, "glRenderbufferStorage",
                (GLenum, GLenum, GLsizei, GLsizei) -> (),
                (GL_RENDERBUFFER, GL_RGBA8, width.max(1), height.max(1)));
            gl!(libs, "glFramebufferRenderbuffer",
                (GLenum, GLenum, GLenum, GLuint) -> (),
                (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo));
            gl!(libs, "glViewport", (GLint, GLint, GLsizei, GLsizei) -> (),
                (0, 0, width.max(1), height.max(1)));
            self.next += 1;
            self.map.insert(self.next, GlCtx {
                context, surface, default_fbo: fbo, renderbuffer: rbo,
            });
            self.next
        }
    }

    fn make_current(&self, ctx: &GlCtx) -> bool {
        let egl = egl_state().unwrap();
        unsafe {
            egl!(egl.libs, "eglMakeCurrent",
                (EGLDisplay, EGLSurface, EGLSurface, EGLContext) -> EGLBoolean,
                (egl.display, ctx.surface, ctx.surface, ctx.context)) != 0
        }
    }

    pub fn destroy(&mut self, id: i32) {
        let Some(ctx) = self.map.remove(&id) else { return };
        let egl = egl_state().unwrap();
        unsafe {
            self.make_current(&ctx);
            let mut fbo = ctx.default_fbo;
            let mut rbo = ctx.renderbuffer;
            gl!(egl.libs, "glDeleteFramebuffers", (GLsizei, *const GLuint) -> (), (1, &mut fbo));
            gl!(egl.libs, "glDeleteRenderbuffers", (GLsizei, *const GLuint) -> (), (1, &mut rbo));
            egl!(egl.libs, "eglDestroyContext", (EGLDisplay, EGLContext) -> EGLBoolean,
                (egl.display, ctx.context));
            egl!(egl.libs, "eglDestroySurface", (EGLDisplay, EGLSurface) -> EGLBoolean,
                (egl.display, ctx.surface));
        }
    }

    pub fn call(&mut self, id: i32, payload: &serde_json::Value) -> serde_json::Value {
        use serde_json::json;
        let Some(egl) = egl_state() else { return json!(null) };
        let libs = &egl.libs;
        let Some(ctx) = self.map.get(&id) else { return json!(null) };
        if !self.make_current(ctx) {
            return json!(null);
        }
        let method = payload["m"].as_str().unwrap_or("");
        let a = payload["a"].as_array().cloned().unwrap_or_default();
        let n = |i: usize| a.get(i).and_then(|v| v.as_i64()).unwrap_or(0);
        let u = |i: usize| a.get(i).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let f = |i: usize| a.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let s = |i: usize| a.get(i).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let bytes = |i: usize| -> Vec<u8> {
            match a.get(i) {
                Some(v) => v.get("b")
                    .and_then(|b| b.as_str())
                    .and_then(|b| BASE64.decode(b).ok())
                    .unwrap_or_default(),
                None => Vec::new(),
            }
        };
        let f32s = |i: usize| -> Vec<f32> {
            a.get(i)
                .and_then(|v| v.as_array())
                .map(|xs| xs.iter().filter_map(|x| x.as_f64().map(|d| d as f32)).collect())
                .unwrap_or_default()
        };
        let i32s = |i: usize| -> Vec<i32> {
            a.get(i)
                .and_then(|v| v.as_array())
                .map(|xs| xs.iter().filter_map(|x| x.as_i64().map(|d| d as i32)).collect())
                .unwrap_or_default()
        };
        let u32s = |i: usize| -> Vec<u32> {
            a.get(i)
                .and_then(|v| v.as_array())
                .map(|xs| xs.iter().filter_map(|x| x.as_u64().map(|d| d as u32)).collect())
                .unwrap_or_default()
        };
        unsafe {
            match method {
                // --- state ---
                "viewport" => gl!(libs, "glViewport", (GLint, GLint, GLsizei, GLsizei) -> (),
                    (n(0) as i32, n(1) as i32, n(2) as i32, n(3) as i32)),
                "scissor" => gl!(libs, "glScissor", (GLint, GLint, GLsizei, GLsizei) -> (),
                    (n(0) as i32, n(1) as i32, n(2) as i32, n(3) as i32)),
                "lineWidth" => gl!(libs, "glLineWidth", (GLfloat) -> (), (f(0))),
                "polygonOffset" => gl!(libs, "glPolygonOffset", (GLfloat, GLfloat) -> (), (f(0), f(1))),
                "depthRange" => gl!(libs, "glDepthRangef", (GLfloat, GLfloat) -> (), (f(0), f(1))),
                "clearColor" => gl!(libs, "glClearColor", (GLfloat, GLfloat, GLfloat, GLfloat) -> (),
                    (f(0), f(1), f(2), f(3))),
                "clearDepth" => gl!(libs, "glClearDepthf", (GLfloat) -> (), (f(0))),
                "clearStencil" => gl!(libs, "glClearStencil", (GLint) -> (), (n(0) as i32)),
                "clear" => gl!(libs, "glClear", (GLbitfield) -> (), (u(0))),
                "enable" => gl!(libs, "glEnable", (GLenum) -> (), (u(0))),
                "disable" => gl!(libs, "glDisable", (GLenum) -> (), (u(0))),
                "blendColor" => gl!(libs, "glBlendColor", (GLfloat, GLfloat, GLfloat, GLfloat) -> (),
                    (f(0), f(1), f(2), f(3))),
                "blendEquation" => gl!(libs, "glBlendEquation", (GLenum) -> (), (u(0))),
                "blendEquationSeparate" => gl!(libs, "glBlendEquationSeparate", (GLenum, GLenum) -> (),
                    (u(0), u(1))),
                "blendFunc" => gl!(libs, "glBlendFunc", (GLenum, GLenum) -> (), (u(0), u(1))),
                "blendFuncSeparate" => gl!(libs, "glBlendFuncSeparate", (GLenum, GLenum, GLenum, GLenum) -> (),
                    (u(0), u(1), u(2), u(3))),
                "colorMask" => gl!(libs, "glColorMask", (GLboolean, GLboolean, GLboolean, GLboolean) -> (),
                    (u(0) as u8, u(1) as u8, u(2) as u8, u(3) as u8)),
                "cullFace" => gl!(libs, "glCullFace", (GLenum) -> (), (u(0))),
                "depthFunc" => gl!(libs, "glDepthFunc", (GLenum) -> (), (u(0))),
                "depthMask" => gl!(libs, "glDepthMask", (GLboolean) -> (), (u(0) as u8)),
                "frontFace" => gl!(libs, "glFrontFace", (GLenum) -> (), (u(0))),
                "hint" => gl!(libs, "glHint", (GLenum, GLenum) -> (), (u(0), u(1))),
                "sampleCoverage" => gl!(libs, "glSampleCoverage", (GLfloat, GLboolean) -> (),
                    (f(0), u(1) as u8)),
                "stencilFunc" => gl!(libs, "glStencilFunc", (GLenum, GLint, GLuint) -> (),
                    (u(0), n(1) as i32, u(2))),
                "stencilFuncSeparate" => gl!(libs, "glStencilFuncSeparate", (GLenum, GLenum, GLint, GLuint) -> (),
                    (u(0), u(1), n(2) as i32, u(3))),
                "stencilMask" => gl!(libs, "glStencilMask", (GLuint) -> (), (u(0))),
                "stencilMaskSeparate" => gl!(libs, "glStencilMaskSeparate", (GLenum, GLuint) -> (),
                    (u(0), u(1))),
                "stencilOp" => gl!(libs, "glStencilOp", (GLenum, GLenum, GLenum) -> (),
                    (u(0), u(1), u(2))),
                "stencilOpSeparate" => gl!(libs, "glStencilOpSeparate", (GLenum, GLenum, GLenum, GLenum) -> (),
                    (u(0), u(1), u(2), u(3))),
                "pixelStorei" => gl!(libs, "glPixelStorei", (GLenum, GLint) -> (),
                    (u(0), n(1) as i32)),
                "flush" => gl!(libs, "glFlush", () -> (), ()),
                "finish" => gl!(libs, "glFinish", () -> (), ()),
                // --- drawing ---
                "drawArrays" => gl!(libs, "glDrawArrays", (GLenum, GLint, GLsizei) -> (),
                    (u(0), n(1) as i32, n(2) as i32)),
                "drawElements" => gl!(libs, "glDrawElements", (GLenum, GLsizei, GLenum, GLintptr) -> (),
                    (u(0), n(1) as i32, u(2), n(3) as isize)),
                "readPixels" => {
                    let (x, y, w, h) = (n(0) as i32, n(1) as i32, n(2) as i32, n(3) as i32);
                    let (format, ty) = (u(4), u(5));
                    let mut buf = vec![0u8; (w.max(0) * h.max(0) * 4) as usize];
                    gl!(libs, "glReadPixels",
                        (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, *mut c_void) -> (),
                        (x, y, w, h, format, ty, buf.as_mut_ptr().cast()));
                    return json!({"b": BASE64.encode(&buf)});
                }
                // --- queries ---
                "getError" => return json!(gl!(libs, "glGetError", () -> GLenum, ())),
                "isEnabled" => return json!(gl!(libs, "glIsEnabled", (GLenum) -> GLboolean, (u(0))) != 0),
                "getParameter" => {
                    let pname = u(0);
                    return gl_parameter(libs, pname);
                }
                "getSupportedExtensions" => {
                    let p = gl!(libs, "glGetString", (GLenum) -> *const GLubyte, (GL_EXTENSIONS));
                    let list = if p.is_null() {
                        Vec::new()
                    } else {
                        std::ffi::CStr::from_ptr(p.cast())
                            .to_string_lossy()
                            .split_whitespace()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                    };
                    return json!(list);
                }
                "getShaderPrecisionFormat" => {
                    let mut range = [0i32; 2];
                    let mut precision = 0i32;
                    gl!(libs, "glGetShaderPrecisionFormat",
                        (GLenum, GLenum, *mut GLint, *mut GLint) -> (),
                        (u(0), u(1), range.as_mut_ptr(), &mut precision));
                    return json!({"mn": range[0], "mx": range[1], "p": precision});
                }
                "checkFramebufferStatus" => return json!(gl!(libs, "glCheckFramebufferStatus",
                    (GLenum) -> GLenum, (u(0)))),
                "getBufferParameter" => {
                    let mut v = 0i32;
                    gl!(libs, "glGetBufferParameteriv", (GLenum, GLenum, *mut GLint) -> (),
                        (u(0), u(1), &mut v));
                    return json!(v);
                }
                "getFramebufferAttachmentParameter" => {
                    let mut v = 0i32;
                    gl!(libs, "glGetFramebufferAttachmentParameteriv",
                        (GLenum, GLenum, GLenum, *mut GLint) -> (),
                        (u(0), u(1), u(2), &mut v));
                    return json!(v);
                }
                "getRenderbufferParameter" => {
                    let mut v = 0i32;
                    gl!(libs, "glGetRenderbufferParameteriv", (GLenum, GLenum, *mut GLint) -> (),
                        (u(0), u(1), &mut v));
                    return json!(v);
                }
                "getTexParameter" => {
                    let mut v = 0i32;
                    gl!(libs, "glGetTexParameteriv", (GLenum, GLenum, *mut GLint) -> (),
                        (u(0), u(1), &mut v));
                    return json!(v);
                }
                "getVertexAttrib" => {
                    let mut v = 0i32;
                    gl!(libs, "glGetVertexAttribiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), u(1), &mut v));
                    return json!(v);
                }
                "getVertexAttribOffset" => {
                    let mut v: *mut c_void = std::ptr::null_mut();
                    gl!(libs, "glGetVertexAttribPointerv", (GLuint, GLenum, *mut *mut c_void) -> (),
                        (u(0), u(1), &mut v));
                    return json!(v as isize);
                }
                "getProgramParameter" => {
                    let mut v = 0i32;
                    gl!(libs, "glGetProgramiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), u(1), &mut v));
                    return json!(v);
                }
                "getShaderParameter" => {
                    let mut v = 0i32;
                    gl!(libs, "glGetShaderiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), u(1), &mut v));
                    return json!(v);
                }
                "getProgramInfoLog" => {
                    let mut len = 0i32;
                    gl!(libs, "glGetProgramiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), 0x8B84, &mut len));
                    let mut buf = vec![0u8; len.max(0) as usize + 1];
                    let mut out = 0i32;
                    gl!(libs, "glGetProgramInfoLog",
                        (GLuint, GLsizei, *mut GLsizei, *mut GLchar) -> (),
                        (u(0), buf.len() as i32, &mut out, buf.as_mut_ptr().cast()));
                    buf.truncate(out.max(0) as usize);
                    return json!(String::from_utf8_lossy(&buf));
                }
                "getShaderInfoLog" => {
                    let mut len = 0i32;
                    gl!(libs, "glGetShaderiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), 0x8B84, &mut len));
                    let mut buf = vec![0u8; len.max(0) as usize + 1];
                    let mut out = 0i32;
                    gl!(libs, "glGetShaderInfoLog",
                        (GLuint, GLsizei, *mut GLsizei, *mut GLchar) -> (),
                        (u(0), buf.len() as i32, &mut out, buf.as_mut_ptr().cast()));
                    buf.truncate(out.max(0) as usize);
                    return json!(String::from_utf8_lossy(&buf));
                }
                "getShaderSource" => {
                    let mut len = 0i32;
                    gl!(libs, "glGetShaderiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), 0x8B88, &mut len));
                    let mut buf = vec![0u8; len.max(0) as usize + 1];
                    let mut out = 0i32;
                    gl!(libs, "glGetShaderSource",
                        (GLuint, GLsizei, *mut GLsizei, *mut GLchar) -> (),
                        (u(0), buf.len() as i32, &mut out, buf.as_mut_ptr().cast()));
                    buf.truncate(out.max(0) as usize);
                    return json!(String::from_utf8_lossy(&buf));
                }
                "getActiveAttrib" | "getActiveUniform" => {
                    let mut len = 0i32;
                    let pname = if method == "getActiveAttrib" { 0x8B89 } else { 0x8B8A };
                    gl!(libs, "glGetProgramiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), pname, &mut len));
                    let mut buf = vec![0u8; len.max(1) as usize + 1];
                    let mut size = 0i32;
                    let mut ty = 0u32;
                    let mut out = 0i32;
                    if method == "getActiveAttrib" {
                        gl!(libs, "glGetActiveAttrib",
                            (GLuint, GLuint, GLsizei, *mut GLsizei, *mut GLint, *mut GLenum, *mut GLchar) -> (),
                            (u(0), u(1), buf.len() as i32, &mut out, &mut size, &mut ty,
                                buf.as_mut_ptr().cast()));
                    } else {
                        gl!(libs, "glGetActiveUniform",
                            (GLuint, GLuint, GLsizei, *mut GLsizei, *mut GLint, *mut GLenum, *mut GLchar) -> (),
                            (u(0), u(1), buf.len() as i32, &mut out, &mut size, &mut ty,
                                buf.as_mut_ptr().cast()));
                    }
                    buf.truncate(out.max(0) as usize);
                    return json!({"n": String::from_utf8_lossy(&buf), "s": size, "t": ty});
                }
                "getAttribLocation" => return json!(gl!(libs, "glGetAttribLocation",
                    (GLuint, *const GLchar) -> GLint,
                    (u(0), std::ffi::CString::new(s(1)).unwrap().as_ptr()))),
                "getUniformLocation" => return json!(gl!(libs, "glGetUniformLocation",
                    (GLuint, *const GLchar) -> GLint,
                    (u(0), std::ffi::CString::new(s(1)).unwrap().as_ptr()))),
                "getUniform" => {
                    let mut v = [0f32; 16];
                    gl!(libs, "glGetUniformfv", (GLuint, GLint, *mut GLfloat) -> (),
                        (u(0), n(1) as i32, v.as_mut_ptr()));
                    return json!(v);
                }
                "drawArraysInstanced" => gl!(libs, "glDrawArraysInstanced",
                    (GLenum, GLint, GLsizei, GLsizei) -> (),
                    (u(0), n(1) as i32, n(2) as i32, n(3) as i32)),
                "drawElementsInstanced" => gl!(libs, "glDrawElementsInstanced",
                    (GLenum, GLsizei, GLenum, GLintptr, GLsizei) -> (),
                    (u(0), n(1) as i32, u(2), n(3) as isize, n(4) as i32)),
                "bindAttribLocation" => gl!(libs, "glBindAttribLocation",
                    (GLuint, GLuint, *const GLchar) -> (),
                    (u(0), u(1), std::ffi::CString::new(s(2)).unwrap().as_ptr())),
                "getAttachedShaders" => {
                    let mut count = 0i32;
                    gl!(libs, "glGetProgramiv", (GLuint, GLenum, *mut GLint) -> (),
                        (u(0), 0x8B85, &mut count));
                    let mut shaders = vec![0u32; count.max(0) as usize];
                    let mut out = 0i32;
                    gl!(libs, "glGetAttachedShaders",
                        (GLuint, GLsizei, *mut GLsizei, *mut GLuint) -> (),
                        (u(0), count, &mut out, shaders.as_mut_ptr()));
                    shaders.truncate(out.max(0) as usize);
                    return json!(shaders);
                }
                // --- shaders / programs ---
                "createShader" => return json!(gl!(libs, "glCreateShader", (GLenum) -> GLuint, (u(0)))),
                "shaderSource" => {
                    let src = std::ffi::CString::new(s(1)).unwrap();
                    let ptr = src.as_ptr();
                    gl!(libs, "glShaderSource", (GLuint, GLsizei, *const *const GLchar, *const GLint) -> (),
                        (u(0), 1, &ptr, std::ptr::null()));
                }
                "compileShader" => gl!(libs, "glCompileShader", (GLuint) -> (), (u(0))),
                "deleteShader" => gl!(libs, "glDeleteShader", (GLuint) -> (), (u(0))),
                "isShader" => return json!(gl!(libs, "glIsShader", (GLuint) -> GLboolean, (u(0))) != 0),
                "createProgram" => return json!(gl!(libs, "glCreateProgram", () -> GLuint, ())),
                "attachShader" => gl!(libs, "glAttachShader", (GLuint, GLuint) -> (), (u(0), u(1))),
                "detachShader" => gl!(libs, "glDetachShader", (GLuint, GLuint) -> (), (u(0), u(1))),
                "linkProgram" => gl!(libs, "glLinkProgram", (GLuint) -> (), (u(0))),
                "useProgram" => gl!(libs, "glUseProgram", (GLuint) -> (), (u(0))),
                "validateProgram" => gl!(libs, "glValidateProgram", (GLuint) -> (), (u(0))),
                "deleteProgram" => gl!(libs, "glDeleteProgram", (GLuint) -> (), (u(0))),
                "isProgram" => return json!(gl!(libs, "glIsProgram", (GLuint) -> GLboolean, (u(0))) != 0),
                // --- buffers ---
                "genBuffers" => {
                    let mut v = 0u32;
                    gl!(libs, "glGenBuffers", (GLsizei, *mut GLuint) -> (), (1, &mut v));
                    return json!(v);
                }
                "bindBuffer" => gl!(libs, "glBindBuffer", (GLenum, GLuint) -> (), (u(0), u(1))),
                "deleteBuffer" => {
                    let v = u(0);
                    gl!(libs, "glDeleteBuffers", (GLsizei, *const GLuint) -> (), (1, &v));
                }
                "isBuffer" => return json!(gl!(libs, "glIsBuffer", (GLuint) -> GLboolean, (u(0))) != 0),
                "bufferData" => {
                    let data = bytes(1);
                    if data.is_empty() && a.get(1).map(|v| v.get("n")).flatten().is_some() {
                        let size = a[1]["n"].as_i64().unwrap_or(0) as isize;
                        gl!(libs, "glBufferData", (GLenum, GLsizeiptr, *const c_void, GLenum) -> (),
                            (u(0), size, std::ptr::null(), u(2)));
                    } else {
                        gl!(libs, "glBufferData", (GLenum, GLsizeiptr, *const c_void, GLenum) -> (),
                            (u(0), data.len() as isize, data.as_ptr().cast(), u(2)));
                    }
                }
                "bufferSubData" => {
                    let data = bytes(2);
                    gl!(libs, "glBufferSubData", (GLenum, GLintptr, GLsizeiptr, *const c_void) -> (),
                        (u(0), n(1) as isize, data.len() as isize, data.as_ptr().cast()));
                }
                // --- textures ---
                "genTextures" => {
                    let mut v = 0u32;
                    gl!(libs, "glGenTextures", (GLsizei, *mut GLuint) -> (), (1, &mut v));
                    return json!(v);
                }
                "activeTexture" => gl!(libs, "glActiveTexture", (GLenum) -> (), (u(0))),
                "bindTexture" => gl!(libs, "glBindTexture", (GLenum, GLuint) -> (), (u(0), u(1))),
                "deleteTexture" => {
                    let v = u(0);
                    gl!(libs, "glDeleteTextures", (GLsizei, *const GLuint) -> (), (1, &v));
                }
                "isTexture" => return json!(gl!(libs, "glIsTexture", (GLuint) -> GLboolean, (u(0))) != 0),
                "texImage2D" => {
                    let data = bytes(8);
                    let ptr = if data.is_empty() { std::ptr::null() } else { data.as_ptr().cast() };
                    gl!(libs, "glTexImage2D",
                        (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, *const c_void) -> (),
                        (u(0), n(1) as i32, n(2) as i32, n(3) as i32, n(4) as i32, n(5) as i32,
                            u(6), u(7), ptr));
                }
                "texParameteri" => gl!(libs, "glTexParameteri", (GLenum, GLenum, GLint) -> (),
                    (u(0), u(1), n(2) as i32)),
                "texParameterf" => gl!(libs, "glTexParameterf", (GLenum, GLenum, GLfloat) -> (),
                    (u(0), u(1), f(2))),
                "generateMipmap" => gl!(libs, "glGenerateMipmap", (GLenum) -> (), (u(0))),
                // --- renderbuffers ---
                "genRenderbuffers" => {
                    let mut v = 0u32;
                    gl!(libs, "glGenRenderbuffers", (GLsizei, *mut GLuint) -> (), (1, &mut v));
                    return json!(v);
                }
                "bindRenderbuffer" => gl!(libs, "glBindRenderbuffer", (GLenum, GLuint) -> (),
                    (u(0), u(1))),
                "renderbufferStorage" => gl!(libs, "glRenderbufferStorage",
                    (GLenum, GLenum, GLsizei, GLsizei) -> (),
                    (u(0), u(1), n(2) as i32, n(3) as i32)),
                "deleteRenderbuffer" => {
                    let v = u(0);
                    gl!(libs, "glDeleteRenderbuffers", (GLsizei, *const GLuint) -> (), (1, &v));
                }
                "isRenderbuffer" => return json!(gl!(libs, "glIsRenderbuffer",
                    (GLuint) -> GLboolean, (u(0))) != 0),
                // --- framebuffers ---
                "genFramebuffers" => {
                    let mut v = 0u32;
                    gl!(libs, "glGenFramebuffers", (GLsizei, *mut GLuint) -> (), (1, &mut v));
                    return json!(v);
                }
                "bindFramebuffer" => {
                    // The page's default framebuffer is the canvas backing
                    // store created at context init.
                    let target = u(0);
                    let handle = u(1);
                    gl!(libs, "glBindFramebuffer", (GLenum, GLuint) -> (),
                        (target, if handle == 0 { ctx.default_fbo } else { handle }));
                }
                "framebufferRenderbuffer" => gl!(libs, "glFramebufferRenderbuffer",
                    (GLenum, GLenum, GLenum, GLuint) -> (), (u(0), u(1), u(2), u(3))),
                "framebufferTexture2D" => gl!(libs, "glFramebufferTexture2D",
                    (GLenum, GLenum, GLenum, GLuint, GLint) -> (),
                    (u(0), u(1), u(2), u(3), n(4) as i32)),
                "deleteFramebuffer" => {
                    let v = u(0);
                    gl!(libs, "glDeleteFramebuffers", (GLsizei, *const GLuint) -> (), (1, &v));
                }
                "isFramebuffer" => return json!(gl!(libs, "glIsFramebuffer",
                    (GLuint) -> GLboolean, (u(0))) != 0),
                // --- vertex attribs ---
                "vertexAttribPointer" => gl!(libs, "glVertexAttribPointer",
                    (GLuint, GLint, GLenum, GLboolean, GLsizei, GLintptr) -> (),
                    (u(0), n(1) as i32, u(2), u(3) as u8, n(4) as i32, n(5) as isize)),
                "enableVertexAttribArray" => gl!(libs, "glEnableVertexAttribArray", (GLuint) -> (),
                    (u(0))),
                "disableVertexAttribArray" => gl!(libs, "glDisableVertexAttribArray", (GLuint) -> (),
                    (u(0))),
                "vertexAttrib1f" => gl!(libs, "glVertexAttrib1f", (GLuint, GLfloat) -> (),
                    (u(0), f(1))),
                "vertexAttrib2f" => gl!(libs, "glVertexAttrib2f", (GLuint, GLfloat, GLfloat) -> (),
                    (u(0), f(1), f(2))),
                "vertexAttrib3f" => gl!(libs, "glVertexAttrib3f",
                    (GLuint, GLfloat, GLfloat, GLfloat) -> (), (u(0), f(1), f(2), f(3))),
                "vertexAttrib4f" => gl!(libs, "glVertexAttrib4f",
                    (GLuint, GLfloat, GLfloat, GLfloat, GLfloat) -> (),
                    (u(0), f(1), f(2), f(3), f(4))),
                "vertexAttrib1fv" => {
                    let v = f32s(1);
                    gl!(libs, "glVertexAttrib1fv", (GLuint, *const GLfloat) -> (),
                        (u(0), v.as_ptr()));
                }
                "vertexAttrib2fv" => {
                    let v = f32s(1);
                    gl!(libs, "glVertexAttrib2fv", (GLuint, *const GLfloat) -> (),
                        (u(0), v.as_ptr()));
                }
                "vertexAttrib3fv" => {
                    let v = f32s(1);
                    gl!(libs, "glVertexAttrib3fv", (GLuint, *const GLfloat) -> (),
                        (u(0), v.as_ptr()));
                }
                "vertexAttrib4fv" => {
                    let v = f32s(1);
                    gl!(libs, "glVertexAttrib4fv", (GLuint, *const GLfloat) -> (),
                        (u(0), v.as_ptr()));
                }
                "vertexAttribDivisor" => gl!(libs, "glVertexAttribDivisor", (GLuint, GLuint) -> (),
                    (u(0), u(1))),
                // --- uniforms ---
                "uniform1i" => gl!(libs, "glUniform1i", (GLint, GLint) -> (),
                    (n(0) as i32, n(1) as i32)),
                "uniform2i" => gl!(libs, "glUniform2i", (GLint, GLint, GLint) -> (),
                    (n(0) as i32, n(1) as i32, n(2) as i32)),
                "uniform3i" => gl!(libs, "glUniform3i", (GLint, GLint, GLint, GLint) -> (),
                    (n(0) as i32, n(1) as i32, n(2) as i32, n(3) as i32)),
                "uniform4i" => gl!(libs, "glUniform4i", (GLint, GLint, GLint, GLint, GLint) -> (),
                    (n(0) as i32, n(1) as i32, n(2) as i32, n(3) as i32, n(4) as i32)),
                "uniform1f" => gl!(libs, "glUniform1f", (GLint, GLfloat) -> (),
                    (n(0) as i32, f(1))),
                "uniform2f" => gl!(libs, "glUniform2f", (GLint, GLfloat, GLfloat) -> (),
                    (n(0) as i32, f(1), f(2))),
                "uniform3f" => gl!(libs, "glUniform3f", (GLint, GLfloat, GLfloat, GLfloat) -> (),
                    (n(0) as i32, f(1), f(2), f(3))),
                "uniform4f" => gl!(libs, "glUniform4f", (GLint, GLfloat, GLfloat, GLfloat, GLfloat) -> (),
                    (n(0) as i32, f(1), f(2), f(3), f(4))),
                "uniform1iv" => {
                    let v = i32s(1);
                    gl!(libs, "glUniform1iv", (GLint, GLsizei, *const GLint) -> (),
                        (n(0) as i32, v.len() as i32, v.as_ptr()));
                }
                "uniform2iv" => {
                    let v = i32s(1);
                    gl!(libs, "glUniform2iv", (GLint, GLsizei, *const GLint) -> (),
                        (n(0) as i32, v.len() as i32 / 2, v.as_ptr()));
                }
                "uniform3iv" => {
                    let v = i32s(1);
                    gl!(libs, "glUniform3iv", (GLint, GLsizei, *const GLint) -> (),
                        (n(0) as i32, v.len() as i32 / 3, v.as_ptr()));
                }
                "uniform4iv" => {
                    let v = i32s(1);
                    gl!(libs, "glUniform4iv", (GLint, GLsizei, *const GLint) -> (),
                        (n(0) as i32, v.len() as i32 / 4, v.as_ptr()));
                }
                "uniform1fv" => {
                    let v = f32s(1);
                    gl!(libs, "glUniform1fv", (GLint, GLsizei, *const GLfloat) -> (),
                        (n(0) as i32, v.len() as i32, v.as_ptr()));
                }
                "uniform2fv" => {
                    let v = f32s(1);
                    gl!(libs, "glUniform2fv", (GLint, GLsizei, *const GLfloat) -> (),
                        (n(0) as i32, v.len() as i32 / 2, v.as_ptr()));
                }
                "uniform3fv" => {
                    let v = f32s(1);
                    gl!(libs, "glUniform3fv", (GLint, GLsizei, *const GLfloat) -> (),
                        (n(0) as i32, v.len() as i32 / 3, v.as_ptr()));
                }
                "uniform4fv" => {
                    let v = f32s(1);
                    gl!(libs, "glUniform4fv", (GLint, GLsizei, *const GLfloat) -> (),
                        (n(0) as i32, v.len() as i32 / 4, v.as_ptr()));
                }
                "uniformMatrix2fv" => {
                    let v = f32s(2);
                    gl!(libs, "glUniformMatrix2fv", (GLint, GLsizei, GLboolean, *const GLfloat) -> (),
                        (n(0) as i32, v.len() as i32 / 4, 0, v.as_ptr()));
                }
                "uniformMatrix3fv" => {
                    let v = f32s(2);
                    gl!(libs, "glUniformMatrix3fv", (GLint, GLsizei, GLboolean, *const GLfloat) -> (),
                        (n(0) as i32, v.len() as i32 / 9, 0, v.as_ptr()));
                }
                "uniformMatrix4fv" => {
                    let v = f32s(2);
                    gl!(libs, "glUniformMatrix4fv", (GLint, GLsizei, GLboolean, *const GLfloat) -> (),
                        (n(0) as i32, v.len() as i32 / 16, 0, v.as_ptr()));
                }
                // --- ES3 vertex arrays ---
                "genVertexArrays" => {
                    let mut v = 0u32;
                    gl!(libs, "glGenVertexArrays", (GLsizei, *mut GLuint) -> (), (1, &mut v));
                    return json!(v);
                }
                "bindVertexArray" => gl!(libs, "glBindVertexArray", (GLuint) -> (), (u(0))),
                "deleteVertexArray" => {
                    let v = u(0);
                    gl!(libs, "glDeleteVertexArrays", (GLsizei, *const GLuint) -> (), (1, &v));
                }
                "isVertexArray" => return json!(gl!(libs, "glIsVertexArray",
                    (GLuint) -> GLboolean, (u(0))) != 0),
                "readBuffer" => gl!(libs, "glReadBuffer", (GLenum) -> (), (u(0))),
                "drawBuffers" => {
                    let v = u32s(0);
                    gl!(libs, "glDrawBuffers", (GLsizei, *const GLenum) -> (),
                        (v.len() as i32, v.as_ptr()));
                }
                _ => {}
            }
        }
        json!(null)
    }
}

fn gl_parameter(libs: &Libs, pname: GLenum) -> serde_json::Value {
    use serde_json::json;
    const STRING_PARAMS: &[GLenum] = &[0x1F00, 0x1F01, 0x1F02, 0x8B8C];
    // COLOR_CLEAR_VALUE, LINE_WIDTH, DEPTH_RANGE, BLEND_COLOR,
    // DEPTH_CLEAR_VALUE, ALIASED_POINT_SIZE_RANGE, ALIASED_LINE_WIDTH_RANGE.
    const FLOAT_PARAMS: &[GLenum] = &[
        0x0C22, 0x0B21, 0x0B70, 0x8005, 0x0B73, 0x846D, 0x846E,
    ];
    unsafe {
        if STRING_PARAMS.contains(&pname) {
            let p = gl!(libs, "glGetString", (GLenum) -> *const GLubyte, (pname));
            return if p.is_null() {
                json!(null)
            } else {
                json!(std::ffi::CStr::from_ptr(p.cast()).to_string_lossy())
            };
        }
        if FLOAT_PARAMS.contains(&pname) {
            let mut v = [0f32; 4];
            gl!(libs, "glGetFloatv", (GLenum, *mut GLfloat) -> (), (pname, v.as_mut_ptr()));
            return json!(v);
        }
        // Booleans (COLOR_WRITEMASK, etc.) read back as 0/1 ints; the JS side
        // maps known boolean pnames itself.
        let mut v = [0i32; 16];
        gl!(libs, "glGetIntegerv", (GLenum, *mut GLint) -> (), (pname, v.as_mut_ptr()));
        json!(v)
    }
}
