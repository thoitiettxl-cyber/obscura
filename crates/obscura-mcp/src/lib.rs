// The tools/list JSON literal expanded by serde_json::json! is large
// enough now (32 tool definitions) that the default macro recursion
// limit (128) overflows. Bumping for this crate only.
#![recursion_limit = "512"]

pub mod http;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
#[cfg(feature = "render")]
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use obscura_browser::{BrowserContext, Page};
use obscura_browser::page::PendingNavigationOutcome;
use obscura_dom::NodeId;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Cap on text returned to the agent unless the caller passes a larger
/// `max_chars`. Agents waste context on multi-KB raw page dumps; this
/// keeps a single tool call from burning a window's worth of tokens.
/// Override via tool args.
const DEFAULT_TEXT_LIMIT: usize = 4000;

#[derive(Deserialize)]
struct RpcMessage {
    #[allow(dead_code)]
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

impl RpcResponse {
    fn ok(id: Value, result: Value) -> Self {
        RpcResponse { jsonrpc: "2.0", id, result: Some(result), error: None }
    }

    fn err(id: Value, code: i32, message: impl Into<String>) -> Self {
        RpcResponse { jsonrpc: "2.0", id, result: None, error: Some(RpcError { code, message: message.into() }) }
    }
}

struct ElementRef {
    node_id: NodeId,
    // Native consumers may free an arena slot and reuse its numeric id.
    allocation_generation: u64,
}

pub struct BrowserState {
    /// Open tabs keyed by tab_id (e.g. "tab-1"). BTreeMap so list ordering
    /// is stable across calls (agents reason about "tab #2" deterministically).
    tabs: std::collections::BTreeMap<String, Page>,
    /// The tab id every tool call operates on. None means there are no
    /// open tabs; the next page_mut() call creates one.
    active_tab: Option<String>,
    tab_counter: u32,
    context: Arc<BrowserContext>,
    user_agent: Option<String>,
    operator_network_hints: bool,
    console_messages: Vec<String>,
    /// Element-ref table from the last `browser_snapshot` on the ACTIVE
    /// tab. Agents click / fill / type by `ref` (e.g. `"e3"`) instead of
    /// guessing a CSS selector. Refs are stable within a snapshot; the
    /// table is wiped on document navigation attempts / tab switches and
    /// refilled on the next snapshot call.
    interactive_refs: HashMap<String, ElementRef>,
    ref_generation: u64,
}

impl BrowserState {
    pub fn new(proxy: Option<String>, user_agent: Option<String>, stealth: bool) -> Self {
        // `--storage-dir` is a global flag, so the profile arrives through the
        // environment rather than through every constructor call. Without this
        // the MCP path silently ignored the flag and persisted nothing, cookies
        // included.
        let storage_dir = std::env::var_os("OBSCURA_STORAGE_DIR").map(std::path::PathBuf::from);
        let context = BrowserContext::with_storage_full(
            "mcp".to_string(),
            proxy,
            stealth,
            None,
            storage_dir,
        );
        BrowserState {
            tabs: std::collections::BTreeMap::new(),
            active_tab: None,
            tab_counter: 0,
            context: Arc::new(context),
            user_agent,
            operator_network_hints: false,
            console_messages: Vec::new(),
            interactive_refs: HashMap::new(),
            ref_generation: 0,
        }
    }

    /// Make sure there is at least one tab and return a &mut to the
    /// active tab's Page. Auto-creates a default tab if none exist so
    /// every legacy single-page tool continues to work without
    /// requiring an explicit browser_tab_new.
    fn page_mut(&mut self) -> &mut Page {
        if self.active_tab.is_none() {
            self.tab_counter += 1;
            let id = format!("tab-{}", self.tab_counter);
            let page = Page::new("mcp-page".to_string(), self.context.clone());
            page.set_console_messages_enabled(true);
            self.tabs.insert(id.clone(), page);
            self.active_tab = Some(id);
        }
        let id = self.active_tab.as_ref().unwrap().clone();
        self.activate(&id);
        self.tabs.get_mut(&id).expect("active tab must exist")
    }

    fn new_tab(&mut self) -> String {
        self.tab_counter += 1;
        let id = format!("tab-{}", self.tab_counter);
        let page = Page::new(format!("mcp-{id}"), self.context.clone());
        page.set_console_messages_enabled(true);
        self.tabs.insert(id.clone(), page);
        self.active_tab = Some(id.clone());
        self.interactive_refs.clear();
        id
    }

    /// Scoped isolate entries allow inactive tabs to keep their realms, as in CDP.
    fn activate(&mut self, tab_id: &str) {
        if let Some(page) = self.tabs.get_mut(tab_id) {
            if !page.has_js() { page.resume_js(); }
        }
    }

    /// Retire frame handles before disposing their owning runtime.
    fn close_tab(&mut self, tab_id: &str) -> bool {
        if let Some(page) = self.tabs.get_mut(tab_id) {
            page.suspend_js();
        }
        self.tabs.remove(tab_id).is_some()
    }

    fn has_active_page_runtime(&self) -> bool {
        self.active_tab
            .as_ref()
            .and_then(|tab_id| self.tabs.get(tab_id))
            .is_some_and(Page::has_js)
    }

    /// Advance the active page by one wake-driven browser task and immediately
    /// consume any navigation that task queued. MCP owns its pages continuously,
    /// so leaving either half for the next tool call strands timers, fetches,
    /// and location/form/click navigations while the transport waits on stdin.
    async fn advance_active_page_tasks(&mut self) -> Result<bool, String> {
        let reached_idle = self.page_mut().run_autonomous_event_loop_turn().await?;
        self.finish_active_page_turn(reached_idle).await
    }

    // Complete navigation outside the cancelable wait: it consumes pending state before I/O.
    async fn finish_active_page_turn(&mut self, reached_idle: bool) -> Result<bool, String> {
        let navigation = self.page_mut()
            .process_pending_navigation_outcome()
            .await
            .inspect_err(|_| self.interactive_refs.clear())
            .map_err(|error| error.to_string())?;
        if navigation == PendingNavigationOutcome::Document {
            self.interactive_refs.clear();
        }
        Ok(reached_idle && navigation == PendingNavigationOutcome::None)
    }

    /// Consume a navigation that a synthesized interaction just queued,
    /// before the tool replies.
    ///
    /// A click on a submit button, an Enter keypress, or an evaluated
    /// `location.href` assignment does not issue a request by itself. It runs
    /// the page's own glue and leaves the navigation as pending state, which
    /// becomes a request only when a driving layer converts it. The CDP path
    /// already converts it, in `Input.dispatchMouseEvent` and after
    /// `Runtime.evaluate`, which is why the same click POSTs over CDP and
    /// does nothing over MCP (#618).
    ///
    /// [`Self::advance_active_page_tasks`] cannot cover this. It is armed
    /// after every dispatch, but it sits in a `biased` select behind
    /// `read_line`, so a client that sends its next tool call immediately,
    /// which an agent does, wins that race every time. The reply is also
    /// written before any pump turn could run, so the tool would answer
    /// "Clicked" while the request has not left the process.
    async fn settle_synthetic_navigation(&mut self) -> Result<(), String> {
        let navigation = self
            .page_mut()
            .process_pending_navigation_outcome()
            .await
            .inspect_err(|_| self.interactive_refs.clear())
            .map_err(|error| error.to_string())?;
        if navigation == PendingNavigationOutcome::Document {
            // The ref table names elements in a document that has gone away.
            self.interactive_refs.clear();
        }
        if navigation != PendingNavigationOutcome::None {
            // One slice on the landed document, so a tool that reads the URL
            // or the text next sees the new page rather than an empty one.
            self.page_mut()
                .run_autonomous_event_loop_turn()
                .await
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Resolve `ref=eN` to a CSS selector that uniquely targets the
    /// element. Snapshot writes `data-obscura-ref="eN"` onto every
    /// interactable, so the attribute survives across calls as long as
    /// the page isn't re-rendered without it. Returns `Err` if the ref
    /// hasn't been registered (caller must call browser_snapshot first).
    fn ref_to_selector(&self, r: &str) -> Result<String, String> {
        let reference = self.interactive_refs.get(r).ok_or_else(|| format!(
                "unknown ref '{r}'; call browser_snapshot first to refresh the ref table"
            ))?;
        let selector = format!("[data-obscura-ref=\"{r}\"]");
        let same_node = self.active_tab.as_ref()
            .and_then(|id| self.tabs.get(id))
            .and_then(|page| page.with_dom(|dom| {
                dom.node_allocation_generation(reference.node_id) == Some(reference.allocation_generation)
                    && dom.query_selector(&selector).ok().flatten() == Some(reference.node_id)
            }))
            .unwrap_or(false);
        if !same_node {
            return Err(format!("stale ref '{r}'; call browser_snapshot to refresh the ref table"));
        }
        Ok(selector)
    }
}

pub(crate) async fn dispatch(method: &str, id: Value, params: &Value, state: &mut BrowserState) -> RpcResponse {
    match method {
        "initialize" => handle_initialize(id, params),
        "ping" => RpcResponse::ok(id, json!({})),
        "tools/list" => handle_tools_list(id),
        "tools/call" => handle_tool_call(id, params, state).await,
        "resources/list" => RpcResponse::ok(id, json!({"resources": []})),
        "prompts/list" => RpcResponse::ok(id, json!({"prompts": []})),
        _ => RpcResponse::err(id, -32601, format!("Unknown method: {method}")),
    }
}

pub async fn run(proxy: Option<String>, user_agent: Option<String>, stealth: bool) -> Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut writer = stdout;

    let mut state = BrowserState::new(proxy, user_agent, stealth);
    state.operator_network_hints = true;
    // Cookies and Web Storage are written on every navigation; this covers the
    // writes an agent makes after a page has settled, on the way out.
    let mut runtime_pump_armed = false;

    loop {
        // MCP stdio transport: newline-delimited JSON (one message per line)
        let mut line = String::new();
        let n = if runtime_pump_armed {
            tokio::select! {
                biased;
                read = reader.read_line(&mut line) => Some(read?),
                pump_result = state.page_mut().run_autonomous_event_loop_turn() => {
                    let pump_result = match pump_result {
                        Ok(reached_idle) => state.finish_active_page_turn(reached_idle).await,
                        Err(error) => Err(error),
                    };
                    match pump_result {
                        Ok(reached_idle) => runtime_pump_armed = !reached_idle,
                        Err(error) => {
                            runtime_pump_armed = false;
                            eprintln!("MCP page task failed: {error}");
                        }
                    }
                    None
                }
            }
        } else {
            Some(reader.read_line(&mut line).await?)
        };
        let Some(n) = n else {
            continue;
        };
        if n == 0 {
            state.context.save_storage();
            return Ok(());
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: RpcMessage = match serde_json::from_str(trimmed) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Notifications (no id) need no response
        if msg.id.is_none() {
            continue;
        }

        let id = msg.id.clone().unwrap_or(Value::Null);
        let response = dispatch(&msg.method, id, &msg.params, &mut state).await;
        runtime_pump_armed = state.has_active_page_runtime();

        let mut body = serde_json::to_string(&response)?;
        body.push('\n');
        writer.write_all(body.as_bytes()).await?;
        writer.flush().await?;
    }
}

fn handle_initialize(id: Value, params: &Value) -> RpcResponse {
    let _client_version = params.get("protocolVersion").and_then(Value::as_str).unwrap_or("");
    RpcResponse::ok(id, json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "obscura-mcp",
            // Same build version the CLI reports (tag-derived at release time),
            // so MCP clients see the version the binary was actually cut from.
            "version": env!("OBSCURA_BUILD_VERSION")
        }
    }))
}

fn handle_tools_list(id: Value) -> RpcResponse {
    #[allow(unused_mut)]
    let mut tools = json!([
            {
                "name": "browser_navigate",
                "description": "Navigate to a URL and wait for the page to load",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "URL to navigate to" },
                        "waitUntil": {
                            "type": "string",
                            "enum": ["load", "domcontentloaded", "networkidle0"],
                            "description": "Navigation wait condition (default: load)"
                        },
                        "dismissConsent": {
                            "type": "boolean",
                            "description": "Opt-in: after load, click a visible button of a known consent manager (OneTrust, Cookiebot, Quantcast) to dismiss its banner (default: false)"
                        },
                        "macro": { "type": "string", "description": "Search macro expanding to a URL (e.g. @google_search); combined with `query`. url takes precedence when a macro yields nothing" },
                        "query": { "type": "string", "description": "Query string interpolated into the search macro" }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "browser_snapshot",
                "description": "Get URL, title, actionable element references, control states, and visible body text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_chars": { "type": "number", "minimum": 0, "description": "Character budget shared by whole control descriptions and visible body text (default: 4000); URL and title are separately capped at 256 characters each. When the content exceeds the budget the last ~12% is kept as a tail so pagination/nav links survive" },
                        "offset": { "type": "number", "minimum": 0, "description": "Character offset into the full snapshot for paging (default 0); the truncation marker advertises the next offset" }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "browser_click",
                "description": "Click an element. Pass `ref` (preferred, from browser_snapshot / browser_interactive_elements) OR a `selector`.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ref": { "type": "string", "description": "Element ref like 'e3' from a recent snapshot" },
                        "selector": { "type": "string", "description": "CSS selector (fallback if ref unavailable)" }
                    }
                }
            },
            {
                "name": "browser_fill",
                "description": "Set the value of an input element. Pass `ref` (preferred) OR `selector`.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ref": { "type": "string" },
                        "selector": { "type": "string" },
                        "value": { "type": "string", "description": "Value to set" }
                    },
                    "required": ["value"]
                }
            },
            {
                "name": "browser_type",
                "description": "Type text into an input element (appends to existing value). Pass `ref` (preferred) OR `selector`.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ref": { "type": "string" },
                        "selector": { "type": "string" },
                        "text": { "type": "string", "description": "Text to type" }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "browser_press_key",
                "description": "Dispatch a keyboard event on an element or the document",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "key": { "type": "string", "description": "Key name (e.g. Enter, Tab, Escape)" },
                        "selector": { "type": "string", "description": "CSS selector (optional, defaults to document)" }
                    },
                    "required": ["key"]
                }
            },
            {
                "name": "browser_select_option",
                "description": "Select an option from a <select> element",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "CSS selector of the <select> element" },
                        "value": { "type": "string", "description": "Value or text of the option to select" }
                    },
                    "required": ["selector", "value"]
                }
            },
            {
                "name": "browser_evaluate",
                "description": "Evaluate a JavaScript expression in the page context and return the result",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": { "type": "string", "description": "JavaScript expression to evaluate; a promise it returns is awaited" },
                        "timeoutMs": { "type": "number", "description": "How long to wait for a returned promise, in milliseconds (default: 5000)" }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "browser_wait_for_function",
                "description": "Wait until a JavaScript expression returns true, polling the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "function": { "type": "string", "description": "JavaScript expression that must become truthy" },
                        "timeout": { "type": "number", "description": "Timeout in milliseconds (default: 30000)" },
                        "pollIntervalMs": { "type": "number", "description": "Poll interval in milliseconds (default: 100)" }
                    },
                    "required": ["function"]
                }
            },
            {
                "name": "browser_wait_for",
                "description": "Wait for a CSS selector to appear in the DOM",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string", "description": "CSS selector to wait for" },
                        "timeout": { "type": "number", "description": "Timeout in seconds (default: 30)" }
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "browser_network_requests",
                "description": "Return the list of network requests made by the current page",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "browser_console_messages",
                "description": "Return the console messages logged by the current page",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "browser_close",
                "description": "Close the current browser page and reset state",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "browser_markdown",
                "description": "Extract the current page as Markdown (headings, paragraphs, lists, links, code blocks). Use this instead of browser_snapshot when you want token-dense structured content rather than plain text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_chars": { "type": "number", "description": "Truncate to this many characters (default 4000); the last ~12% is kept as a tail so trailing links survive" },
                        "offset": { "type": "number", "minimum": 0, "description": "Character offset into the full markdown for paging (default 0)" }
                    }
                }
            },
            {
                "name": "browser_links",
                "description": "List every anchor link on the current page as one JSON object per line: {text, href}. Use when you need to enumerate where to navigate next.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": { "type": "number", "description": "Max number of links to return (default 100)" },
                        "internal_only": { "type": "boolean", "description": "If true, only return links on the same origin as the current page" }
                    }
                }
            },
            {
                "name": "browser_interactive_elements",
                "description": "List every clickable / typeable element on the current page with a stable ref ID and a brief description. Use this BEFORE clicking or filling so you can refer to elements by ref instead of guessing a CSS selector. Refs look like 'e3' and stay valid until the next navigation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": { "type": "number", "description": "Max number of elements (default 100)" },
                        "max_chars": { "type": "number", "description": "Character budget for the listing (default 4000); the last ~12% is kept as a tail so trailing controls survive" },
                        "offset": { "type": "number", "minimum": 0, "description": "Character offset into the full listing for paging (default 0)" }
                    }
                }
            },
            {
                "name": "browser_structure",
                "description": "Return a structured view of the page's forms and tables as JSON: form fields with labels/roles/disabled state, select options, submit controls, and table headers with row previews. Sensitive fields (password, token, api key, card, ...) never report a value. Controls carry the same ref as browser_snapshot where one exists.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_chars": { "type": "number", "description": "Character budget for the JSON (default 4000); the last ~12% is kept as a tail" },
                        "offset": { "type": "number", "minimum": 0, "description": "Character offset into the full JSON for paging (default 0)" }
                    }
                }
            },
            {
                "name": "browser_table_rows",
                "description": "Read rows of one table advertised by browser_structure, paginated by row offset. Rowspan/colspan are expanded so rows align with the header fields. Tables containing credential inputs are rejected.",
                "inputSchema": {
                    "type": "object",
                    "required": ["table_id"],
                    "properties": {
                        "table_id": { "type": "string", "description": "Table id from browser_structure (element id or positional table-N)" },
                        "offset": { "type": "number", "minimum": 0, "description": "Row offset for paging (default 0)" }
                    }
                }
            },
            {
                "name": "browser_back",
                "description": "Navigate back in the page history (equivalent to the browser back button).",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "browser_forward",
                "description": "Navigate forward in the page history.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "browser_reload",
                "description": "Reload the current page.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "browser_get_cookies",
                "description": "Return all cookies in the browser's cookie jar as one JSON object per line.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "domain": { "type": "string", "description": "Filter to cookies on this domain (default: all)" }
                    }
                }
            },
            {
                "name": "browser_set_cookie",
                "description": "Add or replace a cookie in the jar. Use this to skip a login flow when you already have a session token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "value": { "type": "string" },
                        "domain": { "type": "string", "description": "e.g. example.com or .example.com" },
                        "path": { "type": "string", "description": "default '/'" },
                        "secure": { "type": "boolean" },
                        "http_only": { "type": "boolean" }
                    },
                    "required": ["name", "value", "domain"]
                }
            },
            {
                "name": "browser_clear_cookies",
                "description": "Wipe every cookie from the jar.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "browser_wait_for_text",
                "description": "Wait until a substring appears anywhere in the rendered page text. Use when you want to wait for a result message or notification rather than a specific selector.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" },
                        "timeout": { "type": "number", "description": "Seconds (default 30)" }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "browser_detect_forms",
                "description": "List every <form> on the page with its action URL, method, and a description of each input/textarea/select. Use to understand a form's structure before filling it in.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "browser_fill_form",
                "description": "Fill multiple inputs in one call. `fields` is an array of {ref?, selector?, value, type?}. type='text' (default) sets value, type='check'/'uncheck' toggles checkboxes, type='select' picks an option by value or visible text. Saves N round-trips vs N browser_fill calls.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ref": { "type": "string" },
                                    "selector": { "type": "string" },
                                    "value": { "type": "string" },
                                    "type": { "type": "string", "enum": ["text", "check", "uncheck", "select"] }
                                }
                            }
                        },
                        "submit_ref": { "type": "string", "description": "Optional: click this element after filling (e.g. submit button ref)" },
                        "submit_selector": { "type": "string" }
                    },
                    "required": ["fields"]
                }
            },
            {
                "name": "browser_scroll",
                "description": "Scroll the page or an element. `direction` is 'top'|'bottom'|'up'|'down'|'left'|'right' (default 'down'). `amount` in pixels (default viewport height). Use 'bottom' to trigger infinite-scroll loaders.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "direction": { "type": "string", "enum": ["top", "bottom", "up", "down", "left", "right"] },
                        "amount": { "type": "number", "description": "Pixels (default: one viewport)" },
                        "ref": { "type": "string", "description": "Optional element to scroll into view" },
                        "selector": { "type": "string" }
                    }
                }
            },
            {
                "name": "browser_get_attribute",
                "description": "Read an attribute of an element (href, src, value, class, data-*, etc.). Returns the raw attribute value as a string, or empty string if missing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ref": { "type": "string" },
                        "selector": { "type": "string" },
                        "attribute": { "type": "string", "description": "Attribute name (e.g. href, value, src)" }
                    },
                    "required": ["attribute"]
                }
            },
            {
                "name": "browser_count",
                "description": "Count how many elements on the page match a CSS selector. Cheap existence / pagination probe.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": { "type": "string" }
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "browser_extract",
                "description": "Extract a structured object from the page. Two schema forms: (1) a map of {field_name: css_selector} where each field gets the matching element's text content (or attribute via 'selector@attr', e.g. 'a@href'; append '[]' to the field name for an array); (2) a JSON-schema object {'type':'object','properties':{field:{'x-ref':'e1-2','type':'number'}},'required':[...]} where each 'x-ref' is a ref from the last browser_snapshot and the element's text is coerced to the declared type (string|number|integer|boolean).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "object",
                            "description": "Map of field_name to CSS selector. Suffix selector with '@attr' for attribute, suffix field name with '[]' for array."
                        }
                    },
                    "required": ["schema"]
                }
            },
            {
                "name": "browser_tab_new",
                "description": "Open a new tab (isolated browser page). Returns the tab ID; subsequent tool calls operate on the most recently opened or browser_tab_switch'd tab.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "Optional URL to navigate the new tab to" }
                    }
                }
            },
            {
                "name": "browser_tab_list",
                "description": "List all open tabs with their ID, URL, title, and which one is active.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "browser_tab_switch",
                "description": "Switch the active tab. All subsequent tool calls (snapshot, click, etc.) target this tab.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tab_id": { "type": "string" }
                    },
                    "required": ["tab_id"]
                }
            },
            {
                "name": "browser_tab_close",
                "description": "Close a tab by ID (default: the active tab). If you close the active tab, the next remaining tab becomes active.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tab_id": { "type": "string" }
                    }
                }
            },
            {
                "name": "browser_search",
                "description": "Find substring matches in the visible page text. Returns each match with its surrounding context. Use this to confirm content exists before scraping or to locate a section.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "case_sensitive": { "type": "boolean" },
                        "limit": { "type": "number", "description": "Max matches to return (default 10)" },
                        "context_chars": { "type": "number", "description": "Chars on each side of the match (default 80)" }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "browser_storage_state",
                "description": "Export the full authentication / session state (cookies + localStorage + sessionStorage) as a JSON object in Playwright storageState shape (httpOnly/sameSite cookie fields, {name, value} storage entries; sessionStorage is an obscura extension). Save this to skip a login on a subsequent run via browser_set_storage_state. Pass `path` to write the JSON to a file instead of returning it (atomic tmp+rename, mode 0600).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Optional file path to write the exported JSON to (atomic, owner-only)." }
                    }
                }
            },
            {
                "name": "browser_set_storage_state",
                "description": "Restore session state previously returned by browser_storage_state — inline as `state`, or from a file via `path` (accepts Playwright context.storageState() files). Use to bring an authenticated session back without re-logging in.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "object",
                            "description": "{cookies: [...], origins: [{origin, localStorage: [...], sessionStorage: [...]}]}"
                        },
                        "path": {
                            "type": "string",
                            "description": "File containing a storage-state JSON object, e.g. written by browser_storage_state or Playwright's context.storageState()."
                        }
                    }
                }
            }
    ]).as_array().cloned().expect("MCP tool list must be an array");

    #[cfg(feature = "render")]
    {
        tools.extend([
            json!({
                "name": "browser_screenshot",
                "description": "Capture the current rendered viewport as a PNG image. Width and height default to the page's current CSS viewport.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "width": { "type": "number", "exclusiveMinimum": 0, "maximum": 32768, "description": "Optional CSS-pixel capture width" },
                        "height": { "type": "number", "exclusiveMinimum": 0, "maximum": 32768, "description": "Optional CSS-pixel capture height" }
                    },
                    "additionalProperties": false
                }
            }),
            json!({
                "name": "browser_pdf",
                "description": "Export the current rendered document as a paginated raster PDF using print media and bounded PDF defaults.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "landscape": { "type": "boolean" },
                        "print_background": { "type": "boolean" },
                        "scale": { "type": "number", "minimum": 0.1, "maximum": 2.0 },
                        "paper_width": { "type": "number", "exclusiveMinimum": 0, "maximum": 200, "description": "Paper width in inches" },
                        "paper_height": { "type": "number", "exclusiveMinimum": 0, "maximum": 200, "description": "Paper height in inches" },
                        "margin_top": { "type": "number", "minimum": 0, "description": "Top margin in inches" },
                        "margin_bottom": { "type": "number", "minimum": 0, "description": "Bottom margin in inches" },
                        "margin_left": { "type": "number", "minimum": 0, "description": "Left margin in inches" },
                        "margin_right": { "type": "number", "minimum": 0, "description": "Right margin in inches" }
                    },
                    "additionalProperties": false
                }
            }),
        ]);
    }

    if web_search_enabled() {
        tools.push(json!({
            "name": "web_search",
            "description": "Search the web and return matching pages, one JSON object per line with title, url, and a short snippet. Use it to find pages to open with browser_navigate. Queries are sent to the Keenable search API.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "number", "description": "Max results to return (default 5, max 50)" }
                },
                "required": ["query"]
            }
        }));
    }

    RpcResponse::ok(id, json!({ "tools": tools }))
}

async fn handle_tool_call(id: Value, params: &Value, state: &mut BrowserState) -> RpcResponse {
    let name = match params.get("name").and_then(Value::as_str) {
        Some(n) => n,
        None => return RpcResponse::err(id, -32602, "Missing tool name"),
    };
    let args = params.get("arguments").unwrap_or(&Value::Null);

    #[cfg(feature = "render")]
    {
        let media_result = match name {
            "browser_screenshot" => Some(tool_screenshot(args, state).await),
            "browser_pdf" => Some(tool_pdf(args, state).await),
            _ => None,
        };
        if let Some(result) = media_result {
            return match result {
                Ok(content) => RpcResponse::ok(id, json!({ "content": [content] })),
                Err(error) => RpcResponse::ok(id, json!({
                    "content": [{ "type": "text", "text": format!("Error: {error}") }],
                    "isError": true
                })),
            };
        }
    }

    let result = match name {
        "browser_navigate" => tool_navigate(args, state).await,
        "browser_snapshot" => tool_snapshot(args, state),
        "browser_click" => tool_click(args, state).await,
        "browser_fill" => tool_fill(args, state).await,
        "browser_type" => tool_type(args, state).await,
        "browser_press_key" => tool_press_key(args, state).await,
        "browser_select_option" => tool_select_option(args, state),
        "browser_evaluate" => tool_evaluate(args, state).await,
        "browser_wait_for" => tool_wait_for(args, state).await,
        "browser_wait_for_function" => tool_wait_for_function(args, state).await,
        "browser_network_requests" => tool_network_requests(state),
        "browser_console_messages" => tool_console_messages(state),
        "browser_close" => tool_close(state),
        // Tier 1 agent-UX additions
        "browser_markdown" => tool_markdown(args, state),
        "browser_links" => tool_links(args, state),
        "browser_interactive_elements" => tool_interactive_elements(args, state),
        "browser_back" => tool_back(state).await,
        "browser_forward" => tool_forward(state).await,
        "browser_reload" => tool_reload(state).await,
        "browser_get_cookies" => tool_get_cookies(args, state),
        "browser_set_cookie" => tool_set_cookie(args, state),
        "browser_clear_cookies" => tool_clear_cookies(state),
        "browser_wait_for_text" => tool_wait_for_text(args, state).await,
        // Tier 2 agent-UX additions
        "browser_detect_forms" => tool_detect_forms(state),
        "browser_structure" => tool_structure(args, state),
        "browser_table_rows" => tool_table_rows(args, state),
        "browser_fill_form" => tool_fill_form(args, state).await,
        "browser_scroll" => tool_scroll(args, state),
        "browser_get_attribute" => tool_get_attribute(args, state),
        "browser_count" => tool_count(args, state),
        "browser_extract" => tool_extract(args, state),
        "browser_tab_new" => tool_tab_new(args, state).await,
        "browser_tab_list" => tool_tab_list(state),
        "browser_tab_switch" => tool_tab_switch(args, state),
        "browser_tab_close" => tool_tab_close(args, state),
        "browser_search" => tool_search(args, state),
        "browser_storage_state" => tool_storage_state(args, state),
        "browser_set_storage_state" => tool_set_storage_state(args, state),
        "web_search" if web_search_enabled() => tool_web_search(args, state).await,
        _ => Err(format!("Unknown tool: {name}")),
    };

    match result {
        Ok(content) => RpcResponse::ok(id, json!({
            "content": [{ "type": "text", "text": content }]
        })),
        Err(e) => RpcResponse::ok(id, json!({
            "content": [{ "type": "text", "text": format!("Error: {e}") }],
            "isError": true
        })),
    }
}

#[cfg(feature = "render")]
fn validate_tool_options(args: &Value, allowed: &[&str]) -> Result<(), String> {
    if args.is_null() {
        return Ok(());
    }
    let object = args.as_object().ok_or("tool arguments must be an object")?;
    if let Some(name) = object.keys().find(|name| !allowed.contains(&name.as_str())) {
        return Err(format!("unsupported option '{name}'"));
    }
    Ok(())
}

#[cfg(feature = "render")]
fn optional_number(args: &Value, name: &str) -> Result<Option<f32>, String> {
    let Some(value) = args.get(name) else {
        return Ok(None);
    };
    let value = value.as_f64().ok_or_else(|| format!("'{name}' must be a number"))?;
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(format!("'{name}' must be a finite number"));
    }
    Ok(Some(value as f32))
}

#[cfg(feature = "render")]
fn optional_bool(args: &Value, name: &str) -> Result<Option<bool>, String> {
    let Some(value) = args.get(name) else {
        return Ok(None);
    };
    value.as_bool().map(Some).ok_or_else(|| format!("'{name}' must be a boolean"))
}

#[cfg(feature = "render")]
fn validate_screenshot_viewport(viewport: (f32, f32)) -> Result<(), String> {
    const MAX_DIMENSION: f32 = 32_768.0;
    const MAX_PIXELS: f64 = (16 * 1024 * 1024) as f64;
    let (width, height) = viewport;
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0
        || width > MAX_DIMENSION || height > MAX_DIMENSION
    {
        return Err("screenshot dimensions must be finite, positive, and at most 32768 CSS pixels".into());
    }
    if f64::from(width.ceil()) * f64::from(height.ceil()) > MAX_PIXELS {
        return Err("screenshot dimensions exceed the 16-megapixel capture limit".into());
    }
    Ok(())
}

#[cfg(feature = "render")]
async fn tool_screenshot(args: &Value, state: &mut BrowserState) -> Result<Value, String> {
    validate_tool_options(args, &["width", "height"])?;
    let current = state.page_mut().viewport;
    let viewport = (
        optional_number(args, "width")?.unwrap_or(current.0),
        optional_number(args, "height")?.unwrap_or(current.1),
    );
    validate_screenshot_viewport(viewport)?;

    let page = state.page_mut();
    let _ = page.prepare_screenshot_resources(1_000).await;
    let png = page.screenshot(viewport).ok_or("the current page has no renderable viewport")?;
    Ok(json!({
        "type": "image",
        "data": BASE64.encode(png),
        "mimeType": "image/png"
    }))
}

#[cfg(feature = "render")]
async fn tool_pdf(args: &Value, state: &mut BrowserState) -> Result<Value, String> {
    validate_tool_options(args, &[
        "landscape", "print_background", "scale", "paper_width", "paper_height",
        "margin_top", "margin_bottom", "margin_left", "margin_right",
    ])?;
    let mut options = obscura_browser::RasterPdfOptions::default();
    if let Some(value) = optional_bool(args, "landscape")? { options.landscape = value; }
    if let Some(value) = optional_bool(args, "print_background")? { options.print_background = value; }
    if let Some(value) = optional_number(args, "scale")? { options.scale = value; }
    if let Some(value) = optional_number(args, "paper_width")? { options.paper_width_in = value; }
    if let Some(value) = optional_number(args, "paper_height")? { options.paper_height_in = value; }
    if let Some(value) = optional_number(args, "margin_top")? { options.margin_top_in = value; }
    if let Some(value) = optional_number(args, "margin_bottom")? { options.margin_bottom_in = value; }
    if let Some(value) = optional_number(args, "margin_left")? { options.margin_left_in = value; }
    if let Some(value) = optional_number(args, "margin_right")? { options.margin_right_in = value; }

    let page = state.page_mut();
    let _ = page.prepare_screenshot_resources(1_000).await;
    let pdf = page.raster_pdf(options).map_err(|error| error.to_string())?;
    Ok(json!({
        "type": "resource",
        "resource": {
            "uri": "obscura://capture/current-page.pdf",
            "mimeType": "application/pdf",
            "blob": BASE64.encode(pdf)
        }
    }))
}

/// Resolve a tool call's element target from either `ref` (preferred) or
/// `selector` (fallback). Agents that called `browser_snapshot` /
/// `browser_interactive_elements` get a ref table they can refer to;
/// scripted clients can still pass raw CSS selectors.
fn resolve_target(args: &Value, state: &BrowserState) -> Result<String, String> {
    if let Some(r) = args.get("ref").and_then(Value::as_str) {
        return state.ref_to_selector(r);
    }
    if let Some(sel) = args.get("selector").and_then(Value::as_str) {
        return Ok(sel.to_string());
    }
    Err("Missing 'ref' or 'selector' parameter".to_string())
}

fn evaluate_tool_script(page: &mut Page, script: &str) -> Result<Value, String> {
    // Keep the watchdog armed through result serialization, which can run page code.
    let watchdog = page.js.as_mut().ok_or("Navigate before evaluating JavaScript")?
        .arm_watchdog(std::time::Duration::from_secs(5));
    let result = page.evaluate_script(script);
    if page.js.as_mut().expect("evaluation runtime").disarm_watchdog(watchdog) {
        return Err("JavaScript execution or serialization exceeded the 5000ms deadline".into());
    }
    result
}

/// Clamp text to `max_chars` and tack on a `...(truncated, N more chars)`
/// marker so the agent can ask for more if needed. Default ceiling is
/// 4 KiB to prevent a single tool call from consuming a window of context.
fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(max_chars).collect();
    let remaining = text.chars().count() - max_chars;
    format!("{head}\n...(truncated, {remaining} more chars)")
}

/// Port of camofox's `windowSnapshot`: when a text exceeds the budget, keep
/// a head chunk plus the trailing `tail` chars so pagination and navigation
/// affordances at the end of the page survive truncation. `offset` (chars)
/// pages through the body; the marker advertises the next offset. Chunk and
/// tail boundaries snap to line starts so a `ref=` line is never advertised
/// truncated.
fn window_text(text: &str, max_chars: usize, offset: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    if max_chars == 0 || total == 0 {
        return String::new();
    }
    if total <= max_chars && offset == 0 {
        return text.to_string();
    }
    let tail_len = (max_chars / 8).clamp(64, 500).min(total);
    let budget = max_chars.saturating_sub(tail_len + 96).max(1);
    let start = offset.min(total - tail_len);
    let mut end = (start + budget).min(total - tail_len);
    // Never end mid-line: a split `ref=` line would advertise a dead reference.
    if let Some(rel) = chars[start..end].iter().rposition(|c| *c == '\n') {
        end = start + rel + 1;
    }
    // The tail starts at a line boundary too, when one exists inside it.
    let mut tail_start = total - tail_len;
    if let Some(rel) = chars[tail_start..].iter().position(|c| *c == '\n') {
        if rel > 0 {
            tail_start += rel + 1;
        }
    }
    let has_more = end < total - tail_len;
    let chunk: String = chars[start..end].iter().collect();
    let tail: String = chars[tail_start..].iter().collect();
    if has_more {
        format!("{chunk}[... truncated at char {end} of {total}; pass offset={end} for more. Pagination links below. ...]\n{tail}")
    } else {
        format!("{chunk}{tail}")
    }
}

async fn tool_navigate(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let url = args.get("url").and_then(Value::as_str)
        .ok_or("Missing url parameter")?;
    if url::Url::parse(url)
        .ok()
        .is_some_and(|parsed| parsed.scheme() == "file")
    {
        return Err("file:// navigation is disabled for MCP".to_string());
    }
    let wait_until = args.get("waitUntil").and_then(Value::as_str).unwrap_or("load");

    let macro_name = args.get("macro").and_then(Value::as_str).unwrap_or("");
    let query = args.get("query").and_then(Value::as_str).unwrap_or("");
    let url = if !macro_name.is_empty()
        && macro_name != "__NO__" && macro_name != "none" && macro_name != "null"
    {
        expand_search_macro(macro_name, query).unwrap_or_else(|| url.to_string())
    } else {
        url.to_string()
    };
    let url = url.as_str();

    let condition = obscura_browser::lifecycle::WaitUntil::from_str(wait_until);
    // A timed-out load can already have replaced the document's arena.
    state.interactive_refs.clear();
    let ua = state.user_agent.clone();
    let operator_hints = state.operator_network_hints;
    let page = state.page_mut();
    if let Some(ref ua) = ua {
        page.http_client.set_user_agent(ua).await;
    }

    // camofox getSearchFallbacks: a google search that fails retries on
    // duckduckgo then bing before surfacing the error.
    let mut navigate_err = match page.navigate_with_wait(url, condition).await {
        Ok(()) => None,
        Err(e) => Some(e.to_string()),
    };
    if navigate_err.is_some() && macro_name == "@google_search" {
        for fallback in [
            format!("https://duckduckgo.com/?q={}", encode_query_component(query)),
            format!("https://www.bing.com/search?q={}", encode_query_component(query)),
        ] {
            match page.navigate_with_wait(&fallback, condition).await {
                Ok(()) => { navigate_err = None; break; }
                Err(e) => navigate_err = Some(e.to_string()),
            }
        }
    }
    if let Some(message) = navigate_err {
        if operator_hints {
            if let Some(hint) = obscura_net::private_network_error_hint(&message) {
                return Err(format!("{}\n{}", message, hint));
            }
        }
        return Err(message);
    }

    let dismissed = args.get("dismissConsent").and_then(Value::as_bool).unwrap_or(false)
        .then(|| page.evaluate(DISMISS_CONSENT_SCRIPT).as_str().map(str::to_string))
        .flatten();

    let mut summary = format!("Navigated to {} — \"{}\"", page.url_string(), page.title);
    if let Some(selector) = dismissed {
        summary.push_str(&format!("\nDismissed consent dialog via {selector}"));
    }
    Ok(summary)
}

fn encode_query_component(query: &str) -> String {
    // encodeURIComponent semantics: byte_serialize emits '+' for space.
    url::form_urlencoded::byte_serialize(query.as_bytes())
        .collect::<String>()
        .replace('+', "%20")
}

// camofox lib/macros.js: a small fixed table of @macro -> URL templates.
fn expand_search_macro(macro_name: &str, query: &str) -> Option<String> {
    let q = encode_query_component(query);
    let template = match macro_name {
        "@google_search" => format!("https://www.google.com/search?q={q}"),
        "@youtube_search" => format!("https://www.youtube.com/results?search_query={q}"),
        "@amazon_search" => format!("https://www.amazon.com/s?k={q}"),
        "@reddit_search" => format!("https://www.reddit.com/search.json?q={q}&limit=25"),
        "@reddit_subreddit" => format!("https://www.reddit.com/r/{q}.json?limit=25"),
        "@wikipedia_search" => format!("https://en.wikipedia.org/wiki/Special:Search?search={q}"),
        "@twitter_search" => format!("https://twitter.com/search?q={q}"),
        "@yelp_search" => format!("https://www.yelp.com/search?find_desc={q}"),
        "@spotify_search" => format!("https://open.spotify.com/search/{q}"),
        "@netflix_search" => format!("https://www.netflix.com/search?q={q}"),
        "@linkedin_search" => format!("https://www.linkedin.com/search/results/all/?keywords={q}"),
        "@instagram_search" => format!("https://www.instagram.com/explore/tags/{q}"),
        "@tiktok_search" => format!("https://www.tiktok.com/search?q={q}"),
        "@twitch_search" => format!("https://www.twitch.tv/search?term={q}"),
        _ => return None,
    };
    Some(template)
}

// Restrict this opt-in behavior to consent-manager controls (camofox
// dismissConsentDialogs): broad dialog/modal selectors could close
// application UI unrelated to consent. Clicks the first visible match only.
const DISMISS_CONSENT_SCRIPT: &str = r#"(function(){
  var selectors = [
    '#onetrust-banner-sdk #onetrust-accept-btn-handler',
    '#onetrust-banner-sdk #onetrust-reject-all-handler',
    '#onetrust-banner-sdk #onetrust-close-btn-container button',
    '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
    '#CybotCookiebotDialogBodyButtonDecline',
    '#qc-cmp2-ui button[mode="primary"]'
  ];
  for (var i = 0; i < selectors.length; i++) {
    var el = document.querySelector(selectors[i]);
    if (el && (typeof el.checkVisibility !== 'function' || el.checkVisibility({checkVisibilityCSS: true}))) {
      el.click();
      return selectors[i];
    }
  }
  return null;
})()"#;

fn tool_snapshot(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let max_chars = args.get("max_chars").and_then(Value::as_u64).map(|n| n as usize)
        .unwrap_or(DEFAULT_TEXT_LIMIT);
    let offset = args.get("offset").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(0);
    rebuild_interactive_refs(state)?;
    let page = state.page_mut();
    let url = truncate(&page.url_string(), 256);
    let title = truncate(&page.title, 256);
    if max_chars == 0 { return Ok(format!("URL: {url}\nTitle: {title}")); }
    let context = accessibility_context(page);
    let body_text = page.with_dom(|dom| {
        if let Ok(Some(body)) = dom.query_selector("body") {
            extract_text(dom, body, context.as_ref())
        } else {
            String::new()
        }
    }).unwrap_or_default();

    let lines = context.as_ref().map(|context| interactive_lines(state, context, usize::MAX)).unwrap_or_default();
    let mut full = lines.join("\n");
    if !full.is_empty() { full.push('\n'); }
    full.push_str(body_text.trim());
    let content = window_text(&full, max_chars, offset);
    Ok(format!("URL: {url}\nTitle: {title}\n\n{content}"))
}

async fn tool_click(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let selector = resolve_target(args, state)?;

    let js = format!(
        r#"(function(){{
            var el = document.querySelector({sel});
            if (!el) return "error:element not found";
            el.click();
            return "ok";
        }})()"#,
        sel = serde_json::to_string(&selector).unwrap()
    );

    let result = evaluate_tool_script(state.page_mut(), &js)?;
    if result.as_str() == Some("error:element not found") {
        Err(format!("Element not found: {selector}"))
    } else {
        state.settle_synthetic_navigation().await?;
        Ok(format!("Clicked '{selector}'"))
    }
}

async fn tool_fill(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let selector = resolve_target(args, state)?;
    let value = args.get("value").and_then(Value::as_str)
        .ok_or("Missing value parameter")?;

    let js = format!(
        r#"(function(){{
            var el = document.querySelector({sel});
            if (!el) return "error:element not found";
            globalThis.__obscura_setFieldValue(el, "value", {val});
            globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event("input", {{bubbles:true}})));
            globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event("change", {{bubbles:true}})));
            return "ok";
        }})()"#,
        sel = serde_json::to_string(&selector).unwrap(),
        val = serde_json::to_string(value).unwrap()
    );

    let result = evaluate_tool_script(state.page_mut(), &js)?;
    if result.as_str() == Some("error:element not found") {
        Err(format!("Element not found: {selector}"))
    } else {
        state.settle_synthetic_navigation().await?;
        Ok(format!("Filled '{selector}' with value"))
    }
}

async fn tool_type(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let selector = resolve_target(args, state)?;
    let text = args.get("text").and_then(Value::as_str)
        .ok_or("Missing text parameter")?;

    let js = format!(
        r#"(function(){{
            var el = document.querySelector({sel});
            if (!el) return "error:element not found";
            globalThis.__obscura_setFieldValue(el, "value", (el.value || "") + {txt});
            globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event("input", {{bubbles:true}})));
            return "ok";
        }})()"#,
        sel = serde_json::to_string(&selector).unwrap(),
        txt = serde_json::to_string(text).unwrap()
    );

    let result = evaluate_tool_script(state.page_mut(), &js)?;
    if result.as_str() == Some("error:element not found") {
        Err(format!("Element not found: {selector}"))
    } else {
        state.settle_synthetic_navigation().await?;
        Ok(format!("Typed into '{selector}'"))
    }
}

async fn tool_press_key(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let key = args.get("key").and_then(Value::as_str)
        .ok_or("Missing key parameter")?;
    let selector = args.get("selector").and_then(Value::as_str);

    let target = match selector {
        Some(sel) => format!("document.querySelector({})", serde_json::to_string(sel).unwrap()),
        None => "document".to_string(),
    };

    let js = format!(
        r#"(function(){{
            var t = {target};
            if (!t) return "error:element not found";
            t.dispatchEvent(new KeyboardEvent("keydown", {{key:{key},bubbles:true}}));
            t.dispatchEvent(new KeyboardEvent("keyup", {{key:{key},bubbles:true}}));
            return "ok";
        }})()"#,
        target = target,
        key = serde_json::to_string(key).unwrap()
    );

    let result = evaluate_tool_script(state.page_mut(), &js)?;
    if result.as_str() == Some("error:element not found") {
        return Err("Element not found for key press".into());
    }
    state.settle_synthetic_navigation().await?;
    Ok(format!("Pressed key '{key}'"))
}

fn tool_select_option(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let selector = args.get("selector").and_then(Value::as_str)
        .ok_or("Missing selector parameter")?;
    let value = args.get("value").and_then(Value::as_str)
        .ok_or("Missing value parameter")?;

    let js = format!(
        r#"(function(){{
            var el = document.querySelector({sel});
            if (!el) return "error:element not found";
            var opts = Array.from(el.options);
            var opt = opts.find(function(o){{ return o.value === {val} || o.text === {val}; }});
            if (!opt) return "error:option not found";
            el.value = opt.value;
            el.dispatchEvent(new Event("change", {{bubbles:true}}));
            return "ok";
        }})()"#,
        sel = serde_json::to_string(selector).unwrap(),
        val = serde_json::to_string(value).unwrap()
    );

    let result = evaluate_tool_script(state.page_mut(), &js)?;
    match result.as_str() {
        Some("error:element not found") => Err(format!("Element not found: {selector}")),
        Some("error:option not found") => Err(format!("Option not found: {value}")),
        _ => Ok(format!("Selected '{value}' in '{selector}'")),
    }
}

async fn tool_evaluate(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let expression = args.get("expression").and_then(Value::as_str)
        .ok_or("Missing expression parameter")?;
    let timeout_ms = args.get("timeoutMs").and_then(Value::as_u64).unwrap_or(5_000).max(1);
    let budget = std::time::Duration::from_millis(timeout_ms);
    std::time::Instant::now().checked_add(budget).ok_or("timeoutMs is too large")?;
    let page = state.page_mut();
    let watchdog = page.js.as_mut().ok_or("Navigate before evaluating JavaScript")?.arm_watchdog(budget);
    let result = page.evaluate_for_cdp_with_timeout(expression, true, true, timeout_ms).await;
    // MCP never exposes remote handles. Release results, including exceptions.
    page.release_object_group();
    if page.js.as_mut().expect("evaluation runtime").disarm_watchdog(watchdog) {
        // Termination can prevent the first cleanup; retry under a fresh bounded guard.
        let cleanup = page.js.as_mut().expect("evaluation runtime").arm_watchdog(budget);
        page.release_object_group();
        page.js.as_mut().expect("evaluation runtime").disarm_watchdog(cleanup);
        return Err(format!("JavaScript evaluation exceeded {timeout_ms}ms"));
    }
    let result = result?;
    if result.thrown {
        return Err(result.description);
    }
    let result = result.value.unwrap_or(Value::Null);
    state.settle_synthetic_navigation().await?;
    Ok(match &result {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    })
}

/// Poll a JavaScript predicate until it returns something truthy.
///
/// Agents otherwise express "wait until the app finished loading" as a loop of
/// `browser_evaluate` plus a sleep, which costs a round trip per iteration and
/// is where flaky automation comes from.
async fn tool_wait_for_function(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let expression = args.get("function").or_else(|| args.get("expression"))
        .and_then(Value::as_str)
        .ok_or("Missing function parameter")?;
    let timeout_ms = args.get("timeout").and_then(Value::as_u64).unwrap_or(30_000);
    let poll_ms = args.get("pollIntervalMs").and_then(Value::as_u64).unwrap_or(100).max(10);

    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(timeout_ms);
    let mut attempts: u64 = 0;
    loop {
        attempts += 1;
        let value = evaluate_tool_script(state.page_mut(), &format!("JSON.stringify((function(){{ try {{ return (function(){{ return !!({expression}); }})(); }} catch (e) {{ return false; }} }})())"))?;
        let truthy = value
            .as_str()
            .map(|raw| raw == "true")
            .unwrap_or(value.as_bool().unwrap_or(false));
        if truthy {
            return Ok(format!("condition met after {attempts} poll(s)"));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!(
                "condition not met within {timeout_ms} ms after {attempts} poll(s): {expression}"
            ));
        }
        let poll_deadline = (tokio::time::Instant::now()
            + tokio::time::Duration::from_millis(poll_ms)).min(deadline);
        while tokio::time::Instant::now() < poll_deadline {
            match tokio::time::timeout_at(
                poll_deadline, state.page_mut().run_autonomous_event_loop_turn(),
            ).await {
                Ok(result) => {
                    // Navigation consumes pending state before I/O, so do not cancel it mid-turn.
                    if state.finish_active_page_turn(result?).await? {
                        tokio::time::sleep_until(poll_deadline).await;
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }
}

async fn tool_wait_for(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let selector = args.get("selector").and_then(Value::as_str)
        .ok_or("Missing selector parameter")?;
    let timeout_secs = args.get("timeout").and_then(Value::as_f64).unwrap_or(30.0) as u64;

    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);
    // Exponential backoff: 5 -> 10 -> 20 -> ... -> 200 ms. The old fixed
    // 200ms tick added up to a full poll cycle of latency every time;
    // a selector that appears in 30ms now returns in ~35ms instead of
    // the next 200ms tick.
    let mut tick_ms: u64 = 5;
    loop {
        let found = state.page_mut().with_dom(|dom| {
            dom.query_selector(selector).ok().flatten().is_some()
        }).unwrap_or(false);

        if found {
            return Ok(format!("Found '{selector}'"));
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(format!("Timeout waiting for '{selector}'"));
        }

        let tick = tokio::time::Duration::from_millis(tick_ms);
        match tokio::time::timeout(tick, state.advance_active_page_tasks()).await {
            Ok(result) => {
                result?;
            }
            Err(_) => {}
        }
        if tick_ms < 200 { tick_ms = (tick_ms * 2).min(200); }
    }
}

fn tool_network_requests(state: &mut BrowserState) -> Result<String, String> {
    let page = state.page_mut();
    page.sync_js_network_events();
    let events = &page.network_events;

    if events.is_empty() {
        return Ok("No network requests recorded.".to_string());
    }

    let lines: Vec<String> = events.iter().map(|e| {
        format!("[{}] {} {} ({}B)", e.status, e.method, e.url, e.body_size)
    }).collect();

    Ok(lines.join("\n"))
}

fn tool_console_messages(state: &mut BrowserState) -> Result<String, String> {
    let messages = state.page_mut().take_pending_console_messages();
    state.console_messages.extend(messages);
    if state.console_messages.len() > 1_024 {
        let overflow = state.console_messages.len() - 1_024;
        state.console_messages.drain(..overflow);
    }
    if state.console_messages.is_empty() {
        Ok("No console messages.".to_string())
    } else {
        Ok(state.console_messages.join("\n"))
    }
}

fn tool_close(state: &mut BrowserState) -> Result<String, String> {
    // Retire each page's frame handles before clearing the tab map.
    for page in state.tabs.values_mut() {
        page.suspend_js();
    }
    state.tabs.clear();
    state.active_tab = None;
    state.console_messages.clear();
    state.interactive_refs.clear();
    Ok("All browser tabs closed.".to_string())
}

// ===== Tier 1 agent-UX additions =====

/// Convert the rendered page to Markdown by running the JS-side converter
/// already used by `obscura fetch --dump markdown`. More token-dense than
/// browser_snapshot for content-heavy pages (article bodies, docs sites).
fn tool_markdown(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let max_chars = args.get("max_chars").and_then(Value::as_u64).map(|n| n as usize)
        .unwrap_or(DEFAULT_TEXT_LIMIT);
    let offset = args.get("offset").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(0);
    let page = state.page_mut();
    Ok(window_text(&page.markdown(), max_chars, offset))
}

/// Enumerate every `<a href>` on the page. One JSON object per line so
/// the agent can grep / split without round-tripping to a JSON parser.
fn tool_links(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(100) as usize;
    let internal_only = args.get("internal_only").and_then(Value::as_bool).unwrap_or(false);
    let page = state.page_mut();
    let base_origin = url::Url::parse(&page.url_string())
        .ok()
        .map(|u| u.origin())
        .unwrap_or_else(|| url::Url::parse("about:blank").unwrap().origin());

    let js = r#"(function(){
        var out = [];
        var seen = new Set();
        var as = document.querySelectorAll('a[href]');
        for (var i = 0; i < as.length; i++) {
            var a = as[i];
            var href = a.href || '';
            if (!href || href === '#' || href.startsWith('javascript:')) continue;
            if (seen.has(href)) continue;
            seen.add(href);
            var t = (a.innerText || a.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 200);
            out.push({text: t, href: href});
        }
        return out;
    })()"#;
    let val = evaluate_tool_script(page, js)?;
    let arr = val.as_array().cloned().unwrap_or_default();
    let lines: Vec<String> = arr.into_iter()
        .filter(|item| {
            if !internal_only { return true; }
            item.get("href").and_then(|v| v.as_str())
                .and_then(|h| url::Url::parse(h).ok())
                .map(|u| u.origin() == base_origin)
                .unwrap_or(false)
        })
        .take(limit)
        .map(|item| item.to_string())
        .collect();
    if lines.is_empty() {
        Ok("No links found.".to_string())
    } else {
        Ok(lines.join("\n"))
    }
}

/// List every interactable element with a stable ref ID, the kind of
/// element, and a one-line description. Agents pass `ref` to click/fill/
/// type instead of crafting selectors. Also assigns `data-obscura-ref`
/// to each element so the ref survives until the next navigation.
fn tool_interactive_elements(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(100) as usize;
    let max_chars = args.get("max_chars").and_then(Value::as_u64).map(|n| n as usize)
        .unwrap_or(DEFAULT_TEXT_LIMIT);
    let offset = args.get("offset").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(0);
    rebuild_interactive_refs(state)?;
    let context = accessibility_context(state.page_mut());
    let lines = context.as_ref().map(|context| interactive_lines(state, context, limit)).unwrap_or_default();
    if lines.is_empty() {
        return Ok("No interactive elements on this page.".to_string());
    }
    Ok(window_text(&lines.join("\n"), max_chars, offset))
}

fn accessibility_context(page: &Page) -> Option<obscura_browser::accessibility::AccessibilityContext> {
    #[cfg(feature = "render")]
    let styles = page.js.as_ref().map(|js| js.accessibility_styles()).unwrap_or_default();
    #[cfg(not(feature = "render"))]
    let styles = HashMap::new();
    page.with_dom(|dom| obscura_browser::accessibility::AccessibilityContext::for_document(dom, &styles))
}

fn interactive_lines(state: &BrowserState, context: &obscura_browser::accessibility::AccessibilityContext, limit: usize) -> Vec<String> {
    let Some(page) = state.active_tab.as_ref().and_then(|id| state.tabs.get(id)) else { return Vec::new(); };
    let refs: HashMap<_, _> = state.interactive_refs.iter().map(|(reference, node)| (node.node_id.raw(), reference.as_str())).collect();
    page.with_dom(|dom| dom.descendants(dom.document()).into_iter().filter_map(|id| {
        let reference = refs.get(&id.raw())?;
        let ax = context.describe_node(dom, id)?;
        let node = dom.get_node(id)?;
        let tag = node.as_element()?.local.as_ref();
        let preview = |value: &str| value.chars().take(80).collect::<String>();
        let label = ax["name"]["value"].as_str().map(str::to_owned)
            .unwrap_or_else(|| extract_text(dom, id, Some(context)).trim().to_string());
        let label = preview(&label);
        let role = ax["role"]["value"].as_str().unwrap_or("generic");
        let mut line = format!("ref={reference} {tag} {label:?} role={role}");
        if let Some(kind) = node.get_attribute("type") { line.push_str(&format!(" type={:?}", preview(kind))); }
        if let Some(name) = node.get_attribute("name") { line.push_str(&format!(" name={:?}", preview(name))); }
        if let Some(value) = ax["value"]["value"].as_str() { line.push_str(&format!(" value={:?}", preview(value))); }
        if let Some(properties) = ax["properties"].as_array() {
            for property in properties {
                let name = property["name"].as_str().unwrap_or("");
                if matches!(name, "checked" | "disabled" | "required" | "expanded" | "selected" | "pressed") {
                    line.push_str(&format!(" {name}={}", property["value"]["value"]));
                }
            }
        }
        Some(line)
    }).take(limit).collect()).unwrap_or_default()
}

/// Rebuild the ref table: walk the DOM, find every interactable, assign
/// a stable `data-obscura-ref="eN"` attribute, remember the nid for later
/// validation. Called on every snapshot / interactive-elements call so the
/// agent always sees fresh refs.
fn rebuild_interactive_refs(state: &mut BrowserState) -> Result<(), String> {
    state.interactive_refs.clear();
    state.ref_generation = state.ref_generation.checked_add(1)
        .ok_or("Element reference generation exhausted")?;
    let prefix = format!("e{}-", state.ref_generation);
    let page = state.page_mut();
    if !page.has_js() { return Ok(()); }
    // Never reuse tokens: an old snapshot must not name a different action.
    let tag_js = format!(r#"(function(){{
        var sel = 'a[href], button, input:not([type=hidden]), select, textarea, [role=button], [role=link], [role=checkbox], [role=tab], [role=menuitem], [role=option], [onclick], [tabindex]:not([tabindex="-1"])';
        var els = document.querySelectorAll(sel);
        var refs = [];
        for (var i = 0; i < els.length; i++) {{
            var ref = '{prefix}' + (i + 1);
            els[i].setAttribute('data-obscura-ref', ref);
            refs.push(ref);
        }}
        return refs;
    }})()"#);
    let val = evaluate_tool_script(page, &tag_js)?;
    let refs: Vec<String> = val.as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    // Map ref -> nid via a second pass so ref_to_selector can sanity-check.
    for r in refs {
        let selector = format!("[data-obscura-ref=\"{r}\"]");
        let page = state.page_mut();
        let node = page.with_dom(|dom| {
            let node_id = dom.query_selector(&selector).ok().flatten()?;
            Some(ElementRef { node_id, allocation_generation: dom.node_allocation_generation(node_id)? })
        }).flatten();
        if let Some(node) = node {
            state.interactive_refs.insert(r, node);
        }
    }
    Ok(())
}

async fn tool_back(state: &mut BrowserState) -> Result<String, String> {
    let history_url = state.page_mut().with_dom(|_| ()).map(|_| ());
    let _ = history_url;
    // We track simple page history on the Page itself; navigate to the
    // entry before the cursor.
    let page = state.page_mut();
    if page.history.len() < 2 || page.history_index == 0 {
        return Err("No previous page in history.".to_string());
    }
    let prev_idx = page.history_index - 1;
    let url = page.history[prev_idx].clone();
    page.set_history_index(prev_idx);
    let condition = obscura_browser::lifecycle::WaitUntil::DomContentLoaded;
    let stash = (page.history.clone(), page.history_index);
    state.interactive_refs.clear();
    state.page_mut().navigate_with_wait(&url, condition).await.map_err(|e| e.to_string())?;
    let page = state.page_mut();
    page.history = stash.0;
    page.history_index = stash.1;
    Ok(format!("Back to {url}"))
}

async fn tool_forward(state: &mut BrowserState) -> Result<String, String> {
    let page = state.page_mut();
    if page.history_index + 1 >= page.history.len() {
        return Err("No forward page in history.".to_string());
    }
    let next_idx = page.history_index + 1;
    let url = page.history[next_idx].clone();
    page.set_history_index(next_idx);
    let condition = obscura_browser::lifecycle::WaitUntil::DomContentLoaded;
    let stash = (page.history.clone(), page.history_index);
    state.interactive_refs.clear();
    state.page_mut().navigate_with_wait(&url, condition).await.map_err(|e| e.to_string())?;
    let page = state.page_mut();
    page.history = stash.0;
    page.history_index = stash.1;
    Ok(format!("Forward to {url}"))
}

async fn tool_reload(state: &mut BrowserState) -> Result<String, String> {
    let url = state.page_mut().url_string();
    if url == "about:blank" {
        return Err("Nothing to reload.".to_string());
    }
    let condition = obscura_browser::lifecycle::WaitUntil::DomContentLoaded;
    state.interactive_refs.clear();
    state.page_mut().navigate_with_wait(&url, condition).await.map_err(|e| e.to_string())?;
    Ok(format!("Reloaded {url}"))
}

fn tool_get_cookies(args: &Value, state: &BrowserState) -> Result<String, String> {
    let domain_filter = args.get("domain").and_then(Value::as_str);
    let cookies = state.context.cookie_jar.get_all_cookies();
    let lines: Vec<String> = cookies.iter()
        .filter(|c| domain_filter.is_none_or(|d| c.domain == obscura_net::canonical_domain(d)))
        .map(|c| serde_json::to_string(&json!({
            "name": c.name,
            "value": c.value,
            "domain": c.domain,
            "path": c.path,
            "secure": c.secure,
            "http_only": c.http_only,
        })).unwrap_or_default())
        .collect();
    if lines.is_empty() {
        Ok("No cookies.".to_string())
    } else {
        Ok(lines.join("\n"))
    }
}

fn tool_set_cookie(args: &Value, state: &BrowserState) -> Result<String, String> {
    let name = args.get("name").and_then(Value::as_str)
        .ok_or("Missing name parameter")?;
    let value = args.get("value").and_then(Value::as_str)
        .ok_or("Missing value parameter")?;
    let domain = args.get("domain").and_then(Value::as_str)
        .ok_or("Missing domain parameter")?;
    let path = args.get("path").and_then(Value::as_str).unwrap_or("/");
    let secure = args.get("secure").and_then(Value::as_bool).unwrap_or(false);
    let http_only = args.get("http_only").and_then(Value::as_bool).unwrap_or(false);
    let cookie = obscura_net::CookieInfo {
        name: name.to_string(),
        value: value.to_string(),
        domain: domain.to_string(),
        path: path.to_string(),
        secure,
        http_only,
        same_site: String::new(),
        expires: None,
    };
    state.context.cookie_jar.set_cookies_from_cdp(vec![cookie]);
    Ok(format!("Set cookie {name} on {domain}{path}"))
}

fn tool_clear_cookies(state: &BrowserState) -> Result<String, String> {
    state.context.cookie_jar.clear();
    Ok("Cleared all cookies.".to_string())
}

async fn tool_wait_for_text(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let needle = args.get("text").and_then(Value::as_str)
        .ok_or("Missing text parameter")?;
    let timeout_secs = args.get("timeout").and_then(Value::as_f64).unwrap_or(30.0) as u64;
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);
    let escaped = serde_json::to_string(needle).unwrap_or_else(|_| "\"\"".to_string());
    let js = format!(r#"(function(){{
        var t = (document.body && (document.body.innerText || document.body.textContent)) || '';
        return t.indexOf({needle}) >= 0;
    }})()"#, needle = escaped);
    // Exponential backoff like browser_wait_for (see comment there).
    let mut tick_ms: u64 = 5;
    loop {
        let found = evaluate_tool_script(state.page_mut(), &js)?.as_bool().unwrap_or(false);
        if found {
            return Ok(format!("Found text {needle:?}"));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!("Timeout waiting for text {needle:?}"));
        }
        let tick = tokio::time::Duration::from_millis(tick_ms);
        match tokio::time::timeout(tick, state.advance_active_page_tasks()).await {
            Ok(result) => {
                result?;
            }
            Err(_) => {}
        }
        if tick_ms < 200 { tick_ms = (tick_ms * 2).min(200); }
    }
}

// ===== Tier 2 agent-UX additions =====

/// Describe every <form> on the page. For each form, list its action,
/// method, and every field (input/select/textarea) with its name, type,
/// current value, and any visible label text. Agents call this to
/// understand a form's shape before filling it.
fn tool_detect_forms(state: &mut BrowserState) -> Result<String, String> {
    let page = state.page_mut();
    let js = r#"(function(){
        var forms = document.querySelectorAll('form');
        var out = [];
        for (var i = 0; i < forms.length; i++) {
            var f = forms[i];
            var fields = [];
            var inputs = f.querySelectorAll('input, select, textarea, button');
            for (var j = 0; j < inputs.length; j++) {
                var el = inputs[j];
                var tag = el.tagName.toLowerCase();
                var type = (el.getAttribute('type') || (tag === 'input' ? 'text' : tag)).toLowerCase();
                if (tag === 'input' && type === 'hidden') continue;
                var name = el.getAttribute('name') || '';
                var label = '';
                if (el.id) {
                    var lab = document.querySelector('label[for="' + el.id + '"]');
                    if (lab) label = (lab.innerText || lab.textContent || '').trim();
                }
                if (!label) label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || '';
                // Same redaction as browser_structure: sensitive fields never report a value.
                var sensitive = type === 'password'
                    || /(pass(?:word)?|secret|token|api[_-]?key|auth|credential|card|cvv)/i.test((el.id || '') + ' ' + name + ' ' + (el.getAttribute('autocomplete') || ''));
                var opts = null;
                if (tag === 'select') {
                    opts = [];
                    var os = el.querySelectorAll('option');
                    for (var k = 0; k < os.length; k++) {
                        opts.push({ value: os[k].value, text: (os[k].textContent || '').trim() });
                    }
                }
                fields.push({
                    tag: tag,
                    type: type,
                    name: name,
                    value: sensitive ? null : (el.value || ''),
                    checked: el.checked || false,
                    required: el.required || false,
                    label: label.trim().slice(0, 100),
                    ref: el.getAttribute('data-obscura-ref') || null,
                    options: opts,
                });
            }
            out.push({
                index: i,
                id: f.id || '',
                name: f.getAttribute('name') || '',
                action: f.action || '',
                method: (f.method || 'get').toLowerCase(),
                fields: fields,
            });
        }
        return out;
    })()"#;
    let val = evaluate_tool_script(page, js)?;
    if val.is_null() {
        return Ok("No forms found.".to_string());
    }
    serde_json::to_string_pretty(&val).map_err(|e| e.to_string())
}

// ---- page-structure extraction (port of camofox lib/page-structure.js) ----
// Read-only structured view of forms and tables, with sensitive-field value
// redaction. Implemented natively on DomTree (no JS round-trip).

const STRUCTURE_MAX_FORMS: usize = 20;
const STRUCTURE_MAX_FIELDS: usize = 50;
const STRUCTURE_MAX_TABLES: usize = 10;
const STRUCTURE_MAX_ROWS: usize = 50;
const STRUCTURE_MAX_ROWS_READ: usize = 200;
const STRUCTURE_MAX_CELLS: usize = 20;
const STRUCTURE_MAX_OPTIONS: usize = 50;
const STRUCTURE_MAX_TEXT: usize = 300;

fn structure_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(STRUCTURE_MAX_TEXT).collect()
}

/// camofox SENSITIVE_FIELD_PATTERN: /(pass(?:word)?|secret|token|api[_-]?key|
/// auth|credential|card|cvv)/i over `id name autocomplete`, plus type=password.
fn structure_is_sensitive(dom: &obscura_dom::DomTree, node: NodeId) -> bool {
    let Some(node) = dom.get_node(node) else { return false };
    if node.get_attribute("type").is_some_and(|t| t.eq_ignore_ascii_case("password")) {
        return true;
    }
    let haystack = format!("{} {} {}",
        node.get_attribute("id").unwrap_or(""),
        node.get_attribute("name").unwrap_or(""),
        node.get_attribute("autocomplete").unwrap_or(""))
        .to_lowercase();
    ["pass", "secret", "token", "apikey", "api_key", "api-key", "auth", "credential", "card", "cvv"]
        .iter().any(|needle| haystack.contains(needle))
}

/// Label chain: aria-label -> aria-labelledby -> label[for] -> wrapping <label>
/// -> title/name/id. `labels_by_for` maps `for` attr to label node, built once
/// per call instead of a document query per control.
fn structure_label(dom: &obscura_dom::DomTree, node_id: NodeId, labels_by_for: &HashMap<String, NodeId>) -> String {
    let attr = |name: &str| dom.get_node(node_id).and_then(|n| n.get_attribute(name).map(str::to_owned));
    if let Some(aria) = attr("aria-label") {
        if !aria.trim().is_empty() { return structure_text(&aria); }
    }
    if let Some(ids) = attr("aria-labelledby") {
        let labels: Vec<String> = ids.split_whitespace()
            .filter_map(|id| dom.get_element_by_id(id))
            .map(|label| structure_text(&dom.text_content(label)))
            .filter(|label| !label.is_empty())
            .collect();
        if !labels.is_empty() { return labels.join(" "); }
    }
    if let Some(id) = attr("id") {
        if let Some(label) = labels_by_for.get(&id) {
            let text = structure_text(&dom.text_content(*label));
            if !text.is_empty() { return text; }
        }
    }
    for ancestor in dom.ancestors(node_id) {
        let is_label = dom.get_node(ancestor)
            .and_then(|n| n.as_element().map(|q| q.local.as_ref() == "label"))
            .unwrap_or(false);
        if is_label { return structure_text(&dom.text_content(ancestor)); }
    }
    structure_text(&attr("title").or_else(|| attr("name")).or_else(|| attr("id")).unwrap_or_default())
}

fn structure_disabled(dom: &obscura_dom::DomTree, node_id: NodeId) -> bool {
    if dom.get_node(node_id).map(|n| n.get_attribute("disabled").is_some()).unwrap_or(false) { return true; }
    // A control inside a disabled fieldset is disabled too.
    dom.ancestors(node_id).iter().any(|&ancestor| {
        dom.get_node(ancestor)
            .map(|n| n.as_element().is_some_and(|q| q.local.as_ref() == "fieldset")
                && n.get_attribute("disabled").is_some())
            .unwrap_or(false)
    })
}

/// The value the page would report for the control: live form-control state
/// first, then the attribute, then textarea text content.
fn structure_value(dom: &obscura_dom::DomTree, node_id: NodeId) -> String {
    if let Some(value) = dom.form_control_state(node_id).and_then(|s| s.value) {
        if !value.is_empty() { return structure_text(&value); }
    }
    if let Some(value) = dom.get_node(node_id).and_then(|n| n.get_attribute("value").map(str::to_owned)) {
        return structure_text(&value);
    }
    let is_textarea = dom.get_node(node_id)
        .and_then(|n| n.as_element().map(|q| q.local.as_ref() == "textarea"))
        .unwrap_or(false);
    if is_textarea { return structure_text(&dom.text_content(node_id)); }
    String::new()
}

fn structure_labels_by_for(dom: &obscura_dom::DomTree) -> HashMap<String, NodeId> {
    dom.query_selector_all("label").unwrap_or_default().into_iter()
        .filter_map(|label| {
            let for_attr = dom.get_node(label).and_then(|n| n.get_attribute("for").map(str::to_owned))?;
            Some((for_attr, label))
        })
        .collect()
}

fn structure_field(dom: &obscura_dom::DomTree, node_id: NodeId, labels_by_for: &HashMap<String, NodeId>) -> Value {
    let node = dom.get_node(node_id);
    let tag = node.as_ref()
        .and_then(|n| n.as_element().map(|q| q.local.as_ref().to_lowercase()))
        .unwrap_or_default();
    let raw_type = node.as_ref()
        .and_then(|n| n.get_attribute("type").map(str::to_lowercase))
        .unwrap_or_else(|| if tag == "select" { "select".to_string() } else { String::new() });
    let role = if tag == "select" { "combobox" }
        else if raw_type == "checkbox" { "checkbox" }
        else if raw_type == "radio" { "radio" }
        else { "textbox" };
    let mut field = json!({
        "label": structure_label(dom, node_id, labels_by_for),
        "role": role,
        "tag": tag,
        "type": raw_type,
        "disabled": structure_disabled(dom, node_id),
    });
    if let Some(id) = node.as_ref().and_then(|n| n.get_attribute("id")) {
        if !id.is_empty() { field["id"] = json!(id); }
    }
    if let Some(name) = node.as_ref().and_then(|n| n.get_attribute("name")) {
        field["name"] = json!(name);
    }
    if let Some(reference) = node.as_ref().and_then(|n| n.get_attribute("data-obscura-ref")) {
        field["ref"] = json!(reference);
    }
    // Sensitive controls and buttons/files never report a value.
    if !structure_is_sensitive(dom, node_id) && tag != "button" && raw_type != "file" {
        field["value"] = json!(structure_value(dom, node_id));
    }
    if tag == "select" {
        let all_options = dom.select_options(node_id).unwrap_or_default();
        let options: Vec<Value> = all_options.iter().take(STRUCTURE_MAX_OPTIONS).map(|option| {
            let label = structure_text(&dom.text_content(*option));
            let value = dom.get_node(*option)
                .and_then(|n| n.get_attribute("value").map(str::to_owned))
                .unwrap_or_else(|| label.clone());
            let selected = dom.get_node(*option).map(|n| n.get_attribute("selected").is_some()).unwrap_or(false)
                || dom.form_control_state(*option).is_some_and(|s| s.checked == Some(true));
            json!({ "label": label, "value": value, "selected": selected })
        }).collect();
        field["options"] = json!(options);
        field["optionsTruncated"] = json!(all_options.len() > STRUCTURE_MAX_OPTIONS);
    }
    field
}

fn extract_page_structure(dom: &obscura_dom::DomTree) -> Value {
    let labels_by_for = structure_labels_by_for(dom);
    let all_forms = dom.query_selector_all("form").unwrap_or_default();
    let forms: Vec<Value> = all_forms.iter().take(STRUCTURE_MAX_FORMS).map(|form| {
        let elements: Vec<NodeId> = dom.query_selector_all_from(*form, "input, select, textarea")
            .unwrap_or_default().into_iter()
            .filter(|el| {
                let tag = dom.get_node(*el).and_then(|n| n.as_element().map(|q| q.local.as_ref().to_lowercase()));
                let ty = dom.get_node(*el).and_then(|n| n.get_attribute("type").map(str::to_lowercase)).unwrap_or_default();
                tag.is_some_and(|t| t == "input" || t == "select" || t == "textarea")
                    && !matches!(ty.as_str(), "hidden" | "submit" | "reset" | "button" | "image")
            })
            .collect();
        let fields: Vec<Value> = elements.iter().take(STRUCTURE_MAX_FIELDS)
            .map(|el| structure_field(dom, *el, &labels_by_for)).collect();
        let submit_controls: Vec<Value> = dom.query_selector_all_from(*form, "button, input[type=\"submit\"], input[type=\"image\"]")
            .unwrap_or_default().iter().take(STRUCTURE_MAX_FIELDS).map(|el| {
                let node = dom.get_node(*el);
                // camofox falls back to element.value; a <button>'s visible
                // name is its text, so fall back to text content after that.
                let label = {
                    let label = structure_label(dom, *el, &labels_by_for);
                    if !label.is_empty() { label }
                    else {
                        let value = structure_value(dom, *el);
                        if !value.is_empty() { value } else { structure_text(&dom.text_content(*el)) }
                    }
                };
                let mut control = json!({ "label": label, "disabled": structure_disabled(dom, *el) });
                if let Some(id) = node.as_ref().and_then(|n| n.get_attribute("id")) {
                    if !id.is_empty() { control["id"] = json!(id); }
                }
                if let Some(name) = node.as_ref().and_then(|n| n.get_attribute("name")) {
                    control["name"] = json!(name);
                }
                if let Some(reference) = node.as_ref().and_then(|n| n.get_attribute("data-obscura-ref")) {
                    control["ref"] = json!(reference);
                }
                control
            }).collect();
        let node = dom.get_node(*form);
        let mut entry = json!({
            "method": node.as_ref().and_then(|n| n.get_attribute("method")).unwrap_or("get").to_lowercase(),
            "fields": fields,
            "fieldsTruncated": elements.len() > STRUCTURE_MAX_FIELDS,
            "submitControls": submit_controls,
        });
        let label = structure_label(dom, *form, &labels_by_for);
        if !label.is_empty() { entry["label"] = json!(label); }
        if let Some(id) = node.as_ref().and_then(|n| n.get_attribute("id")) {
            if !id.is_empty() { entry["id"] = json!(id); }
        }
        if let Some(name) = node.as_ref().and_then(|n| n.get_attribute("name")) {
            entry["name"] = json!(name);
        }
        entry
    }).collect();

    let all_tables = dom.query_selector_all("table").unwrap_or_default();
    let tables: Vec<Value> = all_tables.iter().take(STRUCTURE_MAX_TABLES).enumerate().map(|(index, table)| {
        let node = dom.get_node(*table);
        let table_id = node.as_ref().and_then(|n| n.get_attribute("id")).filter(|id| !id.is_empty())
            .map(str::to_owned).unwrap_or_else(|| format!("table-{}", index + 1));
        let headers: Vec<String> = dom.query_selector_all_from(*table, "thead th").unwrap_or_default()
            .iter().map(|cell| structure_text(&dom.text_content(*cell))).collect();
        let rows = dom.query_selector_all_from(*table, "tbody tr").unwrap_or_default();
        let visible_rows: Vec<Vec<String>> = rows.iter().take(STRUCTURE_MAX_ROWS).map(|row| {
            dom.query_selector_all_from(*row, "th, td").unwrap_or_default()
                .iter().take(STRUCTURE_MAX_CELLS).map(|cell| structure_text(&dom.text_content(*cell))).collect()
        }).collect();
        let mut entry = json!({
            "tableId": table_id,
            "headers": headers,
            "columns": headers,
            "rowCount": rows.len(),
            "rows": visible_rows,
            "rowsTruncated": rows.len() > STRUCTURE_MAX_ROWS,
        });
        let label = structure_label(dom, *table, &labels_by_for);
        if !label.is_empty() { entry["label"] = json!(label); }
        if let Some(id) = node.as_ref().and_then(|n| n.get_attribute("id")) {
            if !id.is_empty() { entry["id"] = json!(id); }
        }
        entry
    }).collect();
    let evidence_limits: Vec<Value> = tables.iter().map(|table| json!({
        "source": "table",
        "tableId": table["tableId"],
        "tableLabel": table.get("label"),
        "availableDimensions": table["headers"],
        "visibleRowCount": table["rowCount"],
        "rowsTruncated": table["rowsTruncated"],
    })).collect();
    json!({
        "forms": forms,
        "formsTruncated": all_forms.len() > STRUCTURE_MAX_FORMS,
        "tables": tables,
        "tablesTruncated": all_tables.len() > STRUCTURE_MAX_TABLES,
        "evidenceLimits": evidence_limits,
    })
}

fn tool_structure(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let max_chars = args.get("max_chars").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(DEFAULT_TEXT_LIMIT);
    let offset = args.get("offset").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(0);
    rebuild_interactive_refs(state)?;
    let structure = state.page_mut().with_dom(|dom| extract_page_structure(dom)).unwrap_or_else(|| json!({}));
    let text = serde_json::to_string_pretty(&structure).map_err(|e| e.to_string())?;
    Ok(window_text(&text, max_chars, offset))
}

/// Row-paginated read of one advertised table, with rowspan/colspan expanded
/// so every row aligns cell-for-cell with the headers. Tables containing
/// credential inputs are not a data source.
fn tool_table_rows(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let requested_id = args.get("table_id").and_then(Value::as_str)
        .or_else(|| args.get("tableId").and_then(Value::as_str))
        .ok_or("Missing table_id parameter")?;
    let requested_offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let page = state.page_mut();
    page.with_dom(|dom| {
        let tables = dom.query_selector_all("table").unwrap_or_default();
        // Resolve against the same ids extract_page_structure advertises:
        // the element id when present, else the positional `table-N` fallback.
        let index = tables.iter().take(STRUCTURE_MAX_TABLES).enumerate().find_map(|(i, table)| {
            let id = dom.get_node(*table)
                .and_then(|n| n.get_attribute("id").filter(|id| !id.is_empty()).map(str::to_owned))
                .unwrap_or_else(|| format!("table-{}", i + 1));
            (id == requested_id).then_some(i)
        });
        let Some(index) = index else {
            return Err(format!("Table not found or exceeds the {STRUCTURE_MAX_TABLES}-table limit: {requested_id}"));
        };
        let table = tables[index];
        // A table that holds sensitive controls is not a readable data source.
        let sensitive_table = dom.query_selector_all_from(table, "input").unwrap_or_default().iter().any(|input| {
            dom.get_node(*input).map(|n| {
                n.get_attribute("type").is_some_and(|t| t.eq_ignore_ascii_case("password"))
                    || ["password", "secret", "token", "api_key"].iter().any(|needle| {
                        n.get_attribute("name").unwrap_or("").to_lowercase().contains(needle)
                    })
            }).unwrap_or(false)
        });
        if sensitive_table {
            return Err(format!("Table {requested_id} contains credential inputs and is not a data source"));
        }

        // Expand rowspan/colspan: carry a cell's value down through the rows it spans.
        let dom_rows = dom.query_selector_all_from(table, "tbody tr").unwrap_or_default();
        let start = requested_offset.min(dom_rows.len());
        let mut carry: HashMap<usize, (String, usize)> = HashMap::new();
        let mut grid: Vec<Vec<String>> = Vec::new();
        for row in dom_rows.iter().take(start + STRUCTURE_MAX_ROWS_READ) {
            let cells_dom = dom.query_selector_all_from(*row, "th, td").unwrap_or_default();
            let mut cells: Vec<String> = Vec::new();
            let mut column = 0usize;
            let mut cell_index = 0usize;
            while cell_index < cells_dom.len() || carry.contains_key(&column) {
                if let Some((value, remaining)) = carry.get_mut(&column) {
                    cells.push(value.clone());
                    *remaining -= 1;
                    if *remaining == 0 { carry.remove(&column); }
                    column += 1;
                    continue;
                }
                let cell = cells_dom[cell_index];
                cell_index += 1;
                let value = structure_text(&dom.text_content(cell));
                let (col_span, row_span) = dom.get_node(cell).map(|n| {
                    (n.get_attribute("colspan").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1),
                     n.get_attribute("rowspan").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1))
                }).unwrap_or((1, 1));
                for span_offset in 0..col_span {
                    let cell_value = if span_offset == 0 { value.clone() } else { String::new() };
                    cells.push(cell_value.clone());
                    if row_span > 1 { carry.insert(column + span_offset, (cell_value, row_span - 1)); }
                }
                column += col_span;
            }
            grid.push(cells.into_iter().take(STRUCTURE_MAX_CELLS).collect());
        }
        let total_rows = dom_rows.len();
        let returned: Vec<Vec<String>> = grid.into_iter().skip(start).take(STRUCTURE_MAX_ROWS_READ).collect();
        let next_offset = start + returned.len();
        let node = dom.get_node(table);
        let label = node.as_ref()
            .and_then(|n| n.get_attribute("aria-label").or_else(|| n.get_attribute("title")).or_else(|| n.get_attribute("name")).or_else(|| n.get_attribute("id")))
            .map(structure_text);
        let fields: Vec<String> = dom.query_selector_all_from(table, "thead th").unwrap_or_default()
            .iter().map(|cell| structure_text(&dom.text_content(*cell))).collect();
        let table_id = node.as_ref().and_then(|n| n.get_attribute("id")).filter(|id| !id.is_empty())
            .map(str::to_owned).unwrap_or_else(|| format!("table-{}", index + 1));
        Ok(serde_json::to_string_pretty(&json!({
            "tableId": table_id,
            "label": label,
            "fields": fields,
            "totalRows": total_rows,
            "offset": start,
            "rows": returned,
            "pagination": {
                "available": next_offset < total_rows,
                "unit": "row",
                "nextOffset": if next_offset < total_rows { json!(next_offset) } else { Value::Null },
            },
        })).map_err(|e| e.to_string())?)
    }).unwrap_or_else(|| Err("No DOM available".to_string()))
}

/// Fill multiple fields in one call. Each entry: {ref|selector, value, type?}.
/// type='text' (default) sets value, 'check'/'uncheck' toggles checkbox,
/// 'select' picks an option by value or visible text. Optional
/// `submit_ref`/`submit_selector` clicks after filling.
async fn tool_fill_form(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let fields = args.get("fields").and_then(Value::as_array)
        .ok_or("Missing fields array")?
        .clone();
    let mut filled = 0u32;
    let mut errors = Vec::new();
    for field in fields {
        let value = field.get("value").and_then(Value::as_str).unwrap_or("");
        let kind = field.get("type").and_then(Value::as_str).unwrap_or("text");
        let selector = match resolve_target(&field, state) {
            Ok(s) => s,
            Err(e) => { errors.push(e); continue; }
        };
        let js = match kind {
            "check" => format!(r#"(function(){{
                var el = document.querySelector({sel});
                if (!el) return "error:not found";
                globalThis.__obscura_setFieldValue(el, 'checked', true);
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('input', {{bubbles:true}})));
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('change', {{bubbles:true}})));
                return "ok";
            }})()"#, sel = serde_json::to_string(&selector).unwrap()),
            "uncheck" => format!(r#"(function(){{
                var el = document.querySelector({sel});
                if (!el) return "error:not found";
                globalThis.__obscura_setFieldValue(el, 'checked', false);
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('input', {{bubbles:true}})));
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('change', {{bubbles:true}})));
                return "ok";
            }})()"#, sel = serde_json::to_string(&selector).unwrap()),
            "select" => format!(r#"(function(){{
                var el = document.querySelector({sel});
                if (!el) return "error:not found";
                var want = {val};
                var matched = false;
                for (var i = 0; i < el.options.length; i++) {{
                    var o = el.options[i];
                    if (o.value === want || (o.textContent || '').trim() === want) {{
                        el.selectedIndex = i;
                        matched = true;
                        break;
                    }}
                }}
                if (!matched) return "error:no matching option";
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('input', {{bubbles:true}})));
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('change', {{bubbles:true}})));
                return "ok";
            }})()"#, sel = serde_json::to_string(&selector).unwrap(), val = serde_json::to_string(value).unwrap()),
            _ => format!(r#"(function(){{
                var el = document.querySelector({sel});
                if (!el) return "error:not found";
                globalThis.__obscura_setFieldValue(el, 'value', {val});
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('input', {{bubbles:true}})));
                globalThis.__obscura_dispatchDomEvent(el, globalThis.__obscura_markTrusted(new Event('change', {{bubbles:true}})));
                return "ok";
            }})()"#, sel = serde_json::to_string(&selector).unwrap(), val = serde_json::to_string(value).unwrap()),
        };
        let res = evaluate_tool_script(state.page_mut(), &js)?;
        match res.as_str() {
            Some("ok") => filled += 1,
            Some(e) => errors.push(format!("{selector}: {e}")),
            None => errors.push(format!("{selector}: unknown error")),
        }
    }

    if !errors.is_empty() {
        return Err(format!("Filled {filled} fields; not submitted. Errors: {}", errors.join("; ")));
    }

    // Optional submit click
    let submit_target = if args.get("submit_ref").is_some() || args.get("submit_selector").is_some() {
        let pseudo = json!({
            "ref": args.get("submit_ref"),
            "selector": args.get("submit_selector"),
        });
        Some(resolve_target(&pseudo, state)?)
    } else { None };
    if let Some(sel) = submit_target {
        let js = format!(r#"(function(){{
            var el = document.querySelector({sel});
            if (!el) return "error:not found";
            el.click();
            return "ok";
        }})()"#, sel = serde_json::to_string(&sel).unwrap());
        let result = evaluate_tool_script(state.page_mut(), &js)?;
        if result.as_str() != Some("ok") {
            return Err(format!("Submit target not found: {sel}"));
        }
    }

    state.settle_synthetic_navigation().await?;
    Ok(format!("Filled {filled} fields."))
}

/// Scroll the page (or an element) by direction + amount, or scroll an
/// element into view. Used to trigger infinite-scroll loaders or to
/// reach off-viewport content. Returns the new scroll position.
fn tool_scroll(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let direction = args.get("direction").and_then(Value::as_str).unwrap_or("down");
    let amount = args.get("amount").and_then(Value::as_f64);

    // Element scroll-into-view path
    if args.get("ref").is_some() || args.get("selector").is_some() {
        let selector = resolve_target(args, state)?;
        let js = format!(r#"(function(){{
            var el = document.querySelector({sel});
            if (!el) return "error:not found";
            el.scrollIntoView({{behavior:'instant', block:'center'}});
            return JSON.stringify({{x: window.scrollX, y: window.scrollY}});
        }})()"#, sel = serde_json::to_string(&selector).unwrap());
        let res = evaluate_tool_script(state.page_mut(), &js)?;
        if res.as_str() == Some("error:not found") {
            return Err(format!("Element not found: {selector}"));
        }
        return Ok(format!("Scrolled element into view. {}", res.as_str().unwrap_or("")));
    }

    // Page-level scroll. Also dispatch a 'scroll' event so infinite-
    // scroll handlers fire (we don't have a real layout engine, so the
    // window.scrollY value won't change but the event is what matters).
    let amt = amount.unwrap_or(720.0);
    let js = format!(r#"(function(){{
        var dir = {dir};
        var amt = {amt};
        switch (dir) {{
            case 'top': window.scrollTo(0, 0); break;
            case 'bottom': window.scrollTo(0, document.body.scrollHeight); break;
            case 'up': window.scrollBy(0, -amt); break;
            case 'down': window.scrollBy(0, amt); break;
            case 'left': window.scrollBy(-amt, 0); break;
            case 'right': window.scrollBy(amt, 0); break;
        }}
        try {{ window.dispatchEvent(new Event('scroll', {{bubbles:true}})); }} catch(e) {{}}
        try {{ document.dispatchEvent(new Event('scroll', {{bubbles:true}})); }} catch(e) {{}}
        return JSON.stringify({{x: window.scrollX, y: window.scrollY, max_y: document.body.scrollHeight, viewport_h: window.innerHeight}});
    }})()"#,
        dir = serde_json::to_string(direction).unwrap(),
        amt = amt,
    );
    let res = evaluate_tool_script(state.page_mut(), &js)?;
    Ok(format!("Scrolled {direction}. {}", res.as_str().unwrap_or("")))
}

fn tool_get_attribute(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let selector = resolve_target(args, state)?;
    let attr = args.get("attribute").and_then(Value::as_str)
        .ok_or("Missing attribute parameter")?;
    let js = format!(r#"(function(){{
        var el = document.querySelector({sel});
        if (!el) return null;
        var v = el.getAttribute({a});
        if (v === null && {a} === 'value') v = el.value || '';
        return v == null ? '' : v;
    }})()"#,
        sel = serde_json::to_string(&selector).unwrap(),
        a = serde_json::to_string(attr).unwrap(),
    );
    let res = evaluate_tool_script(state.page_mut(), &js)?;
    if res.is_null() {
        return Err(format!("Element not found: {selector}"));
    }
    Ok(res.as_str().unwrap_or("").to_string())
}

fn tool_count(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let selector = args.get("selector").and_then(Value::as_str)
        .ok_or("Missing selector parameter")?;
    let js = format!(
        "document.querySelectorAll({sel}).length",
        sel = serde_json::to_string(selector).unwrap()
    );
    let res = evaluate_tool_script(state.page_mut(), &js)?;
    // V8 numbers come back as f64 even when they are integer-valued; as_u64
    // returns None for f64 in serde_json, so coerce via f64.
    let n = res.as_u64()
        .or_else(|| res.as_f64().map(|f| f as u64))
        .unwrap_or(0);
    Ok(n.to_string())
}

/// Extract structured data: `schema` is a map of field name to CSS
/// selector. Suffix selector with `@attr` to read an attribute instead
/// of text. Suffix field name with `[]` to return an array (queries all
/// matching elements rather than the first).
fn tool_extract(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let schema = args.get("schema").and_then(Value::as_object)
        .ok_or("Missing schema object")?
        .clone();
    // JSON-schema mode (camofox extractDeterministic): properties resolve by
    // `x-ref` against the refs of the last browser_snapshot.
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        return tool_extract_xref(&schema, properties, state);
    }
    let schema_json = serde_json::to_string(&schema).unwrap();
    let js = format!(r#"(function(){{
        var schema = {schema};
        var out = {{}};
        for (var key in schema) {{
            if (!Object.prototype.hasOwnProperty.call(schema, key)) continue;
            var spec = schema[key];
            var is_array = key.endsWith('[]');
            var name = is_array ? key.slice(0, -2) : key;
            // Selector may end with `@attr` to read an attribute.
            var attr = null;
            var sel = spec;
            var at = spec.lastIndexOf('@');
            if (at > 0 && spec.indexOf(' ', at) < 0) {{
                attr = spec.slice(at + 1);
                sel = spec.slice(0, at);
            }}
            var get = function(el) {{
                if (!el) return null;
                if (attr) return el.getAttribute(attr) || '';
                return ((el.innerText || el.textContent) || '').trim();
            }};
            if (is_array) {{
                var els = document.querySelectorAll(sel);
                var arr = [];
                for (var i = 0; i < els.length; i++) arr.push(get(els[i]));
                out[name] = arr;
            }} else {{
                out[name] = get(document.querySelector(sel));
            }}
        }}
        return out;
    }})()"#, schema = schema_json);
    let res = evaluate_tool_script(state.page_mut(), &js)?;
    serde_json::to_string_pretty(&res).map_err(|e| e.to_string())
}

const XREF_SUPPORTED_TYPES: &[&str] = &["string", "number", "integer", "boolean", "object", "null"];

fn coerce_xref_value(raw: &str, ty: &str) -> Option<Value> {
    match ty {
        "" | "string" => Some(json!(raw.trim())),
        "number" => {
            let filtered: String = raw
                .chars()
                .filter(|c| c.is_ascii_digit() || ".eE+-".contains(*c))
                .collect();
            filtered.parse::<f64>().ok().map(|n| json!(n))
        }
        "integer" => {
            let filtered: String = raw
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '-')
                .collect();
            filtered.parse::<i64>().ok().map(|n| json!(n))
        }
        "boolean" => match raw.trim().to_lowercase().as_str() {
            "true" | "yes" | "1" => Some(json!(true)),
            "false" | "no" | "0" => Some(json!(false)),
            _ => None,
        },
        _ => Some(json!(raw)),
    }
}

/// Port of camofox `extractDeterministic`: each property's `x-ref` names a
/// ref from the last `browser_snapshot`; the resolved element's text is
/// coerced to the declared type. Required properties that cannot be
/// extracted abort the call.
fn tool_extract_xref(
    schema: &serde_json::Map<String, Value>,
    properties: &serde_json::Map<String, Value>,
    state: &mut BrowserState,
) -> Result<String, String> {
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("top-level schema must have type: object".to_string());
    }
    for (prop, def) in properties {
        let def_obj = def.as_object()
            .ok_or_else(|| format!("property \"{prop}\" must be an object"))?;
        if let Some(ty) = def_obj.get("type").and_then(Value::as_str) {
            if !XREF_SUPPORTED_TYPES.contains(&ty) {
                return Err(format!("property \"{prop}\" has unsupported type \"{ty}\""));
            }
        }
    }
    let required: Vec<&str> = schema.get("required").and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut out = serde_json::Map::new();
    for (prop, def) in properties {
        let ref_id = def.get("x-ref").and_then(Value::as_str);
        let ty = def.get("type").and_then(Value::as_str).unwrap_or("string");
        let entry = ref_id.and_then(|r| state.interactive_refs.get(r))
            .map(|e| (e.node_id, e.allocation_generation));
        let value = entry.and_then(|(node_id, generation)| {
            state.page_mut().with_dom(|dom| {
                // A stale ref may point at a reused arena slot.
                if dom.node_allocation_generation(node_id) != Some(generation) {
                    return None;
                }
                let text = dom.text_content(node_id);
                if ty == "object" {
                    return Some(json!(text));
                }
                coerce_xref_value(&text, ty)
            })
        }).flatten().unwrap_or(Value::Null);
        if value == Value::Null && required.contains(&prop.as_str()) {
            return Err(format!(
                "required property \"{prop}\" could not be extracted (x-ref={})",
                ref_id.unwrap_or("n/a")
            ));
        }
        out.insert(prop.clone(), value);
    }
    serde_json::to_string_pretty(&Value::Object(out)).map_err(|e| e.to_string())
}

async fn tool_tab_new(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let url = args.get("url").and_then(Value::as_str);
    let previous_active = state.active_tab.clone();
    let id = state.new_tab();
    if let Some(u) = url {
        let ua = state.user_agent.clone();
        let page = state.page_mut();
        if let Some(ref ua) = ua {
            page.http_client.set_user_agent(ua).await;
        }
        // camofox new-page recovery: a failed first navigation must not leave
        // an unregistered/orphan tab behind.
        match page.navigate_with_wait(u, obscura_browser::lifecycle::WaitUntil::DomContentLoaded)
            .await
        {
            Ok(()) => Ok(format!("Opened {id} and navigated to {}", page.url_string())),
            Err(e) => {
                state.tabs.remove(&id);
                state.active_tab = previous_active.filter(|t| state.tabs.contains_key(t));
                state.interactive_refs.clear();
                Err(e.to_string())
            }
        }
    } else {
        Ok(format!("Opened {id} (about:blank)."))
    }
}

fn tool_tab_list(state: &BrowserState) -> Result<String, String> {
    if state.tabs.is_empty() {
        return Ok("No tabs open.".to_string());
    }
    let lines: Vec<String> = state.tabs.iter().map(|(id, page)| {
        let active = if Some(id) == state.active_tab.as_ref() { "*" } else { " " };
        let url = page.url_string();
        let title = page.title.replace('\n', " ");
        format!("{active} {id}  {url}  \"{title}\"")
    }).collect();
    Ok(lines.join("\n"))
}

fn tool_tab_switch(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let tab_id = args.get("tab_id").and_then(Value::as_str)
        .ok_or("Missing tab_id parameter")?;
    if !state.tabs.contains_key(tab_id) {
        return Err(format!("No such tab: {tab_id}"));
    }
    state.active_tab = Some(tab_id.to_string());
    state.interactive_refs.clear();
    Ok(format!("Active tab: {tab_id}"))
}

fn tool_tab_close(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let tab_id = args.get("tab_id").and_then(Value::as_str)
        .map(String::from)
        .or_else(|| state.active_tab.clone())
        .ok_or("No tab to close")?;
    if !state.close_tab(&tab_id) {
        return Err(format!("No such tab: {tab_id}"));
    }
    if state.active_tab.as_deref() == Some(&tab_id) {
        // Promote some remaining tab to active, if any.
        state.active_tab = state.tabs.keys().next().cloned();
        state.interactive_refs.clear();
    }
    let summary = if let Some(ref a) = state.active_tab {
        format!("Closed {tab_id}. Active tab now {a}.")
    } else {
        format!("Closed {tab_id}. No tabs remain.")
    };
    Ok(summary)
}

fn is_html_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\x0C' | '\r' | ' ')
}

fn append_text_segment(result: &mut String, pending_space: &mut bool, contents: &str) {
    let trimmed = contents.trim_matches(is_html_whitespace);
    if trimmed.is_empty() {
        if contents.chars().any(is_html_whitespace) {
            *pending_space = true;
        }
        return;
    }

    let begins_with_space = contents.chars().next().is_some_and(is_html_whitespace);
    let result_ends_with_space = result.chars().next_back().is_some_and(char::is_whitespace);
    if (*pending_space || begins_with_space) && !result.is_empty() && !result_ends_with_space {
        result.push(' ');
    }
    result.push_str(trimmed);
    *pending_space = contents.chars().next_back().is_some_and(is_html_whitespace);
}

fn extract_text(dom: &obscura_dom::DomTree, node_id: obscura_dom::NodeId, context: Option<&obscura_browser::accessibility::AccessibilityContext>) -> String {
    use obscura_dom::NodeData;

    enum Work {
        Visit(obscura_dom::NodeId),
        Newline,
    }

    const MAX_NODES: usize = 5_000_000;

    let mut result = String::new();
    let mut pending_space = false;
    let mut stack = vec![Work::Visit(node_id)];
    let mut visited = 0usize;

    while let Some(work) = stack.pop() {
        let id = match work {
            Work::Newline => {
                result.push('\n');
                pending_space = false;
                continue;
            }
            Work::Visit(id) => id,
        };

        visited += 1;
        if visited > MAX_NODES {
            break;
        }

        let node = match dom.get_node(id) {
            Some(node) => node,
            None => continue,
        };

        match &node.data {
            NodeData::Text { contents } => {
                if context.is_none_or(|context| !context.is_hidden(id)) {
                    append_text_segment(&mut result, &mut pending_space, contents);
                }
            }
            NodeData::Element { name, .. } => {
                let tag = name.local.as_ref();
                if matches!(tag, "script" | "style" | "noscript" | "template") {
                    continue;
                }

                let is_block = matches!(
                    tag,
                    "div" | "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                        | "li" | "tr" | "br" | "hr" | "section" | "article"
                        | "header" | "footer" | "nav" | "main" | "aside"
                        | "blockquote" | "pre" | "ul" | "ol" | "table"
                );

                if is_block {
                    result.push('\n');
                    pending_space = false;
                    stack.push(Work::Newline);
                }
                for child in dom.children(id).into_iter().rev() {
                    stack.push(Work::Visit(child));
                }
            }
            _ => {
                for child in dom.children(id).into_iter().rev() {
                    stack.push(Work::Visit(child));
                }
            }
        }
    }

    result
}

// ===== Tier 3 agent-UX additions =====

/// Substring search in visible page text. Returns each match with N chars
/// of surrounding context so the agent can locate the section without
/// pulling the whole page into its window.
fn tool_search(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let query = args.get("query").and_then(Value::as_str)
        .ok_or("Missing query parameter")?;
    let case_sensitive = args.get("case_sensitive").and_then(Value::as_bool).unwrap_or(false);
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
    let context = args.get("context_chars").and_then(Value::as_u64).unwrap_or(80) as usize;

    let page = state.page_mut();
    let body = page.with_dom(|dom| {
        dom.query_selector("body").ok().flatten()
            .map(|b| extract_text(dom, b, None))
            .unwrap_or_default()
    }).unwrap_or_default();

    let out = search_body(&body, query, case_sensitive, context, limit);
    if out.is_empty() {
        Ok(format!("No matches for {query:?}."))
    } else {
        Ok(format!("{} match(es). {}", out.len(),
            out.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n")))
    }
}

/// Case-(in)sensitive substring search returning `{offset, snippet}` per match,
/// where `offset` is a byte offset into `body` and the snippet has `context`
/// bytes of surrounding text snapped to word boundaries.
fn search_body(
    body: &str,
    query: &str,
    case_sensitive: bool,
    context: usize,
    limit: usize,
) -> Vec<serde_json::Value> {
    // Build the case-folded haystack and, for the case-insensitive path, a map
    // from each haystack byte offset back to the corresponding byte offset in
    // `body`. to_lowercase() can change byte length (e.g. "İ" U+0130 -> "i̇"),
    // so a haystack byte offset is not a valid index into the original body.
    let (haystack, offset_map) = if case_sensitive {
        (body.to_string(), None)
    } else {
        let mut hs = String::with_capacity(body.len());
        let mut map: Vec<usize> = Vec::with_capacity(body.len() + 1);
        for (byte_idx, ch) in body.char_indices() {
            for lc in ch.to_lowercase() {
                let mut buf = [0u8; 4];
                let encoded = lc.encode_utf8(&mut buf);
                for _ in 0..encoded.len() {
                    map.push(byte_idx);
                }
                hs.push_str(encoded);
            }
        }
        map.push(body.len());
        (hs, Some(map))
    };
    let needle = if case_sensitive { query.to_string() } else { query.to_lowercase() };
    let to_body = |hs_off: usize| -> usize {
        match &offset_map {
            Some(map) => map.get(hs_off).copied().unwrap_or(body.len()),
            None => hs_off,
        }
    };

    let mut out = Vec::new();
    let mut idx = 0;
    while let Some(pos) = haystack[idx..].find(&needle) {
        let hs_abs = idx + pos;
        let match_start = to_body(hs_abs);
        let match_end = to_body(hs_abs + needle.len());
        let mut start = match_start.saturating_sub(context);
        let mut end = (match_end + context).min(body.len());
        // start/end can land inside a multi-byte character; snap to char
        // boundaries before slicing or body[..start] panics (#257).
        while start > 0 && !body.is_char_boundary(start) { start -= 1; }
        while end < body.len() && !body.is_char_boundary(end) { end += 1; }
        // Trim inward to the nearest whitespace so snippets start/end on words.
        if let Some(i) = body[..start].rfind(|c: char| c.is_whitespace()) {
            start = i + body[i..].chars().next().map_or(1, char::len_utf8);
        }
        if let Some(i) = body[end..].find(|c: char| c.is_whitespace()) {
            end += i;
        }
        let snippet = body.get(start..end).unwrap_or("").trim().replace('\n', " ");
        out.push(json!({
            "offset": match_start,
            "snippet": snippet,
        }));
        idx = hs_abs + needle.len();
        if out.len() >= limit { break; }
    }
    out
}

const KEENABLE_SEARCH_URL: &str = "https://api.keenable.ai/v1/search";
const KEENABLE_PUBLIC_SEARCH_URL: &str = "https://api.keenable.ai/v1/search/public";
const WEB_SEARCH_SNIPPET_CHARS: usize = 500;
// Room for 50 results with multi-KB snippets, bounded before JSON parsing.
const WEB_SEARCH_RESPONSE_BYTES: usize = 1024 * 1024;

/// `web_search` sends the agent's queries to a third-party API, so it is
/// listed and callable only when the operator opts in.
fn web_search_enabled() -> bool {
    std::env::var("OBSCURA_WEB_SEARCH")
        .is_ok_and(|provider| provider.trim().eq_ignore_ascii_case("keenable"))
}

struct WebSearchRequest {
    url: &'static str,
    headers: Vec<(&'static str, String)>,
    body: String,
}

/// The public endpoint needs no account. A key only raises the rate limit.
fn web_search_request(query: &str, limit: u64, api_key: Option<&str>) -> WebSearchRequest {
    let mut headers = vec![
        ("content-type", "application/json".to_string()),
        ("x-keenable-title", "Obscura".to_string()),
    ];
    let url = match api_key {
        Some(key) => {
            headers.push(("x-api-key", key.to_string()));
            KEENABLE_SEARCH_URL
        }
        None => KEENABLE_PUBLIC_SEARCH_URL,
    };
    WebSearchRequest {
        url,
        headers,
        body: json!({ "query": query, "max_results": limit }).to_string(),
    }
}

fn web_search_limit(args: &Value) -> u64 {
    args.get("limit").and_then(|value| {
        value.as_u64().or_else(|| {
            // JSON numbers such as 1.0 are stored as f64, even when integral.
            value.as_f64()
                .filter(|number| number.is_finite() && *number >= 0.0 && number.fract() == 0.0)
                .map(|number| number as u64)
        })
    }).unwrap_or(5).clamp(1, 50)
}

async fn tool_web_search(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let query = args.get("query").and_then(Value::as_str).map(str::trim)
        .filter(|query| !query.is_empty())
        .ok_or("Missing query parameter")?;
    let limit = web_search_limit(args);
    let api_key = std::env::var("KEENABLE_API_KEY").ok()
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty());
    let request = web_search_request(query, limit, api_key.as_deref());

    let body = fetch_web_search(request, state, api_key.is_none()).await?;
    Ok(format_web_results(query, &body))
}

async fn fetch_web_search(
    request: WebSearchRequest,
    state: &BrowserState,
    keyless: bool,
) -> Result<Value, String> {
    // The context's client carries the configured proxy, CA roots and SSRF guard.
    let client = state.context.http_client.request_client().await;
    let mut builder = client.post(request.url).body(request.body);
    for (name, value) in request.headers {
        builder = builder.header(name, value);
    }
    let response = builder.send().await
        .map_err(|error| format!("web search request failed: {error}"))?;
    let status = response.status();
    let response_url = response.url().clone();
    let body = obscura_net::client::read_reqwest_body_limited(
        response, &response_url, WEB_SEARCH_RESPONSE_BYTES,
    ).await
        .map_err(|error| format!("web search response failed: {error}"))?;
    if !status.is_success() {
        let hint = if status.as_u16() == 429 && keyless {
            " (set KEENABLE_API_KEY for higher rate limits)"
        } else {
            ""
        };
        return Err(format!("web search returned HTTP {}{hint}", status.as_u16()));
    }
    serde_json::from_slice(&body)
        .map_err(|error| format!("web search returned invalid JSON: {error}"))
}

/// One JSON object per line, like `browser_links`. Keenable returns the page
/// text in `snippet` (`description` is empty), often several KB of it.
fn format_web_results(query: &str, body: &Value) -> String {
    let compact = |text: &str, max_chars: usize| {
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.chars().count() <= max_chars {
            return text;
        }
        let mut cut: String = text.chars().take(max_chars).collect();
        cut.push_str("...");
        cut
    };
    let lines: Vec<String> = body.get("results").and_then(Value::as_array).into_iter().flatten()
        .filter_map(|result| {
            let url = result.get("url").and_then(Value::as_str)?;
            let field = |name: &str| result.get(name).and_then(Value::as_str).unwrap_or("");
            Some(json!({
                "title": compact(field("title"), 200),
                "url": url,
                "snippet": compact(field("snippet"), WEB_SEARCH_SNIPPET_CHARS),
            }).to_string())
        })
        .collect();
    if lines.is_empty() {
        format!("No results for {query:?}.")
    } else {
        lines.join("\n")
    }
}

/// Export full session state: cookies + localStorage + sessionStorage
/// for every origin the page knows about. Agents stash this between
/// runs to skip a login flow.
/// Write `bytes` to `path` via a sibling tmp file + rename (camofox's
/// persistStorageState pattern), creating parents and keeping the file
/// owner-only: a storage-state file holds live auth material.
fn write_storage_state_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("storage-state.json");
    let tmp = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// One origin's web-storage entries, emitted as Playwright's `{name, value}`
/// objects.
fn storage_entries_json(entries: Vec<(String, String)>) -> Value {
    json!(entries.into_iter().map(|(k, v)| json!({ "name": k, "value": v })).collect::<Vec<_>>())
}

fn tool_storage_state(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    let cookies: Vec<Value> = state.context.cookie_jar.get_all_cookies().iter().map(|c| json!({
        "name": c.name,
        "value": c.value,
        "domain": c.domain,
        "path": c.path,
        "secure": c.secure,
        "httpOnly": c.http_only,
        "sameSite": c.same_site,
        "expires": c.expires,
    })).collect();
    let mut origins = std::collections::BTreeMap::new();
    for (origin, entries) in state.context.local_storage.entries() {
        let entries: Vec<_> = entries.into_iter().map(|(k, v)| (decode_storage_string(k), decode_storage_string(v))).collect();
        origins.insert(origin.clone(), json!({
            "origin": origin, "localStorage": storage_entries_json(entries), "sessionStorage": []
        }));
    }
    if let Some(page) = state.active_tab.as_ref().and_then(|id| state.tabs.get(id)) {
        for (origin, entries) in page.session_storage().entries() {
            let entries: Vec<_> = entries.into_iter().map(|(k, v)| (decode_storage_string(k), decode_storage_string(v))).collect();
            let entry = origins.entry(origin.clone()).or_insert_with(|| json!({
                "origin": origin, "localStorage": [], "sessionStorage": []
            }));
            entry["sessionStorage"] = storage_entries_json(entries);
        }
    }
    let origins: Vec<_> = origins.into_values().collect();
    let out = json!({ "cookies": cookies, "origins": origins });
    let serialized = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        let path = std::path::PathBuf::from(path);
        write_storage_state_atomically(&path, serialized.as_bytes()).map_err(|e| e.to_string())?;
        return Ok(format!(
            "Wrote storage state ({} cookies, {} origins) to {}",
            cookies.len(), origins.len(), path.display(),
        ));
    }
    Ok(serialized)
}

struct StorageOrigin {
    origin: String,
    local_storage: Vec<(String, String)>,
    session_storage: Vec<(String, String)>,
}

/// Storage entries arrive either as Playwright `{name, value}` objects or as
/// `[name, value]` tuples (the shape this tool emitted before). Accept both.
fn parse_storage_entries(value: &Value) -> Result<Vec<(String, String)>, String> {
    let Some(entries) = value.as_array() else {
        return Err("storage entries must be an array".into());
    };
    entries.iter().map(|entry| {
        if let Some(object) = entry.as_object() {
            let name = object.get("name").and_then(Value::as_str);
            let value = object.get("value").and_then(Value::as_str);
            match (name, value) {
                (Some(name), Some(value)) => Ok((name.to_string(), value.to_string())),
                _ => Err("storage entries need string name and value fields".into()),
            }
        } else if let Some(pair) = entry.as_array() {
            match pair.as_slice() {
                [Value::String(k), Value::String(v)] => Ok((k.clone(), v.clone())),
                _ => Err("tuple storage entries need exactly [name, value]".into()),
            }
        } else {
            Err("storage entry must be {name, value} or [name, value]".into())
        }
    }).collect()
}

// Match the Web Storage bridge's escaping of strings that begin with NUL.
fn encode_storage_string(value: String) -> String {
    if value.starts_with('\0') { format!("\0{}", json!(value)) } else { value }
}

fn decode_storage_string(value: String) -> String {
    value.strip_prefix('\0')
        .and_then(|encoded| serde_json::from_str::<String>(encoded).ok())
        .filter(|decoded| decoded.starts_with('\0'))
        .unwrap_or(value)
}

fn apply_storage_state(s: &serde_json::Map<String, Value>, state: &mut BrowserState) -> Result<String, String> {
    let origins: Vec<StorageOrigin> = s
        .get("origins")
        .cloned()
        .unwrap_or_else(|| json!([]))
        .as_array()
        .ok_or("origins must be an array")?
        .iter()
        .map(|entry| {
            Ok(StorageOrigin {
                origin: entry.get("origin").and_then(Value::as_str)
                    .ok_or("each origin needs a string `origin` field")?.to_string(),
                local_storage: entry.get("localStorage")
                    .map(parse_storage_entries).transpose()?.unwrap_or_default(),
                session_storage: entry.get("sessionStorage")
                    .map(parse_storage_entries).transpose()?.unwrap_or_default(),
            })
        })
        .collect::<Result<_, String>>()?;
    for entry in &origins {
        if !url::Url::parse(&entry.origin).is_ok_and(|url| {
            matches!(url.scheme(), "http" | "https") && url.origin().ascii_serialization() == entry.origin
        }) {
            return Err("Storage origins must be canonical http(s) origins without paths or credentials".into());
        }
    }
    let cookies: &[Value] = match s.get("cookies") {
        Some(value) => value.as_array().ok_or("cookies must be an array")?,
        None => &[],
    };
    // Playwright emits httpOnly/sameSite; earlier exports used http_only/
    // same_site and expires as int or float (-1 or absent means session).
    let parsed: Vec<obscura_net::CookieInfo> = cookies.iter().map(|c| {
        Some(obscura_net::CookieInfo {
            name: c.get("name")?.as_str()?.to_string(),
            value: c.get("value")?.as_str()?.to_string(),
            domain: c.get("domain")?.as_str()?.to_string(),
            path: c.get("path").and_then(Value::as_str).unwrap_or("/").to_string(),
            secure: c.get("secure").and_then(Value::as_bool).unwrap_or(false),
            http_only: c.get("httpOnly").or_else(|| c.get("http_only"))
                .and_then(Value::as_bool).unwrap_or(false),
            same_site: c.get("sameSite").or_else(|| c.get("same_site"))
                .and_then(Value::as_str).unwrap_or("").to_string(),
            expires: c.get("expires").and_then(Value::as_i64)
                .or_else(|| c.get("expires").and_then(Value::as_f64).map(|f| f as i64)),
        })
    }).collect::<Option<Vec<_>>>()
        .ok_or("Each cookie requires string name, value and domain fields")?;
    let cookie_count = parsed.len();
    let mut applied = 0usize;
    for entry in origins {
        for (key, value) in entry.local_storage {
            if !state.context.local_storage.set(&entry.origin, encode_storage_string(key), encode_storage_string(value)) {
                return Err(format!("localStorage quota exceeded after restoring {applied} web storage entries; restoration may be partial"));
            }
            applied += 1;
        }
        if !entry.session_storage.is_empty() {
            let storage = state.page_mut().session_storage();
            for (key, value) in entry.session_storage {
                if !storage.set(&entry.origin, encode_storage_string(key), encode_storage_string(value)) {
                    return Err(format!("sessionStorage quota exceeded after restoring {applied} web storage entries; restoration may be partial"));
                }
                applied += 1;
            }
        }
    }
    state.context.cookie_jar.set_cookies_from_cdp(parsed);
    state.context.save_storage();
    Ok(format!("Restored {applied} web storage entries; processed {cookie_count} cookie entries."))
}

fn tool_set_storage_state(args: &Value, state: &mut BrowserState) -> Result<String, String> {
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        // Camofox's loadPersistedStorageState: require the documented shape
        // before trusting the file.
        let raw = std::fs::read_to_string(path).map_err(|e| format!("Cannot read {path}: {e}"))?;
        let parsed: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("{path} is not a storage-state JSON object: {e}"))?;
        let object = parsed.as_object()
            .ok_or_else(|| format!("{path} does not hold a storage-state object"))?;
        if !object.get("cookies").is_some_and(Value::is_array)
            || object.get("origins").is_some_and(|v| !v.is_array())
        {
            return Err(format!("{path}: storage state needs a cookies array (origins optional)"));
        }
        return apply_storage_state(object, state);
    }
    let s = args.get("state").and_then(Value::as_object)
        .ok_or("Missing state object (or `path` to a storage-state file)")?;
    apply_storage_state(s, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookie_tools_restore_public_suffix_host_sessions() {
        let mut state = BrowserState::new(None, None, false);
        tool_set_cookie(&json!({"name":"direct", "value":"one", "domain":"httpbin.org", "secure":true, "http_only":true}), &state).unwrap();
        tool_set_storage_state(&json!({"state":{"cookies":[{
            "name":"restored", "value":"two", "domain":".httpbin.org", "path":"/",
            "secure":true, "http_only":true, "same_site":"Lax", "expires":-1
        }],"origins":[]}}), &mut state).unwrap();
        let listed = tool_get_cookies(&json!({"domain":"httpbin.org"}), &state).unwrap();
        for name in ["direct", "restored"] {
            assert!(listed.contains(name), "successful cookie tools must store {name}: {listed}");
        }
        let header = state.context.cookie_jar.get_cookie_header_same_site(&url::Url::parse("https://httpbin.org/cookies").unwrap());
        assert!(header.contains("direct=one") && header.contains("restored=two"), "{header}");
        assert!(state.context.cookie_jar.get_cookie_header_same_site(&url::Url::parse("https://unrelated.httpbin.org/").unwrap()).is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tab_switch_preserves_globals_listeners_and_pending_promises() {
        let mut state = BrowserState::new(None, None, false);
        let mut tabs = Vec::new();
        for index in 0..3 {
            tool_tab_new(&json!({"url":"data:text/html,<button id=action>Increment</button>"}), &mut state).await.unwrap();
            tabs.push(state.active_tab.clone().unwrap());
            evaluate_tool_script(state.page_mut(), &format!(
                "const tabToken={index}; window.counter=40; document.querySelector('button').addEventListener('click',()=>window.counter++); window.pending=new Promise(resolve=>{{window.finishPending=resolve}}); true"
            )).unwrap();
        }
        for index in [1, 0, 2] {
            tool_tab_switch(&json!({"tab_id":tabs[index]}), &mut state).unwrap();
            assert_eq!(evaluate_tool_script(state.page_mut(), "typeof tabToken").unwrap(), json!("number"), "switching tabs must not replace a realm");
            tool_click(&json!({"selector":"#action"}), &mut state).await.unwrap();
            assert_eq!(evaluate_tool_script(state.page_mut(), "[tabToken,window.counter]").unwrap(), json!([index,41]));
            let result = tool_evaluate(&json!({"expression":"(function(){window.finishPending('retained');return window.pending})()"}), &mut state).await.unwrap();
            assert_eq!(result, "retained");
        }
        tool_close(&mut state).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn live_tabs_with_frames_can_close_in_any_order_and_reopen() {
        let mut state = BrowserState::new(None, None, false);
        let mut tabs = Vec::new();
        for index in 0..3 {
            tool_tab_new(&json!({"url":"data:text/html,<iframe srcdoc='<p>Child frame</p>'></iframe>"}), &mut state).await.unwrap();
            tabs.push(state.active_tab.clone().unwrap());
            evaluate_tool_script(state.page_mut(), &format!("document.querySelector('iframe').contentWindow.frameToken={index}")).unwrap();
        }
        tool_tab_switch(&json!({"tab_id":tabs[0]}), &mut state).unwrap();
        assert_eq!(evaluate_tool_script(state.page_mut(), "document.querySelector('iframe').contentWindow.frameToken").unwrap().as_f64(), Some(0.0));
        for index in [1,0,2] {
            tool_tab_close(&json!({"tab_id":tabs[index]}), &mut state).unwrap();
        }
        assert!(state.tabs.is_empty());
        tool_tab_new(&json!({"url":"data:text/html,<p>New session</p>"}), &mut state).await.unwrap();
        assert_eq!(evaluate_tool_script(state.page_mut(), "6*7").unwrap().as_f64(), Some(42.0));
        tool_close(&mut state).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn snapshot_references_are_actionable_without_an_extra_listing() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<label for=q>Work email</label><input id=q><button aria-label='Save account' onclick='window.saved=true'>Unhelpful text</button>").await.unwrap();
        let snapshot = tool_snapshot(&json!({}), &mut state).unwrap();
        let action = snapshot.lines().find(|line| line.starts_with("ref=") && line.contains("Save account"))
            .expect("snapshot must include an accessible name and usable reference");
        let reference = action.split_whitespace().next().unwrap().strip_prefix("ref=").unwrap();
        let reference = reference.to_string();
        assert!(snapshot.lines().any(|line| line.starts_with("ref=") && line.contains("Work email")), "associated labels must name fields: {snapshot}");
        tool_click(&json!({"ref":reference}), &mut state).await.unwrap();
        assert_eq!(state.page_mut().evaluate("window.saved"), json!(true));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn snapshot_exposes_live_states_without_password_values() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<input id=c type=checkbox checked aria-label=Consent><input id=f value=old aria-label=Account><input id=p type=password aria-label=Password><button disabled>Blocked</button><div role=tab aria-selected=true>Overview</div><button aria-expanded=false>Details</button>").await.unwrap();
        state.page_mut().evaluate("(function(){document.getElementById('c').checked=false;document.getElementById('f').value='fresh';document.getElementById('p').value='private-password';return true;})()");
        let snapshot = tool_snapshot(&json!({}), &mut state).unwrap();
        for (label, detail) in [("Consent", "checked=false"), ("Account", "value=\"fresh\""), ("Blocked", "disabled=true"), ("Overview", "selected=true"), ("Details", "expanded=false")] {
            assert!(snapshot.lines().any(|line| line.starts_with("ref=") && line.contains(label) && line.contains(detail)), "missing {label} {detail}: {snapshot}");
        }
        assert!(!snapshot.contains("private-password"), "automatic snapshots must not echo password values");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn snapshot_budget_bounds_metadata_and_never_splits_a_reference() {
        let mut state = BrowserState::new(None, None, false);
        let html = format!("<title>{}</title><button>Continue</button><p>{}</p>", "界".repeat(1000), "Body ".repeat(2000));
        state.page_mut().navigate(&format!("data:text/html,{html}")).await.unwrap();
        let snapshot = tool_snapshot(&json!({"max_chars":96}), &mut state).unwrap();
        assert!(snapshot.chars().count() < 900, "metadata must not bypass the content budget: {} characters", snapshot.chars().count());
        for line in snapshot.lines().filter(|line| line.starts_with("ref=")) {
            let reference = line.split_whitespace().next().unwrap().strip_prefix("ref=").unwrap();
            assert!(state.interactive_refs.contains_key(reference), "a truncated ref must never be advertised");
            assert!(line.contains("Continue"), "whole descriptions must fit or be omitted");
        }
        assert!(snapshot.contains("truncated") || snapshot.contains("omitted"), "omission must be explicit");
        let empty = tool_snapshot(&json!({"max_chars":0}), &mut state).unwrap();
        assert!(!empty.lines().any(|line| line.starts_with("ref=")));
        assert!(empty.chars().count() < 800);
    }

    #[test]
    fn window_text_passthrough_when_content_fits() {
        assert_eq!(window_text("short text", 4000, 0), "short text");
        assert_eq!(window_text("short text", 4000, 0), "short text");
        assert_eq!(window_text("anything", 0, 0), "");
        assert_eq!(window_text("", 4000, 0), "");
    }

    #[test]
    fn window_text_keeps_tail_and_advertises_next_offset() {
        let body: String = (0..400).map(|i| format!("line {i} padded to a reasonable length\n")).collect();
        let total = body.chars().count();
        let first = window_text(&body, 1000, 0);
        assert!(first.contains("truncated at char"), "marker must appear: {first}");
        assert!(first.contains("line 399"), "tail must keep the last lines: {first}");
        let next_offset: usize = first.split("offset=").nth(1).unwrap()
            .split_whitespace().next().unwrap().parse().unwrap();
        assert!(next_offset > 0 && next_offset < total);
        // Page through until the marker is gone; each window still carries the tail.
        let mut window = first.clone();
        let mut offset = next_offset;
        for _ in 0..40 {
            window = window_text(&body, 1000, offset);
            let Some(rest) = window.split("offset=").nth(1) else { break };
            offset = rest.split_whitespace().next().unwrap().parse().unwrap();
        }
        assert!(!window.contains("offset="), "last window has no next offset: {window}");
        assert!(window.contains("line 399"));
        for line in window.lines() {
            if line.starts_with("line ") {
                assert!(line.len() > 20, "no line may be split mid-content: {line:?}");
            }
        }
    }

    #[test]
    fn window_text_never_splits_a_line() {
        let body: String = (0..50).map(|i| format!("ref=e1-{i} button \\\"b{i}\\\" role=button\n")).collect();
        let windowed = window_text(&body, 300, 0);
        for line in windowed.lines() {
            if line.starts_with("ref=") {
                assert!(line.contains("role=button"), "ref line must be whole: {line:?}");
            }
        }
        assert!(windowed.contains("truncated at char"), "{windowed}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn snapshot_offset_pages_reach_end_of_page_controls() {
        let mut state = BrowserState::new(None, None, false);
        let buttons: String = (0..40).map(|i| format!("<button>btn{i}</button>")).collect();
        let filler = "x".repeat(6000);
        state.page_mut().navigate(&format!("data:text/html,{buttons}<p>{filler}</p><button>pagination-next</button><a href=/page/2>next page</a>")).await.unwrap();
        let first = tool_snapshot(&json!({"max_chars": 1200}), &mut state).unwrap();
        assert!(first.contains("pagination-next"), "tail must keep end-of-page controls: {first}");
        assert!(first.contains("offset="), "marker must advertise next offset");
        // Page through until the marker is gone; the tail is present in every window.
        let mut window = first;
        for _ in 0..20 {
            let Some(rest) = window.split("offset=").nth(1) else { break };
            let next: u64 = rest.split_whitespace().next().unwrap().parse().unwrap();
            window = tool_snapshot(&json!({"max_chars": 1200, "offset": next}), &mut state).unwrap();
        }
        assert!(!window.contains("offset="), "final window must not advertise another offset: {window}");
        assert!(window.contains("pagination-next"), "last window still shows the tail: {window}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn markdown_window_preserves_trailing_links() {
        let mut state = BrowserState::new(None, None, false);
        let filler = "long prose paragraph. ".repeat(600);
        state.page_mut().navigate(&format!("data:text/html,<p>{filler}</p><a href=/next>next</a>")).await.unwrap();
        let page = tool_markdown(&json!({"max_chars": 800}), &mut state).unwrap();
        assert!(page.contains("[next](/next)"), "tail keeps trailing links: {page}");
        assert!(page.contains("offset="));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_elements_window_pages_refs() {
        let mut state = BrowserState::new(None, None, false);
        let buttons: String = (0..60).map(|i| format!("<button>button number {i} with a longer label</button>")).collect();
        state.page_mut().navigate(&format!("data:text/html,{buttons}")).await.unwrap();
        let first = tool_interactive_elements(&json!({"max_chars": 800}), &mut state).unwrap();
        assert!(first.contains("button number 59"), "tail keeps the last controls: {first}");
        for line in first.lines() {
            if line.starts_with("ref=") {
                assert!(line.contains("role=button"), "no ref line may be split: {line:?}");
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn structure_reports_forms_and_redacts_sensitive_values() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(concat!(
            "data:text/html,<form id=login method=post>",
            "<label for=u>Username</label><input id=u name=user value=bob>",
            "<label for=p>Password</label><input id=p type=password name=password value=hunter2>",
            "<input name=api_token autocomplete=off value=sekrit>",
            "<select name=plan><option value=free>Free</option><option value=pro selected>Pro</option></select>",
            "<button type=submit>Sign in</button>",
            "</form>",
            "<table id=prices><thead><tr><th>Item</th><th>Cost</th></tr></thead>",
            "<tbody><tr><td>pen</td><td>3</td></tr><tr><td>book</td><td>9</td></tr></tbody></table>",
        )).await.unwrap();
        let out = tool_structure(&json!({"max_chars": 1000000}), &mut state).unwrap();
        let structure: Value = serde_json::from_str(&out).unwrap();
        let fields = structure["forms"][0]["fields"].as_array().unwrap();
        let by_name = |name: &str| fields.iter().find(|f| f["name"] == name).cloned().unwrap();
        assert_eq!(by_name("user")["label"], "Username");
        assert_eq!(by_name("user")["value"], "bob");
        let password = by_name("password");
        assert_eq!(password["label"], "Password");
        assert!(password.get("value").is_none(), "password value must be redacted: {password}");
        assert!(by_name("api_token").get("value").is_none(), "token-ish names are redacted: {fields:?}");
        let plan = by_name("plan");
        assert_eq!(plan["role"], "combobox");
        assert_eq!(plan["options"][1]["selected"], true);
        assert!(structure["forms"][0]["submitControls"][0]["label"].as_str().unwrap().contains("Sign in"));
        assert!(fields.iter().all(|f| f.get("ref").is_some()), "controls should carry snapshot refs: {fields:?}");
        let table = &structure["tables"][0];
        assert_eq!(table["tableId"], "prices");
        assert_eq!(table["headers"], json!(["Item", "Cost"]));
        assert_eq!(table["rowCount"], 2);
        assert_eq!(table["rows"][1], json!(["book", "9"]));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn structure_table_rows_expands_spans_and_pages() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(concat!(
            "data:text/html,<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody>",
            "<tr><td rowspan=2>r1a</td><td>r1b</td></tr>",
            "<tr><td>r2b</td></tr>",
            "<tr><td colspan=2>r3ab</td></tr>",
            "</tbody></table>",
        )).await.unwrap();
        let first = tool_table_rows(&json!({"table_id": "table-1", "offset": 0}), &mut state).unwrap();
        let page: Value = serde_json::from_str(&first).unwrap();
        let rows = page["rows"].as_array().unwrap();
        assert_eq!(rows[0], json!(["r1a", "r1b"]));
        assert_eq!(rows[1], json!(["r1a", "r2b"]), "rowspan value carries into the next row");
        assert_eq!(rows[2], json!(["r3ab", ""]), "colspan fills remaining cells with empty text");
        assert_eq!(page["pagination"]["available"], false);
        let second = tool_table_rows(&json!({"table_id": "table-1", "offset": 2}), &mut state).unwrap();
        let page: Value = serde_json::from_str(&second).unwrap();
        assert_eq!(page["offset"], 2);
        assert_eq!(page["rows"].as_array().unwrap().len(), 1);
        assert_eq!(page["totalRows"], 3);
        assert!(tool_table_rows(&json!({"table_id": "nope"}), &mut state).is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn structure_rejects_tables_with_credential_inputs() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<table><tbody><tr><td><input name=api_key value=x></td></tr></tbody></table>").await.unwrap();
        let error = tool_table_rows(&json!({"table_id": "table-1"}), &mut state).unwrap_err();
        assert!(error.contains("credential"), "{error}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn snapshot_omits_hidden_content_and_controls() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<p>Visible</p><div hidden><button>Hidden action</button><p>Hidden body</p></div><div aria-hidden=true>Private panel</div>").await.unwrap();
        let snapshot = tool_snapshot(&json!({}), &mut state).unwrap();
        let content = snapshot.split_once("\n\n").unwrap().1;
        for hidden in ["Hidden action", "Hidden body", "Private panel"] {
            assert!(!content.contains(hidden), "hidden content must not be exposed as visible context: {content}");
        }
        assert!(content.contains("Visible"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn snapshots_preserve_generic_interactive_text_and_hidden_referenced_names() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<div tabindex=0 onclick='window.opened=true'>Open <b>folder</b></div><span id=label hidden>Accessible details</span><button aria-labelledby=label>Other text</button>").await.unwrap();
        let snapshot = tool_snapshot(&json!({}), &mut state).unwrap();
        assert!(snapshot.lines().any(|line| line.starts_with("ref=") && line.contains("Open folder")), "unlabelled generic controls must keep their visible text: {snapshot}");
        assert!(snapshot.lines().any(|line| line.starts_with("ref=") && line.contains("Accessible details")), "hidden referenced text still supplies the visible control's name: {snapshot}");
    }

    #[cfg(feature = "render")]
    #[tokio::test(flavor = "current_thread")]
    async fn snapshots_apply_css_visibility_without_hiding_visible_overrides() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<style>.gone{display:none}.invisible{visibility:hidden}.shown{visibility:visible}</style><div class=gone><button>CSS hidden action</button></div><div class=invisible>Hidden text<button class=shown>Visible override</button></div>").await.unwrap();
        let snapshot = tool_snapshot(&json!({}), &mut state).unwrap();
        let content = snapshot.split_once("\n\n").unwrap().1;
        assert!(!content.contains("CSS hidden action") && !content.contains("Hidden text"), "{content}");
        assert!(content.lines().any(|line| line.starts_with("ref=") && line.contains("Visible override")), "a visible descendant must remain actionable: {content}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tool_serialization_deadline_recovers_the_page() {
        // A missing production watchdog must fail this nextest process, not hang CI.
        let (stop, stopped) = std::sync::mpsc::channel();
        let backstop = std::thread::spawn(move || {
            if matches!(stopped.recv_timeout(std::time::Duration::from_secs(10)), Err(std::sync::mpsc::RecvTimeoutError::Timeout)) {
                eprintln!("tool serialization exceeded its execution deadline");
                std::process::exit(1);
            }
        });
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<p>Content</p>").await.unwrap();
        assert_eq!(state.page_mut().evaluate("(function(){globalThis.savedStringify=JSON.stringify;JSON.stringify=function(){while(true){}};return true;})()"), json!(true));
        let result = tool_extract(&json!({"schema":{"text":"p"}}), &mut state);
        state.page_mut().evaluate("(function(){JSON.stringify=savedStringify;return true;})()");
        assert_eq!(state.page_mut().evaluate("6 * 7").as_f64(), Some(42.0), "the terminated page must remain usable");
        stop.send(()).unwrap();
        backstop.join().unwrap();
        assert!(result.is_err(), "serialization overrun must be a tool error");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn evaluate_failures_are_mcp_tool_errors() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<p>Evaluate</p>").await.unwrap();
        let mut responses = Vec::new();
        for expression in [
            "(function(){throw new Error('agent-fixture');})()",
            "Promise.reject(new Error('agent-fixture'))",
            "function (",
            "new Promise(function(){})",
        ] {
            let response = dispatch("tools/call", json!(1), &json!({
                "name": "browser_evaluate", "arguments": {"expression": expression, "timeoutMs": 30}
            }), &mut state).await;
            responses.push((expression, response.result.unwrap()));
        }
        for (expression, result) in responses {
            assert_eq!(result["isError"], json!(true), "failure must not be successful data: {expression}: {result}");
        }
        assert_eq!(tool_evaluate(&json!({"expression": "6 * 7"}), &mut state).await.unwrap().parse::<f64>().unwrap(), 42.0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn evaluate_runs_scripts_once_and_preserves_null_results() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<p>Evaluate</p>").await.unwrap();
        for expression in ["null", "undefined", "Promise.resolve(null)"] {
            assert_eq!(tool_evaluate(&json!({"expression": expression}), &mut state).await.unwrap(), "null");
        }
        assert_eq!(tool_evaluate(&json!({"expression": "var mcpAnswer = 21; mcpAnswer * 2", "timeoutMs": 30}), &mut state).await.unwrap().parse::<f64>().unwrap(), 42.0);
        state.page_mut().evaluate("globalThis.evaluations = 0");
        tool_evaluate(&json!({"expression": "globalThis.evaluations++; undefined", "timeoutMs": 30}), &mut state).await.unwrap();
        assert_eq!(state.page_mut().evaluate("globalThis.evaluations").as_f64(), Some(1.0));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interaction_exceptions_are_mcp_tool_errors() {
        let mut results = Vec::new();
        for (tool, arguments, setup) in [
            ("browser_click", json!({"selector":"#action"}), "document.getElementById('action').click = function(){throw new Error('action failed')};"),
            ("browser_fill", json!({"selector":"#action", "value":"value"}), "globalThis.__obscura_setFieldValue = function(){throw new Error('action failed')};"),
            ("browser_select_option", json!({"selector":"#action", "value":"a"}), "Object.defineProperty(document.getElementById('action'), 'options', {get:function(){throw new Error('action failed')}});"),
            ("browser_press_key", json!({"selector":"#action", "key":"Enter"}), "document.getElementById('action').dispatchEvent = function(){throw new Error('action failed')};"),
            ("browser_scroll", json!({"selector":"#action"}), "document.getElementById('action').scrollIntoView = function(){throw new Error('action failed')};"),
        ] {
            let mut state = BrowserState::new(None, None, false);
            state.page_mut().navigate("data:text/html,<button id=action>Continue</button>").await.unwrap();
            assert_eq!(state.page_mut().evaluate(&format!("(function(){{{setup} return true;}})()")), json!(true));
            let response = dispatch("tools/call", json!(1), &json!({"name":tool,"arguments":arguments}), &mut state).await;
            results.push((tool, response.result.unwrap()));
        }
        for (tool, result) in results {
            assert_eq!(result["isError"], json!(true), "throwing action must not report success: {tool}: {result}");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn form_errors_prevent_submission() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<input id=q><button id=go onclick='window.submits++'>Submit</button><script>window.submits=0</script>").await.unwrap();
        let response = dispatch("tools/call", json!(1), &json!({"name":"browser_fill_form", "arguments":{
            "fields":[{"selector":"#q","value":"kept"},{"selector":"#missing","value":"bad"}],
            "submit_selector":"#go"
        }}), &mut state).await;
        assert_eq!(state.page_mut().evaluate("window.submits").as_f64(), Some(0.0), "a partial form must not be submitted");
        assert_eq!(response.result.unwrap()["isError"], json!(true));
        let response = dispatch("tools/call", json!(2), &json!({"name":"browser_fill_form", "arguments":{
            "fields":[], "submit_ref":"unknown"
        }}), &mut state).await;
        assert_eq!(response.result.unwrap()["isError"], json!(true), "a missing submit target must not be ignored");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_form_submission_reaches_the_server_before_replying() {
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let (base, requests) = spawn_form_recording_server();
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(&base).await.unwrap();
        assert!(requests.recv_timeout(std::time::Duration::from_secs(2)).unwrap().starts_with("GET /"));
        let response = dispatch("tools/call", json!(1), &json!({"name":"browser_fill_form", "arguments":{
            "fields":[{"selector":"#q","value":"complete"}], "submit_selector":"#go"
        }}), &mut state).await;
        assert_ne!(response.result.unwrap()["isError"], json!(true));
        let request = requests.recv_timeout(std::time::Duration::from_secs(1)).expect("submit must complete before the reply");
        assert!(request.starts_with("POST /submitted"), "{request}");
        assert!(request.contains("q=complete"), "{request}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn storage_restore_is_origin_scoped_and_works_before_navigation() {
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let (first, _) = spawn_form_recording_server();
        let (second, _) = spawn_form_recording_server();
        let mut state = BrowserState::new(None, None, false);
        tool_set_storage_state(&json!({"state": {"origins": [
            {"origin": first, "localStorage": [["token", "first"]], "sessionStorage": [["tab", "first-session"]]},
            {"origin": second, "localStorage": [["token", "second"]], "sessionStorage": [["tab", "second-session"]]}
        ]}}), &mut state).expect("restore before the first navigation");
        for (origin, local, session) in [(&first, "first", "first-session"), (&second, "second", "second-session")] {
            state.page_mut().navigate(origin).await.expect("fixture page");
            assert_eq!(state.page_mut().evaluate("[localStorage.getItem('token'), sessionStorage.getItem('tab')]"),
                json!([local, session]), "restored values must belong to {origin}");
        }
        tool_set_storage_state(&json!({"state": {"origins": [
            {"origin": first, "localStorage": [["token", "updated-first"]]}
        ]}}), &mut state).expect("restore an inactive origin");
        assert_eq!(state.page_mut().evaluate("localStorage.getItem('token')"), json!("second"));
        state.page_mut().navigate(&first).await.expect("return to first origin");
        assert_eq!(state.page_mut().evaluate("localStorage.getItem('token')"), json!("updated-first"));
        tool_tab_new(&json!({"url": first}), &mut state).await.expect("second tab");
        assert_eq!(state.page_mut().evaluate("[localStorage.getItem('token'), sessionStorage.getItem('tab')]"),
            json!(["updated-first", null]), "restoring a session must not share sessionStorage across tabs");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn storage_export_retains_previously_visited_origins() {
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let (first, _) = spawn_form_recording_server();
        let (second, _) = spawn_form_recording_server();
        let mut state = BrowserState::new(None, None, false);
        for (origin, value) in [(&first, "first"), (&second, "second")] {
            state.page_mut().navigate(origin).await.expect("fixture page");
            state.page_mut().evaluate(&format!("(function(){{localStorage.setItem('token', '{value}'); sessionStorage.setItem('tab', '{value}-session'); return true;}})()"));
        }
        let exported: Value = serde_json::from_str(&tool_storage_state(&json!({}), &mut state).expect("export")).unwrap();
        let origins = exported["origins"].as_array().expect("origins");
        assert_eq!(origins.len(), 2, "export must not omit the previous origin: {exported}");
        for (origin, value) in [(&first, "first"), (&second, "second")] {
            let entry = origins.iter().find(|entry| entry["origin"] == json!(origin)).expect("visited origin");
            assert_eq!(entry["localStorage"], json!([{"name": "token", "value": value}]));
            assert_eq!(entry["sessionStorage"], json!([{"name": "tab", "value": format!("{value}-session")}]));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn storage_restore_rejects_invalid_input_before_mutation() {
        for invalid in [
            json!({"origin": "null", "localStorage": [["token", "bad"]]}),
            json!({"origin": "https://other.test/path", "localStorage": [["token", "bad"]]}),
            json!({"origin": "https://other.test", "localStorage": [["token", 12]]}),
        ] {
            let mut state = BrowserState::new(None, None, false);
            let result = tool_set_storage_state(&json!({"state": {"origins": [
                {"origin": "https://first.test", "localStorage": [["token", "partial"]]}, invalid
            ]}}), &mut state);
            assert!(result.is_err(), "invalid state must not report success: {result:?}");
            assert!(state.context.local_storage.entries().is_empty(), "invalid input must be validated before writes");
            assert!(state.tabs.is_empty(), "invalid input must not create a tab");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn storage_restore_reports_rejected_quota_writes() {
        let mut state = BrowserState::new(None, None, false);
        let result = tool_set_storage_state(&json!({"state": {"origins": [
            {"origin": "https://first.test", "localStorage": [["oversized", "x".repeat(5 * 1024 * 1024 + 1)]]}
        ]}}), &mut state);
        assert!(result.is_err(), "rejected native storage writes must not report success");
        assert!(state.context.local_storage.entries().is_empty(), "rejected value must not be retained");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn storage_state_round_trip_preserves_nul_prefixed_strings() {
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let (origin, _) = spawn_form_recording_server();
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(&origin).await.expect("fixture page");
        state.page_mut().evaluate(r#"(function(){localStorage.setItem('\0key', '\0"value"'); sessionStorage.setItem('\0tab', '\0session'); return true;})()"#);
        let exported: Value = serde_json::from_str(&tool_storage_state(&json!({}), &mut state).expect("export")).unwrap();
        assert_eq!(exported["origins"][0]["localStorage"], json!([{"name": "\0key", "value": "\0\"value\""}]));
        assert_eq!(exported["origins"][0]["sessionStorage"], json!([{"name": "\0tab", "value": "\0session"}]));
        drop(state);
        let mut restored = BrowserState::new(None, None, false);
        tool_set_storage_state(&json!({"state": exported}), &mut restored).expect("restore");
        restored.page_mut().navigate(&origin).await.expect("restored page");
        assert_eq!(restored.page_mut().evaluate(r#"[localStorage.getItem('\0key'), sessionStorage.getItem('\0tab')]"#),
            json!(["\0\"value\"", "\0session"]));
    }

    #[test]
    fn search_macros_expand_query_and_fall_back_to_url() {
        assert_eq!(
            expand_search_macro("@google_search", "hello world").as_deref(),
            Some("https://www.google.com/search?q=hello%20world")
        );
        assert_eq!(
            expand_search_macro("@youtube_search", "a+b").as_deref(),
            Some("https://www.youtube.com/results?search_query=a%2Bb")
        );
        assert_eq!(expand_search_macro("@unknown_macro", "x"), None);
        // Values that read as "unset" skip expansion entirely.
        let args = json!({"url": "data:text/html,<p>x</p>", "macro": "none", "query": "q"});
        assert_eq!(args["macro"].as_str(), Some("none"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn consent_dismissal_is_opt_in_and_only_targets_consent_managers() {
        let banner = "<div id=onetrust-banner-sdk><button id=onetrust-accept-btn-handler onclick=\"window.consent='accepted'\">Accept</button></div>";

        // Opt-in clicks the consent control.
        let mut state = BrowserState::new(None, None, false);
        tool_navigate(&json!({"url": format!("data:text/html,{banner}"), "dismissConsent": true}), &mut state)
            .await.expect("navigate");
        assert_eq!(state.page_mut().evaluate("window.consent"), json!("accepted"));

        // Same banner without the flag is left alone.
        let mut state = BrowserState::new(None, None, false);
        tool_navigate(&json!({"url": format!("data:text/html,{banner}")}), &mut state)
            .await.expect("navigate");
        assert_eq!(state.page_mut().evaluate("window.consent"), Value::Null);

        // A hidden banner is not clicked.
        let hidden = format!("<div style=\"display:none\">{banner}</div>");
        let mut state = BrowserState::new(None, None, false);
        tool_navigate(&json!({"url": format!("data:text/html,{hidden}"), "dismissConsent": true}), &mut state)
            .await.expect("navigate");
        assert_eq!(state.page_mut().evaluate("window.consent"), Value::Null);

        // An ordinary "Accept" button outside consent-manager markup is untouched.
        let plain = "<button onclick=\"window.consent='wrong'\">Accept cookies</button>";
        let mut state = BrowserState::new(None, None, false);
        tool_navigate(&json!({"url": format!("data:text/html,{plain}"), "dismissConsent": true}), &mut state)
            .await.expect("navigate");
        assert_eq!(state.page_mut().evaluate("window.consent"), Value::Null);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn storage_state_file_export_and_playwright_import() {
        let mut state = BrowserState::new(None, None, false);
        let dir = std::env::temp_dir().join(format!("obscura-storage-state-{}", std::process::id()));
        let path = dir.join("nested/state.json");
        let summary = tool_storage_state(&json!({"path": path.to_str().unwrap()}), &mut state)
            .expect("export to file");
        assert!(summary.contains("0 cookies"), "summary: {summary}");
        assert!(path.exists());
        // Owner-only file, no tmp sibling left behind.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert!(std::fs::read_dir(dir.join("nested")).unwrap().all(|e| e.unwrap().file_name() == "state.json"));

        // A Playwright-shaped file restores: httpOnly/sameSite cookie fields,
        // {name,value} storage entries, float expires.
        std::fs::write(&path, serde_json::to_string(&json!({
            "cookies": [{"name": "s", "value": "v", "domain": "httpbin.org", "path": "/",
                "expires": -1.0, "httpOnly": true, "secure": true, "sameSite": "Lax"}],
            "origins": [{"origin": "https://first.test", "localStorage": [{"name": "k", "value": "v"}]}],
        })).unwrap()).unwrap();
        let mut restored = BrowserState::new(None, None, false);
        tool_set_storage_state(&json!({"path": path.to_str().unwrap()}), &mut restored)
            .expect("file restore");
        assert_eq!(restored.context.local_storage.get("https://first.test", "k").as_deref(), Some("v"));
        let url = url::Url::parse("https://httpbin.org/").unwrap();
        assert!(restored.context.cookie_jar.get_cookie_header_same_site(&url).contains("s=v"));

        // A malformed file is rejected before any restore.
        std::fs::write(&path, "{}").unwrap();
        assert!(tool_set_storage_state(&json!({"path": path.to_str().unwrap()}), &mut restored).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn element_refs_reject_copied_attributes_before_clicking() {
        for mutation in [
            "button.replaceWith(copy)",
            "button.before(copy)",
        ] {
            let mut state = BrowserState::new(None, None, false);
            state.page_mut().navigate("data:text/html,<button id=target>Buy</button>")
                .await.expect("test page");
            tool_snapshot(&json!({}), &mut state).expect("snapshot");
            let reference = state.interactive_refs.keys().next().expect("button ref").clone();
            tool_evaluate(&json!({"expression": format!(
                "(function(){{ window.clicks=0; document.body.addEventListener('click',function(){{window.clicks++}}); var button=document.getElementById('target'); var copy=button.cloneNode(true); {mutation}; return true; }})()"
            )}), &mut state).await.expect("framework rerender");

            let result = tool_click(&json!({"ref": reference}), &mut state).await;
            let clicks = state.page_mut().evaluate("window.clicks");
            assert_eq!(clicks.as_f64(), Some(0.0), "stale ref must not click a clone: {mutation}, {result:?}");
            assert!(result.is_err(), "stale ref must request a fresh snapshot");

            tool_snapshot(&json!({}), &mut state).expect("fresh snapshot");
            let reference = state.interactive_refs.keys().next().expect("fresh ref").clone();
            tool_click(&json!({"ref": reference}), &mut state).await.expect("fresh ref works");
            assert_eq!(state.page_mut().evaluate("window.clicks").as_f64(), Some(1.0));
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn element_refs_from_old_snapshots_cannot_alias_new_targets() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<button id=target>Buy</button>")
            .await.expect("test page");
        tool_snapshot(&json!({}), &mut state).expect("snapshot");
        let reference = state.interactive_refs.keys().next().expect("button ref").clone();
        tool_evaluate(&json!({"expression":
            "(function(){window.clicks=0; document.body.addEventListener('click',function(){window.clicks++}); var other=document.createElement('button'); other.textContent='Delete'; document.body.prepend(other); return true;})()"
        }), &mut state).await.expect("insert another action");
        tool_snapshot(&json!({}), &mut state).expect("new snapshot");
        let result = tool_click(&json!({"ref": reference}), &mut state).await;
        assert_eq!(state.page_mut().evaluate("window.clicks").as_f64(), Some(0.0), "old ref must not act on a new target: {result:?}");
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn element_refs_follow_the_same_connected_node_after_reordering() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<button id=target>Buy</button><p>Details</p>")
            .await.expect("test page");
        tool_snapshot(&json!({}), &mut state).expect("snapshot");
        let reference = state.interactive_refs.keys().next().expect("button ref").clone();
        tool_evaluate(&json!({"expression":
            "(function(){window.clicked=''; var button=document.getElementById('target'); button.addEventListener('click',function(){window.clicked=this.id}); document.body.appendChild(button); return true;})()"
        }), &mut state).await.expect("reorder same node");
        tool_click(&json!({"ref": reference}), &mut state).await.expect("same node ref");
        assert_eq!(state.page_mut().evaluate("window.clicked"), json!("target"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn element_refs_survive_actions_that_stay_on_the_page() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(concat!(
            "data:text/html,<button id=more onclick=\"window.clicks=(window.clicks||0)+1;history.pushState({},'',String.fromCharCode(35)+'route')\">More</button>",
            "<input id=quantity><form onsubmit=\"event.preventDefault()\"><input id=email><button id=send>Send</button></form>",
            "<a id=next href=\"data:text/html,<button id=landed>Landed</button>\">Next</a>",
        )).await.expect("test page");
        tool_interactive_elements(&json!({}), &mut state).expect("listing");
        let refs = state.page_mut().evaluate(r#"(function(){
            return ['more','quantity','email','send','next'].map(function(id){
                return document.getElementById(id).getAttribute('data-obscura-ref');
            });
        })()"#);
        let more = refs[0].as_str().unwrap();
        let quantity = refs[1].as_str().unwrap();
        tool_click(&json!({"ref": more}), &mut state).await.expect("non-navigating click");
        tool_fill(&json!({"ref": quantity, "value": "3"}), &mut state).await.expect("ref survives click");
        tool_scroll(&json!({"direction": "down", "amount": 1}), &mut state).expect("scroll");
        tool_click(&json!({"ref": more}), &mut state).await.expect("ref survives scroll");
        tool_fill_form(&json!({"fields": [{"ref": refs[2], "value": "test@example.test"}], "submit_ref": refs[3]}), &mut state)
            .await.expect("prevented submit");
        tool_fill(&json!({"ref": quantity, "value": "4"}), &mut state).await.expect("ref survives submit");
        assert_eq!(state.page_mut().evaluate("[window.clicks,document.getElementById('quantity').value]"), json!([2, "4"]));
        state.page_mut().evaluate("history.replaceState({},'',String.fromCharCode(35)+'later')");
        assert!(!state.finish_active_page_turn(true).await.expect("same-document history turn"));
        state.ref_to_selector(quantity).expect("ref survives background same-document navigation");

        tool_click(&json!({"ref": refs[4]}), &mut state).await.expect("navigating click");
        assert_eq!(state.page_mut().evaluate("!!document.getElementById('landed')"), json!(true));
        assert!(state.ref_to_selector(more).unwrap_err().contains("unknown ref"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn element_refs_reject_reused_arena_slots() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<input id=target>").await.expect("test page");
        tool_snapshot(&json!({}), &mut state).expect("snapshot");
        let reference = state.interactive_refs.keys().next().unwrap().clone();
        let old_node = state.page_mut().with_dom(|dom| {
            let id = dom.query_selector("#target").unwrap().unwrap();
            dom.remove(id);
            id
        }).unwrap();
        // Native consumers may free an arena slot; its numeric id can then
        // name a different action, even with a copied ref attribute.
        state.page_mut().evaluate(&format!(r#"(function(){{
            var replacement=document.createElement('input'); replacement.id='replacement';
            replacement.setAttribute('data-obscura-ref',{}); document.body.appendChild(replacement);
            return true;
        }})()"#, serde_json::to_string(&reference).unwrap()));
        assert_eq!(state.page_mut().with_dom(|dom| dom.query_selector("#replacement").unwrap()), Some(Some(old_node)));
        let error = tool_fill(&json!({"ref": reference, "value": "wrong action"}), &mut state).await.unwrap_err();
        assert!(error.contains("stale ref"), "{error}");
        assert_eq!(state.page_mut().evaluate("document.getElementById('replacement').value"), json!(""));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn element_refs_retire_after_failed_document_navigation() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        for route in ["navigate", "click", "background"] {
            let mut state = BrowserState::new(None, None, false);
            state.page_mut().navigate("data:text/html,<input id=target><button id=go>Go</button>").await.unwrap();
            tool_snapshot(&json!({}), &mut state).unwrap();
            let reference = state.page_mut().evaluate("document.getElementById('target').getAttribute('data-obscura-ref')").as_str().unwrap().to_string();
            let old_node = state.page_mut().with_dom(|dom| dom.query_selector("#target").unwrap().unwrap()).unwrap();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/replacement", listener.local_addr().unwrap());
            let html = format!("<input id=target data-obscura-ref={reference}><script src=/hang></script>");
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0u8; 4096];
                socket.read(&mut request).await.unwrap();
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}", html.len()).as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
                let (hanging_script, _) = listener.accept().await.unwrap();
                std::future::pending::<()>().await;
                drop(hanging_script);
            });
            state.page_mut().set_navigation_timeout(std::time::Duration::from_millis(300));
            let result = match route {
                "navigate" => tool_navigate(&json!({"url": url}), &mut state).await.map(|_| ()),
                "click" => {
                    state.page_mut().evaluate(&format!("document.getElementById('go').onclick=function(){{location.href={};}}", serde_json::to_string(&url).unwrap()));
                    tool_click(&json!({"selector":"#go"}), &mut state).await.map(|_| ())
                }
                _ => {
                    state.page_mut().evaluate(&format!("location.href={}", serde_json::to_string(&url).unwrap()));
                    state.finish_active_page_turn(true).await.map(|_| ())
                }
            };
            server.abort();
            let _ = server.await;
            assert!(result.unwrap_err().contains("deadline"), "{route}");
            assert_eq!(state.page_mut().evaluate("[!!document.querySelector('script[src=\"/hang\"]'),!!document.getElementById('go')]"), json!([true, false]), "replacement committed before timeout: {route}");
            assert_eq!(state.page_mut().with_dom(|dom| dom.query_selector("#target").unwrap()), Some(Some(old_node)), "replacement reuses the numeric id: {route}");
            assert!(state.ref_to_selector(&reference).unwrap_err().contains("unknown ref"), "{route}");
            assert_eq!(state.page_mut().evaluate("document.getElementById('target').value"), json!(""));
            tool_snapshot(&json!({}), &mut state).unwrap();
            let fresh = state.interactive_refs.keys().next().unwrap();
            assert_ne!(fresh, &reference, "retired tokens must never name the replacement");
            assert!(state.ref_to_selector(&reference).is_err());
        }

        // An error before a replacement commits also retires refs: callers
        // must refresh rather than guessing which phase of a load failed.
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<input id=retained>").await.unwrap();
        tool_snapshot(&json!({}), &mut state).unwrap();
        let reference = state.interactive_refs.keys().next().unwrap().clone();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/unreachable", listener.local_addr().unwrap());
        drop(listener);
        state.page_mut().evaluate(&format!("location.href={}", serde_json::to_string(&url).unwrap()));
        assert!(state.finish_active_page_turn(true).await.is_err());
        assert_eq!(state.page_mut().evaluate("!!document.getElementById('retained')"), json!(true));
        assert!(state.ref_to_selector(&reference).unwrap_err().contains("unknown ref"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn first_evaluation_and_promise_results_work_without_prior_state() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<p>Evaluate</p>").await.unwrap();
        assert_eq!(tool_evaluate(&json!({"expression":"1 + 1"}), &mut state).await.unwrap().parse::<f64>().unwrap(), 2.0);
        assert_eq!(tool_evaluate(&json!({"expression":"Promise.resolve(3)"}), &mut state).await.unwrap().parse::<f64>().unwrap(), 3.0);
    }

    // #1015: case-folding can change byte length (Turkish "İ" -> "i̇"), so a
    // match offset found in the lowercased haystack must be translated back to
    // the original body's byte offset.
    #[test]
    fn search_body_maps_offsets_through_unicode_lowercasing() {
        let body = "İstanbul hava durumu";
        let results = search_body(body, "HAVA", false, 5, 10);
        assert_eq!(results.len(), 1, "should find one match");
        let offset = results[0]["offset"].as_u64().unwrap() as usize;
        assert_eq!(
            offset,
            body.find("hava").unwrap(),
            "offset must be the byte position of the match in the original body"
        );
        assert!(
            results[0]["snippet"]
                .as_str()
                .unwrap()
                .to_lowercase()
                .contains("hava"),
            "snippet must contain the match, got {}",
            results[0]["snippet"]
        );
    }

    fn listed_tools() -> Vec<Value> {
        handle_tools_list(json!(1)).result.expect("tools/list result")
            .get("tools").and_then(Value::as_array).cloned().expect("tools array")
    }

    #[test]
    fn tool_schemas_expose_snapshot_limit_without_nested_properties() {
        let tools = listed_tools();
        let snapshot = tools.iter().find(|tool| tool["name"] == "browser_snapshot")
            .expect("browser_snapshot tool");
        assert_eq!(snapshot["inputSchema"]["properties"]["max_chars"]["type"], "number");
        for tool in tools {
            assert!(
                tool["inputSchema"]["properties"].get("properties").is_none(),
                "{} has a nested duplicate properties object", tool["name"]
            );
        }
    }

    #[cfg(not(feature = "render"))]
    #[test]
    fn render_tools_are_not_advertised_without_render_feature() {
        let tools = listed_tools();
        assert!(tools.iter().all(|tool| {
            tool["name"] != "browser_screenshot" && tool["name"] != "browser_pdf"
        }));
    }

    async fn web_search_response_fixture(
        response: String,
        hold_open: bool,
    ) -> (BrowserState, tokio::task::JoinHandle<()>) {
        use tokio::io::AsyncReadExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let fixture = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = BufReader::new(socket);
            let mut line = String::new();
            socket.read_line(&mut line).await.unwrap();
            assert_eq!(line, "POST http://search.test/v1/search/public HTTP/1.1\r\n");
            let mut length = None;
            loop {
                line.clear();
                assert!(socket.read_line(&mut line).await.unwrap() > 0);
                if line == "\r\n" { break; }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
            }
            let mut request = vec![0; length.expect("request Content-Length")];
            socket.read_exact(&mut request).await.unwrap();
            let sent = socket.get_mut().write_all(response.as_bytes()).await;
            if hold_open {
                if sent.is_ok() {
                    // Leave the response incomplete until the client drops it.
                    let _ = socket.read_u8().await;
                }
            } else {
                sent.unwrap();
            }
        });
        let state = BrowserState::new(Some(format!("http://{address}")), None, false);
        (state, fixture)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn web_search_response_limit_accepts_normal_and_fifty_result_bodies() {
        let small = json!({"results": [{
            "title": "Small fixture", "url": "https://example.test/", "snippet": "small result"
        }]}).to_string();
        let results: Vec<Value> = (0..50).map(|index| json!({
            "title": format!("Result {index}"),
            "url": format!("https://example.test/{index}"),
            "snippet": "word ".repeat(1600),
        })).collect();
        let mut boundary = json!({"results": results}).to_string();
        assert!(boundary.len() < WEB_SEARCH_RESPONSE_BYTES);
        boundary.extend(std::iter::repeat(' ').take(WEB_SEARCH_RESPONSE_BYTES - boundary.len()));

        for (body, count) in [(small, 1), (boundary, 50)] {
            for chunked in [false, true] {
                let response = if chunked {
                    format!(
                        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:X}\r\n{body}\r\n0\r\n\r\n",
                        body.len()
                    )
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                };
                let (state, fixture) = web_search_response_fixture(response, false).await;
                let request = WebSearchRequest {
                    url: "http://search.test/v1/search/public",
                    ..web_search_request("obscura", 50, None)
                };
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    fetch_web_search(request, &state, true),
                ).await.expect("search response timed out").unwrap();
                assert_eq!(result["results"].as_array().unwrap().len(), count);
                assert_eq!(format_web_results("obscura", &result).lines().count(), count);
                if count == 1 {
                    assert_eq!(result["results"][0]["snippet"], "small result");
                } else {
                    assert_eq!(result["results"][49]["title"], "Result 49");
                    assert_eq!(result["results"][49]["snippet"].as_str().unwrap().len(), 8000);
                }
                fixture.await.unwrap();
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn web_search_response_limit_rejects_overflow_before_eof() {
        let limit = WEB_SEARCH_RESPONSE_BYTES;
        let mut body = json!({"results": []}).to_string();
        body.extend(std::iter::repeat(' ').take(limit + 1 - body.len()));
        assert_eq!(body.len(), limit + 1);
        let responses = [
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", limit + 1),
            format!("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:X}\r\n{body}\r\n", body.len()),
            format!("HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\n\r\n{body}"),
        ];
        for response in responses {
            let (state, fixture) = web_search_response_fixture(response, true).await;
            let request = WebSearchRequest {
                url: "http://search.test/v1/search/public",
                ..web_search_request("obscura", 50, None)
            };
            let error = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                fetch_web_search(request, &state, true),
            ).await.expect("oversized response must fail before EOF").unwrap_err();
            assert!(error.contains("web search response failed:"), "{error}");
            assert!(error.contains(&format!("exceeded {limit} byte limit")), "{error}");
            tokio::time::timeout(std::time::Duration::from_secs(5), fixture)
                .await.expect("response connection was not closed").unwrap();
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn web_search_is_hidden_and_rejected_without_opt_in() {
        std::env::remove_var("OBSCURA_WEB_SEARCH");
        assert!(listed_tools().iter().all(|tool| tool["name"] != "web_search"));
        let mut state = BrowserState::new(None, None, false);
        let response = handle_tool_call(
            json!(1),
            &json!({"name": "web_search", "arguments": {"query": "obscura"}}),
            &mut state,
        ).await.result.expect("tool response");
        assert_eq!(response["isError"], true);
        assert!(response.to_string().contains("Unknown tool: web_search"), "{response}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn web_search_is_advertised_when_enabled_and_requires_a_query() {
        std::env::set_var("OBSCURA_WEB_SEARCH", "keenable");
        let tools = listed_tools();
        let tool = tools.iter().find(|tool| tool["name"] == "web_search").expect("web_search tool");
        assert_eq!(tool["inputSchema"]["required"], json!(["query"]));
        let mut state = BrowserState::new(None, None, false);
        let error = tool_web_search(&json!({"query": "  "}), &mut state).await.unwrap_err();
        assert!(error.contains("Missing query"), "{error}");
    }

    #[test]
    fn web_search_request_honors_integer_valued_json_limits() {
        let omitted: Value = serde_json::from_str("{}").unwrap();
        assert_eq!(web_search_limit(&omitted), 5);
        for (limit, expected) in [
            ("1", 1), ("1.0", 1), ("1e0", 1),
            ("50", 50), ("50.0", 50), ("5e1", 50),
            ("0", 1), ("0.0", 1), ("-0.0", 1),
            ("51", 50), ("51.0", 50),
            ("18446744073709551615", 50), ("1e100", 50),
            ("-1", 5), ("-1.0", 5), ("1.5", 5),
            ("null", 5), (r#""1""#, 5), ("true", 5),
        ] {
            let args: Value = serde_json::from_str(&format!(r#"{{"limit":{limit}}}"#)).unwrap();
            let request = web_search_request("obscura", web_search_limit(&args), None);
            let body: Value = serde_json::from_str(&request.body).unwrap();
            assert_eq!(body["max_results"], json!(expected), "limit={limit}");
        }
    }

    #[test]
    fn web_search_request_is_keyless_unless_a_key_is_set() {
        let public = web_search_request("rust headless browser", 5, None);
        assert_eq!(public.url, "https://api.keenable.ai/v1/search/public");
        assert!(public.headers.contains(&("x-keenable-title", "Obscura".to_string())));
        assert!(public.headers.iter().all(|(name, _)| *name != "x-api-key"));
        assert_eq!(
            serde_json::from_str::<Value>(&public.body).unwrap(),
            json!({"query": "rust headless browser", "max_results": 5})
        );

        let keyed = web_search_request("rust", 3, Some("test-key"));
        assert_eq!(keyed.url, "https://api.keenable.ai/v1/search");
        assert!(keyed.headers.contains(&("x-api-key", "test-key".to_string())));
    }

    #[test]
    fn web_search_results_use_snippet_and_stay_compact() {
        let long = "word ".repeat(400);
        let body = json!({"results": [
            {"title": "  Obscura\n docs ", "url": "https://docs.obscura.sh/", "description": "", "snippet": format!("Line one\n\n{long}")},
            {"title": "no url", "snippet": "dropped"}
        ]});
        let out = format_web_results("obscura", &body);
        let lines: Vec<Value> = out.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(lines.len(), 1, "{out}");
        assert_eq!(lines[0]["title"], "Obscura docs");
        assert_eq!(lines[0]["url"], "https://docs.obscura.sh/");
        let snippet = lines[0]["snippet"].as_str().unwrap();
        assert!(snippet.starts_with("Line one word word"), "{snippet}");
        assert!(snippet.ends_with("..."));
        assert_eq!(snippet.chars().count(), WEB_SEARCH_SNIPPET_CHARS + 3);
        assert_eq!(format_web_results("nothing", &json!({"results": []})), "No results for \"nothing\".");
    }

    #[tokio::test]
    async fn local_network_hint_is_limited_to_operator_stdio_state() {
        std::env::remove_var("OBSCURA_ALLOW_PRIVATE_NETWORK");
        let mut state = BrowserState::new(None, None, false);
        let args = json!({"url": "http://localhost:9/"});
        let remote_error = tool_navigate(&args, &mut state).await.unwrap_err();
        assert!(remote_error.contains("is not allowed"));
        assert!(!remote_error.contains("--allow-private-network"));
        state.operator_network_hints = true;
        let local_error = tool_navigate(&args, &mut state).await.unwrap_err();
        assert!(local_error.contains("is not allowed"));
        assert!(local_error.contains("--allow-private-network"));
        assert!(local_error.contains("OBSCURA_ALLOW_PRIVATE_NETWORK=1"));
        let metadata_error = tool_navigate(&json!({"url":"http://169.254.169.254/latest/meta-data/"}), &mut state).await.unwrap_err();
        assert!(metadata_error.contains("is not allowed"));
        assert!(!metadata_error.contains("--allow-private-network"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn mcp_navigation_rejects_local_files() {
        let mut state = BrowserState::new(None, None, false);
        let error = tool_navigate(&json!({"url": "file:///etc/passwd"}), &mut state)
            .await
            .expect_err("MCP must not expose local files");
        assert!(error.contains("file:// navigation is disabled"));
    }

    // Every MCP route that navigates must refuse local files, not only
    // browser_navigate: the browser layer enforces it for all of them.
    #[tokio::test(flavor = "current_thread")]
    async fn extract_xref_resolves_snapshot_refs_with_type_coercion() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(
            "data:text/html,<a href='/buy' id='buy'>Buy now $19.50</a><button id='n'>99</button>",
        ).await.unwrap();
        tool_snapshot(&json!({}), &mut state).unwrap();
        let entries: Vec<(String, NodeId)> = state.interactive_refs.iter()
            .map(|(r, e)| (r.clone(), e.node_id)).collect();
        let mut ref_for = |want: &str| entries.iter().find_map(|(r, nid)| {
            state.page_mut().with_dom(|dom| {
                dom.get_node(*nid)?.get_attribute("id").filter(|id| *id == want)?;
                Some(r.clone())
            }).flatten()
        });
        let buy_ref = ref_for("buy").expect("link ref");
        let n_ref = ref_for("n").expect("button ref");
        let schema = json!({"type":"object","properties":{
            "price":{"x-ref":buy_ref,"type":"number"},
            "count":{"x-ref":n_ref,"type":"integer"},
            "missing":{"x-ref":"e99-1"}
        }});
        let out: Value = serde_json::from_str(
            &tool_extract(&json!({"schema":schema}), &mut state).unwrap()).unwrap();
        assert_eq!(out["price"], json!(19.50));
        assert_eq!(out["count"], json!(99));
        assert_eq!(out["missing"], json!(null));
        // A required property that cannot be extracted aborts the call.
        let required = json!({"type":"object","properties":{
            "missing":{"x-ref":"e99-1"}},"required":["missing"]});
        let err = tool_extract(&json!({"schema":required}), &mut state).unwrap_err();
        assert!(err.contains("required property \"missing\""), "{err}");
        // Non-object top-level type is rejected.
        let bad = json!({"type":"array","properties":{}});
        assert!(tool_extract(&json!({"schema":bad}), &mut state).unwrap_err().contains("type: object"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tab_new_nav_failure_leaves_no_orphan_tab() {
        let mut state = BrowserState::new(None, None, false);
        tool_tab_new(&json!({"url": "data:text/html,<p>kept</p>"}), &mut state).await.unwrap();
        let kept = state.active_tab.clone().unwrap();
        let tab_count = state.tabs.len();
        // Unreachable host fails the navigation.
        let result = tool_tab_new(&json!({"url": "http://127.0.0.1:9/unreachable"}), &mut state).await;
        assert!(result.is_err(), "nav failure must surface as a tool error");
        assert_eq!(state.tabs.len(), tab_count, "failed tab must be dropped: {:?}", state.tabs.keys());
        assert_eq!(state.active_tab.as_deref(), Some(kept.as_str()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn mcp_tab_new_rejects_local_files() {
        let mut state = BrowserState::new(None, None, false);
        let error = tool_tab_new(&json!({"url": "file:///etc/passwd"}), &mut state)
            .await
            .expect_err("browser_tab_new must not expose local files");
        assert!(error.contains("file://"), "{error}");
        assert!(!state.page_mut().url_string().starts_with("file:"));
    }

    // An untrusted page must not be able to pull a local file into the agent's
    // context by planting a file:// link for the agent to click.
    #[tokio::test(flavor = "current_thread")]
    async fn mcp_click_cannot_follow_a_link_into_file_scheme() {
        let path = std::env::temp_dir().join(format!("obscura-mcp-file-link-{}.html", std::process::id()));
        std::fs::write(&path, "<p>local-secret</p>").expect("write fixture");
        let file_url = url::Url::from_file_path(&path).expect("file url").to_string();

        let mut state = BrowserState::new(None, None, false);
        state
            .page_mut()
            .navigate(&format!("data:text/html,<a id=l href='{file_url}'>go</a>"))
            .await
            .expect("web page");
        let web_url = state.page_mut().url_string();

        let click = handle_tool_call(
            json!(1),
            &json!({"name": "browser_click", "arguments": {"selector": "#l"}}),
            &mut state,
        )
        .await
        .result
        .expect("click response");
        assert_ne!(click["isError"], true, "{click}");
        assert_eq!(state.page_mut().url_string(), web_url);

        let snapshot = handle_tool_call(
            json!(2),
            &json!({"name": "browser_snapshot", "arguments": {}}),
            &mut state,
        )
        .await
        .result
        .expect("snapshot response");
        assert!(!snapshot.to_string().contains("local-secret"), "{snapshot}");

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "render")]
    #[test]
    fn render_tools_are_advertised_with_flat_schemas() {
        let tools = listed_tools();
        for name in ["browser_screenshot", "browser_pdf"] {
            let tool = tools.iter().find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(tool["inputSchema"]["type"], "object");
            assert_eq!(tool["inputSchema"]["additionalProperties"], false);
            assert!(tool["inputSchema"]["properties"].is_object());
            assert!(tool["inputSchema"]["properties"].get("properties").is_none());
        }
    }

    #[cfg(feature = "render")]
    #[tokio::test(flavor = "current_thread")]
    async fn render_tool_calls_return_mcp_binary_content_and_reject_bad_options() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(
            "data:text/html,<html style='margin:0'><body style='margin:0;background:red'><div style='width:64px;height:48px'></div></body></html>",
        ).await.expect("render test page should navigate");
        state.page_mut().set_viewport((64.0, 48.0));

        let screenshot = handle_tool_call(
            json!(1), &json!({ "name": "browser_screenshot", "arguments": {} }), &mut state,
        ).await.result.expect("screenshot response");
        let image = &screenshot["content"][0];
        assert_eq!(image["type"], "image");
        assert_eq!(image["mimeType"], "image/png");
        let png = BASE64.decode(image["data"].as_str().expect("PNG base64"))
            .expect("valid PNG base64");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));

        let pdf = handle_tool_call(
            json!(2),
            &json!({ "name": "browser_pdf", "arguments": { "print_background": true } }),
            &mut state,
        ).await.result.expect("PDF response");
        let resource = &pdf["content"][0];
        assert_eq!(resource["type"], "resource");
        assert_eq!(resource["resource"]["mimeType"], "application/pdf");
        let bytes = BASE64.decode(resource["resource"]["blob"].as_str().expect("PDF base64"))
            .expect("valid PDF base64");
        assert!(bytes.starts_with(b"%PDF-"));

        let invalid_screenshot = handle_tool_call(
            json!(3),
            &json!({ "name": "browser_screenshot", "arguments": { "width": 0 } }),
            &mut state,
        ).await.result.expect("invalid screenshot response");
        assert_eq!(invalid_screenshot["isError"], true);

        let invalid_pdf = handle_tool_call(
            json!(4),
            &json!({ "name": "browser_pdf", "arguments": { "scale": 3 } }),
            &mut state,
        ).await.result.expect("invalid PDF response");
        assert_eq!(invalid_pdf["isError"], true);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn console_tool_returns_page_messages() {
        const PAGE: &str = "data:text/html,<button id=log>Log</button><script>\
            console.error('mcp-console-inline');\
            document.getElementById('log').onclick=()=>{\
                console.error('mcp-console-click');\
                setTimeout(()=>{console.error('mcp-console-async');document.body.id='done'},25)\
            }</script>";
        let mut state = BrowserState::new(None, None, false);
        state
            .page_mut()
            .navigate(PAGE)
            .await
            .expect("console test page should navigate");

        let inline = tool_console_messages(&mut state).expect("console tool should succeed");
        assert!(
            inline.contains("mcp-console-inline"),
            "inline console message missing from MCP output: {inline}"
        );

        tool_click(&json!({ "selector": "#log" }), &mut state)
            .await
            .expect("console test button should be clickable");
        tool_wait_for(&json!({ "selector": "#done", "timeout": 2 }), &mut state)
            .await
            .expect("asynchronous console callback should complete");
        let later = tool_console_messages(&mut state).expect("console tool should succeed");
        assert!(later.contains("mcp-console-click"), "{later}");
        assert!(later.contains("mcp-console-async"), "{later}");
    }

    /// Records the method, path and body of every request it serves, so a test
    /// can assert what actually left the process rather than what the page
    /// believes happened. Serves the issue's form at `/` and 200s everything
    /// else.
    fn spawn_form_recording_server() -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0u8; 8192];
                let read = stream.read(&mut buf).unwrap_or(0);
                let raw = String::from_utf8_lossy(&buf[..read]).to_string();
                let start = raw.lines().next().unwrap_or("").to_string();
                let mut parts = start.split_whitespace();
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("").to_string();
                let body = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                let _ = tx.send(format!("{method} {path} body='{body}'"));
                let page = "<!doctype html><meta charset=utf-8>\
                    <form id=f method=POST action=/submitted>\
                    <input name=q id=q value=><button type=submit id=go>Envoyer</button></form>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    page.len(),
                    page
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (base, rx)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn click_on_a_submit_button_issues_the_request_before_replying() {
        // A submit click runs the page's form glue and leaves the navigation as
        // pending state; it only becomes a request when a driving layer converts
        // it. The CDP path converts it, MCP did not, so the same click POSTed
        // over CDP and did nothing over MCP (#618). Worse than doing nothing:
        // `location.href` already reported the destination, so an agent was told
        // the submit had happened while no request had left the process.
        //
        // The assertion is deliberately at the wire, not on `location.href`,
        // because the URL was the thing that lied.
        // nextest gives each test its own process, so this cannot reach a
        // sibling. The recording server is on 127.0.0.1, which the SSRF gate
        // refuses by default, exactly as the issue reporter had to pass
        // --allow-private-network to run their repro.
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let (base, requests) = spawn_form_recording_server();
        let mut state = BrowserState::new(None, None, false);
        state
            .page_mut()
            .navigate(&base)
            .await
            .expect("form page should navigate");
        assert!(
            requests
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("the form page itself must be fetched")
                .starts_with("GET /"),
        );

        tool_fill(&json!({ "selector": "#q", "value": "hello" }), &mut state)
            .await
            .expect("browser_fill should succeed");
        tool_click(&json!({ "selector": "#go" }), &mut state)
            .await
            .expect("browser_click should succeed");

        let submitted = requests
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("the submit must reach the server before browser_click replies");
        assert!(
            submitted.starts_with("POST /submitted"),
            "expected a POST to the form action, got {submitted}"
        );
        assert!(
            submitted.contains("q=hello"),
            "the submitted body must carry the filled field, got {submitted}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn function_wait_advances_fetch_response_callbacks() {
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let (base, requests) = spawn_form_recording_server();
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate(&base).await.expect("fixture page");
        requests.recv_timeout(std::time::Duration::from_secs(2)).expect("page request");
        evaluate_tool_script(state.page_mut(),
            "fetch('/wait-data').then(r => r.text()).then(text => window.fetchedText = text); true"
        ).unwrap();
        assert_eq!(evaluate_tool_script(state.page_mut(), "typeof window.fetchedText").unwrap(), json!("undefined"));

        tool_wait_for_function(&json!({
            "function": "typeof window.fetchedText === 'string' && window.fetchedText.includes('Envoyer')",
            "timeout": 1000, "pollIntervalMs": 20
        }), &mut state).await.expect("fetch body callback must run while waiting");
        assert!(requests.recv_timeout(std::time::Duration::from_secs(2))
            .expect("fetch request").starts_with("GET /wait-data"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn function_wait_finishes_task_navigation_between_polls() {
        use tokio::io::AsyncReadExt;

        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/landed", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let read = stream.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..read]).starts_with("GET /landed "));
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 21\r\nConnection: close\r\n\r\n<p id=landed>done</p>").await.unwrap();
        });
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<p>waiting</p>").await.unwrap();
        evaluate_tool_script(state.page_mut(), &format!(
            "setTimeout(() => location.href = {}, 0); true", serde_json::to_string(&url).unwrap()
        )).unwrap();

        tool_wait_for_function(&json!({
            "function": "document.querySelector('#landed') !== null",
            "timeout": 1000, "pollIntervalMs": 10
        }), &mut state).await.expect("a navigation longer than one poll must finish");
        assert_eq!(state.page_mut().url_string(), url);
        server.await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn function_wait_preserves_idle_polling_and_caps_the_last_interval() {
        let mut state = BrowserState::new(None, None, false);
        state.page_mut().navigate("data:text/html,<p>idle</p>").await.unwrap();
        evaluate_tool_script(state.page_mut(), "window.waitPolls = 0; true").unwrap();
        let started = std::time::Instant::now();
        let error = tool_wait_for_function(&json!({
            "function": "(++window.waitPolls, false)", "timeout": 80, "pollIntervalMs": 1000
        }), &mut state).await.expect_err("false predicate must time out");
        assert!(error.contains("condition not met"), "{error}");
        assert!(started.elapsed() < std::time::Duration::from_millis(800),
            "the last interval must be bounded by the timeout");
        assert_eq!(evaluate_tool_script(state.page_mut(), "window.waitPolls").unwrap().as_f64(), Some(2.0),
            "an idle page must not busy-poll the predicate");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn network_tool_includes_completed_script_fetches() {
        std::env::set_var("OBSCURA_ALLOW_PRIVATE_NETWORK", "1");
        let (base, requests) = spawn_form_recording_server();
        let mut state = BrowserState::new(None, None, false);
        state
            .page_mut()
            .navigate(&base)
            .await
            .expect("test page should navigate");
        requests
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("the test page itself must be fetched");

        let fetch_url = serde_json::to_string(&format!("{base}/script-request"))
            .expect("fetch URL should serialize");
        state.page_mut().evaluate(&format!(
            "fetch({fetch_url}).then(() => document.body.id = 'fetch-complete')"
        ));
        tool_wait_for(
            &json!({ "selector": "#fetch-complete", "timeout": 2 }),
            &mut state,
        )
        .await
        .expect("script fetch should complete");
        let request = requests
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("the script fetch must reach the server");
        assert!(request.starts_with("GET /script-request"), "got {request}");

        let output = tool_network_requests(&mut state).expect("network tool should succeed");
        assert!(
            output.contains("/script-request"),
            "completed script fetch missing from network history: {output}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_tools_notify_controlled_input_tracker() {
        let mut state = BrowserState::new(None, None, false);
        state
            .page_mut()
            .navigate("data:text/html,<div id=root><input id=field></div>")
            .await
            .expect("test page should navigate");

        state.page_mut().evaluate(
            r#"(function () {
                var root = document.getElementById('root');
                var input = document.getElementById('field');
                var descriptor = Object.getOwnPropertyDescriptor(input.constructor.prototype, 'value');
                var tracked = String(input.value);
                Object.defineProperty(input, 'value', {
                    configurable: true,
                    get: function () { return descriptor.get.call(this); },
                    set: function (value) {
                        tracked = String(value);
                        descriptor.set.call(this, value);
                    }
                });
                window.__controlledState = '';
                window.__controlledUpdates = 0;
                window.__lastInputTarget = '';
                window.__lastInputTrusted = false;
                root.addEventListener('input', function (event) {
                    window.__lastInputTarget = event.target.id;
                    window.__lastInputTrusted = event.isTrusted;
                    var next = String(event.target.value);
                    if (next !== tracked) {
                        tracked = next;
                        window.__controlledState = next;
                        window.__controlledUpdates++;
                    }
                });
            })()"#,
        );

        tool_fill(
            &json!({ "selector": "#field", "value": "filled" }),
            &mut state,
        )
        .await
        .expect("browser_fill should succeed");
        tool_type(
            &json!({ "selector": "#field", "text": "-typed" }),
            &mut state,
        )
        .await
        .expect("browser_type should succeed");
        tool_fill_form(
            &json!({
                "fields": [{ "selector": "#field", "value": "form-filled" }]
            }),
            &mut state,
        )
        .await
        .expect("browser_fill_form should succeed");

        let actual = state.page_mut().evaluate(
            r#"JSON.stringify({
                domValue: document.getElementById('field').value,
                controlledState: window.__controlledState,
                controlledUpdates: window.__controlledUpdates,
                lastInputTarget: window.__lastInputTarget,
                lastInputTrusted: window.__lastInputTrusted
            })"#,
        );
        assert_eq!(
            actual,
            json!(r#"{"domValue":"form-filled","controlledState":"form-filled","controlledUpdates":3,"lastInputTarget":"field","lastInputTrusted":true}"#),
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_form_check_and_select_use_native_setter_and_trusted_events() {
        let mut state = BrowserState::new(None, None, false);
        state
            .page_mut()
            .navigate(
                "data:text/html,<div id=root><input id=box type=checkbox>\
                 <select id=sel><option value=a>a</option><option value=b>b</option></select></div>",
            )
            .await
            .expect("test page should navigate");

        // Install a React-style tracker on the checkbox's `checked`, redefined on
        // the instance. A direct `el.checked = true` runs this wrapper in lockstep,
        // so a change handler comparing target.checked to the tracked value sees no
        // change and never commits. Writing through the prototype setter (what
        // __obscura_setFieldValue does) leaves the tracker stale so the edit
        // registers. Also record whether the dispatched change is trusted.
        state.page_mut().evaluate(
            r#"(function () {
                var box = document.getElementById('box');
                var root = document.getElementById('root');
                var d = Object.getOwnPropertyDescriptor(box.constructor.prototype, 'checked');
                var tracked = box.checked;
                Object.defineProperty(box, 'checked', {
                    configurable: true,
                    get: function () { return d.get.call(this); },
                    set: function (v) { tracked = !!v; d.set.call(this, v); }
                });
                window.__checkedCommitted = false;
                window.__checkTrusted = false;
                window.__selectTrusted = false;
                root.addEventListener('change', function (event) {
                    if (event.target.id === 'box') {
                        window.__checkTrusted = event.isTrusted;
                        if (event.target.checked !== tracked) {
                            tracked = event.target.checked;
                            window.__checkedCommitted = true;
                        }
                    } else if (event.target.id === 'sel') {
                        window.__selectTrusted = event.isTrusted;
                    }
                });
            })()"#,
        );

        tool_fill_form(
            &json!({
                "fields": [
                    { "selector": "#box", "type": "check" },
                    { "selector": "#sel", "type": "select", "value": "b" }
                ]
            }),
            &mut state,
        )
        .await
        .expect("browser_fill_form should succeed");

        let actual = state.page_mut().evaluate(
            r#"JSON.stringify({
                domChecked: document.getElementById('box').checked,
                checkedCommitted: window.__checkedCommitted,
                checkTrusted: window.__checkTrusted,
                selValue: document.getElementById('sel').value,
                selectTrusted: window.__selectTrusted
            })"#,
        );
        assert_eq!(
            actual,
            json!(r#"{"domChecked":true,"checkedCommitted":true,"checkTrusted":true,"selValue":"b","selectTrusted":true}"#),
        );
    }
}
