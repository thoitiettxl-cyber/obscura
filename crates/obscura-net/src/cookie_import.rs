//! Import cookies from real browser profiles (Chrome/Chromium, Firefox).
//!
//! Lets `obscura` reuse an authenticated session that already exists in a
//! desktop browser instead of re-implementing each site's login flow. Chrome
//! cookie values are AES-encrypted; Firefox's `moz_cookies` are plaintext.
//!
//! Sources are selected with environment variables so every context that is
//! created later in the process (fetch pages, scrape workers, CDP sessions)
//! picks the import up the same way:
//!   - `OBSCURA_CHROME_PROFILE`   Chrome/Chromium profile directory
//!   - `OBSCURA_FIREFOX_PROFILE`  Firefox profile directory
//!   - `OBSCURA_CHROME_KEY`       `auto` (default), `peanuts`,
//!                                `base64:<raw-key>`, or `system` (macOS)
//!
//! The corresponding CLI flags (`--chrome-profile`, `--firefox-profile`,
//! `--chrome-key`) just set these variables at startup.

use std::path::{Path, PathBuf};

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use rusqlite::{Connection, OpenFlags};
use sha2::Sha256;
use sha2::Digest as _;

use crate::cookies::{CookieInfo, CookieJar};

/// Microseconds between the Windows/Chrome epoch (1601-01-01) and Unix epoch.
const CHROME_EPOCH_OFFSET_MICROS: i64 = 11_644_473_600_000_000;

/// Where Chrome's cookie decryption key comes from.
#[derive(Debug, Clone)]
pub enum ChromeKey {
    /// Read `os_crypt.encrypted_key` from the profile's Local State file,
    /// unwrap it with the legacy "peanuts" key, and fall back to the peanuts
    /// key itself for old-style AES-CBC cookie values. On macOS this resolves
    /// to the Keychain instead.
    Auto,
    /// Raw key, base64-encoded (16 bytes -> AES-128-CBC, 32 -> AES-256-GCM).
    Base64(Vec<u8>),
    /// PBKDF2-HMAC-SHA1("peanuts", "saltysalt", 1 iter): the hardcoded key old
    /// Linux Chrome used for cookie encryption.
    Peanuts,
    /// macOS Keychain "Chrome Safe Storage" (other platforms: error).
    System,
}

impl ChromeKey {
    pub fn parse(value: &str) -> Result<Self, String> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if value.eq_ignore_ascii_case("peanuts") {
            return Ok(Self::Peanuts);
        }
        if value.eq_ignore_ascii_case("system") {
            return Ok(Self::System);
        }
        if let Some(raw) = value.strip_prefix("base64:") {
            let key = B64.decode(raw).map_err(|e| format!("invalid base64 key: {e}"))?;
            if key.len() != 16 && key.len() != 32 {
                return Err("base64 key must decode to 16 or 32 bytes".to_string());
            }
            return Ok(Self::Base64(key));
        }
        Err("key must be `auto`, `peanuts`, `system`, or `base64:<raw-key>`".to_string())
    }
}

#[derive(Debug, Default)]
pub struct ImportReport {
    pub chrome: usize,
    pub firefox: usize,
    /// Human-readable per-source failure descriptions (empty when a source
    /// was not requested or succeeded).
    pub failures: Vec<String>,
}

/// Import cookies from the profiles named by the OBSCURA_*_PROFILE env vars
/// into `jar`. Each source is independent: one failing profile does not stop
/// the other, and a missing env var simply skips that browser.
pub fn import_from_env(jar: &CookieJar) -> ImportReport {
    let mut report = ImportReport::default();
    if let Ok(dir) = std::env::var("OBSCURA_CHROME_PROFILE") {
        let dir = dir.trim();
        if !dir.is_empty() {
            let key = std::env::var("OBSCURA_CHROME_KEY")
                .ok()
                .map(|v| ChromeKey::parse(&v))
                .transpose();
            let key = match key {
                Ok(Some(k)) => k,
                Ok(None) => ChromeKey::Auto,
                Err(e) => {
                    report.failures.push(format!("OBSCURA_CHROME_KEY: {e}"));
                    ChromeKey::Auto
                }
            };
            match import_chrome_profile(Path::new(dir), &key, jar) {
                Ok(n) => {
                    report.chrome = n;
                    tracing::info!("Imported {} Chrome cookies from {}", n, dir);
                }
                Err(e) => {
                    report.failures.push(format!("Chrome profile {dir}: {e}"));
                    tracing::warn!("Chrome cookie import from {} failed: {}", dir, e);
                }
            }
        }
    }
    if let Ok(dir) = std::env::var("OBSCURA_FIREFOX_PROFILE") {
        let dir = dir.trim();
        if !dir.is_empty() {
            match import_firefox_profile(Path::new(dir), jar) {
                Ok(n) => {
                    report.firefox = n;
                    tracing::info!("Imported {} Firefox cookies from {}", n, dir);
                }
                Err(e) => {
                    report.failures.push(format!("Firefox profile {dir}: {e}"));
                    tracing::warn!("Firefox cookie import from {} failed: {}", dir, e);
                }
            }
        }
    }
    report
}

/// Import cookies from a Chrome/Chromium profile directory (the directory
/// that contains `Cookies` or `Network/Cookies`). Returns the number of
/// cookies accepted into the jar.
pub fn import_chrome_profile(
    profile_dir: &Path,
    key: &ChromeKey,
    jar: &CookieJar,
) -> Result<usize, ImportError> {
    let source = [profile_dir.join("Network/Cookies"), profile_dir.join("Cookies")]
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| {
            ImportError::new(format!(
                "no Cookies database under {}",
                profile_dir.display()
            ))
        })?;
    let snapshot = SqliteSnapshot::open(&source).map_err(ImportError::new)?;
    let conn = &snapshot.connection;
    // meta.version >= 24 means decrypted values carry a leading SHA-256 of
    // the cookie's host_key that must be stripped (Chromium schema change).
    let schema_version: i64 = conn
        .query_row("SELECT value FROM meta WHERE key = 'version'", [], |r| {
            r.get::<_, String>(0)
        })
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut keys: Option<Vec<Vec<u8>>> = None;
    let mut statement = conn
        .prepare(
            "SELECT host_key,name,value,encrypted_value,path,expires_utc,is_secure,is_httponly,samesite FROM cookies",
        )
        .map_err(|e| ImportError::new(format!("Chrome cookies schema unsupported: {e}")))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)? != 0,
                row.get::<_, i64>(7)? != 0,
                row.get::<_, i64>(8)?,
            ))
        })
        .map_err(ImportError::new)?;

    let now = unix_now();
    let mut imported = Vec::new();
    for row in rows {
        let (host, name, value, encrypted, path, expires_utc, secure, http_only, same_site) =
            row.map_err(ImportError::new)?;
        let expires = chrome_expiry(expires_utc);
        // Rows kept in the browser's store past their expiry are dead weight;
        // importing one as a session cookie would resurrect it.
        if expires.is_some_and(|e| e <= now) {
            continue;
        }
        let value = if encrypted.is_empty() {
            value
        } else {
            if keys.is_none() {
                keys = Some(resolve_chrome_keys(profile_dir, key).map_err(ImportError::new)?);
            }
            decrypt_chrome_value(&encrypted, keys.as_ref().unwrap(), &host, schema_version)
                .map_err(ImportError::new)?
        };
        let host_only = !host.starts_with('.');
        imported.push((
            CookieInfo {
                name,
                value,
                domain: host,
                path,
                secure,
                http_only,
                same_site: chrome_same_site(same_site).to_string(),
                expires,
            },
            host_only,
        ));
    }
    let n = imported.len();
    jar.set_cookies_from_cdp_with_scope(imported);
    Ok(n)
}

/// Import cookies from a Firefox profile directory (`cookies.sqlite`).
/// Container/private-browsing cookies (non-empty `originAttributes`) are
/// skipped: flattening them into the default partition would change their
/// isolation semantics.
pub fn import_firefox_profile(profile_dir: &Path, jar: &CookieJar) -> Result<usize, ImportError> {
    let source = profile_dir.join("cookies.sqlite");
    if !source.is_file() {
        return Err(ImportError::new(format!(
            "no cookies.sqlite under {}",
            profile_dir.display()
        )));
    }
    let snapshot = SqliteSnapshot::open(&source).map_err(ImportError::new)?;
    let mut statement = snapshot
        .connection
        .prepare(
            "SELECT name,value,host,path,expiry,isSecure,isHttpOnly,sameSite,originAttributes FROM moz_cookies",
        )
        .map_err(|e| ImportError::new(format!("Firefox cookies schema unsupported: {e}")))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)? != 0,
                row.get::<_, i64>(6)? != 0,
                row.get::<_, i64>(7)?,
                row.get::<_, String>(8)?,
            ))
        })
        .map_err(ImportError::new)?;

    let now = unix_now();
    let mut imported = Vec::new();
    for row in rows {
        let (name, value, host, path, expiry, secure, http_only, same_site, origin_attributes) =
            row.map_err(ImportError::new)?;
        if !origin_attributes.is_empty() {
            continue;
        }
        if expiry > 0 && expiry <= now {
            continue;
        }
        imported.push((
            CookieInfo {
                name,
                value,
                domain: host.clone(),
                path,
                secure,
                http_only,
                same_site: chrome_same_site(same_site).to_string(),
                expires: if expiry > 0 { Some(expiry) } else { None },
            },
            !host.starts_with('.'),
        ));
    }
    let n = imported.len();
    jar.set_cookies_from_cdp_with_scope(imported);
    Ok(n)
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn chrome_same_site(v: i64) -> &'static str {
    match v {
        0 => "None",
        2 => "Strict",
        _ => "Lax",
    }
}

fn chrome_expiry(micros: i64) -> Option<i64> {
    if micros == 0 {
        return None;
    }
    Some((micros - CHROME_EPOCH_OFFSET_MICROS) / 1_000_000)
}

/// Copy a SQLite database plus its WAL/SHM companions into a temp dir before
/// opening it read-only. The browser can keep the live files locked; reading
/// the snapshot never blocks on its writer.
struct SqliteSnapshot {
    connection: Connection,
    _dir: tempfile::TempDir,
}

impl SqliteSnapshot {
    fn open(source: &Path) -> Result<Self, String> {
        let dir = tempfile::tempdir().map_err(|e| format!("snapshot dir: {e}"))?;
        let target = dir.path().join("db.sqlite");
        std::fs::copy(source, &target)
            .map_err(|e| format!("snapshot {}: {e}", source.display()))?;
        for suffix in ["-wal", "-shm"] {
            let mut companion = source.as_os_str().to_owned();
            companion.push(suffix);
            let companion = PathBuf::from(companion);
            if companion.is_file() {
                std::fs::copy(&companion, dir.path().join(format!("db.sqlite{suffix}")))
                    .map_err(|e| format!("snapshot {}: {e}", companion.display()))?;
            }
        }
        let connection =
            Connection::open_with_flags(&target, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| format!("open snapshot {}: {e}", source.display()))?;
        Ok(Self { connection, _dir: dir })
    }
}

/// Candidate decryption keys for a Chrome profile, most likely first.
/// Local State's `os_crypt.encrypted_key` holds the AES-GCM key Chrome >= 80
/// uses, wrapped in AES-128-CBC under the legacy "peanuts" password; the
/// peanuts key itself stays in the list for pre-v10 cookie values.
fn resolve_chrome_keys(profile_dir: &Path, key: &ChromeKey) -> Result<Vec<Vec<u8>>, String> {
    match key {
        ChromeKey::Base64(k) => Ok(vec![k.clone()]),
        ChromeKey::Peanuts => Ok(vec![peanuts_key()]),
        ChromeKey::System => system_chrome_key().map(|k| vec![k]),
        ChromeKey::Auto => {
            #[cfg(target_os = "macos")]
            {
                return system_chrome_key().map(|k| vec![k]);
            }
            #[cfg(not(target_os = "macos"))]
            {
                let mut keys = Vec::new();
                if let Some(local_state) = find_local_state(profile_dir) {
                    if let Ok(k) = unwrap_local_state_key(&local_state) {
                        keys.push(k);
                    }
                }
                keys.push(peanuts_key());
                Ok(keys)
            }
        }
    }
}

/// `Local State` sits next to the profile directory inside the user-data dir
/// (`<user-data-dir>/Default` -> `<user-data-dir>/Local State`).
fn find_local_state(profile_dir: &Path) -> Option<PathBuf> {
    let parent = profile_dir.parent()?;
    let direct = parent.join("Local State");
    if direct.is_file() {
        return Some(direct);
    }
    // Some callers pass the user-data dir itself instead of the profile dir.
    let here = profile_dir.join("Local State");
    if here.is_file() {
        return Some(here);
    }
    None
}

fn unwrap_local_state_key(local_state: &Path) -> Result<Vec<u8>, String> {
    let text = std::fs::read_to_string(local_state)
        .map_err(|e| format!("read {}: {e}", local_state.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse Local State: {e}"))?;
    let encoded = json
        .get("os_crypt")
        .and_then(|o| o.get("encrypted_key"))
        .and_then(|k| k.as_str())
        .ok_or_else(|| "Local State has no os_crypt.encrypted_key".to_string())?;
    let wrapped = B64
        .decode(encoded)
        .map_err(|e| format!("decode encrypted_key: {e}"))?;
    let payload = wrapped
        .get(3..)
        .filter(|_| wrapped.starts_with(b"v10") || wrapped.starts_with(b"v11"))
        .ok_or_else(|| "encrypted_key has an unexpected prefix".to_string())?;
    aes_128_cbc_decrypt(&peanuts_key(), payload)
        .map_err(|e| format!("unwrap Local State key: {e}"))
}

fn peanuts_key() -> Vec<u8> {
    let mut key = [0u8; 16];
    pbkdf2::pbkdf2_hmac::<sha1::Sha1>(b"peanuts", b"saltysalt", 1, &mut key);
    key.to_vec()
}

#[cfg(target_os = "macos")]
fn system_chrome_key() -> Result<Vec<u8>, String> {
    let output = std::process::Command::new("security")
        .args(["find-generic-password", "-w", "-s", "Chrome Safe Storage"])
        .output()
        .map_err(|e| format!("query macOS Keychain: {e}"))?;
    if !output.status.success() {
        return Err("macOS Keychain did not grant access to Chrome Safe Storage".to_string());
    }
    let password =
        String::from_utf8(output.stdout).map_err(|_| "Keychain password is not UTF-8")?;
    let mut key = [0u8; 16];
    pbkdf2::pbkdf2_hmac::<sha1::Sha1>(
        password.trim_end().as_bytes(),
        b"saltysalt",
        1003,
        &mut key,
    );
    Ok(key.to_vec())
}

#[cfg(not(target_os = "macos"))]
fn system_chrome_key() -> Result<Vec<u8>, String> {
    Err("`system` Chrome key is only supported on macOS; use `auto`, `peanuts`, or `base64:<raw-key>` on this platform".to_string())
}

fn decrypt_chrome_value(
    encrypted: &[u8],
    keys: &[Vec<u8>],
    host: &str,
    schema_version: i64,
) -> Result<String, String> {
    let payload = encrypted
        .strip_prefix(b"v10")
        .or_else(|| encrypted.strip_prefix(b"v11"))
        .ok_or_else(|| "unsupported encrypted cookie prefix".to_string())?;
    let mut last_err = String::new();
    for key in keys {
        let plaintext = if key.len() == 32 && payload.len() > 12 + 16 {
            aes_256_gcm_decrypt(key, payload)
        } else if key.len() == 16 {
            aes_128_cbc_decrypt(key, payload)
        } else {
            continue;
        };
        let plaintext = match plaintext {
            Ok(p) => p,
            Err(e) => {
                last_err = e;
                continue;
            }
        };
        let plaintext = if schema_version >= 24 {
            let digest = Sha256::digest(host.as_bytes());
            match plaintext.strip_prefix(digest.as_slice()) {
                Some(rest) => rest.to_vec(),
                None => {
                    last_err = "decrypted value host digest mismatch".to_string();
                    continue;
                }
            }
        } else {
            plaintext
        };
        return String::from_utf8(plaintext)
            .map_err(|_| "decrypted cookie value is not UTF-8".to_string());
    }
    Err(if last_err.is_empty() {
        "no Chrome key could decrypt the cookie".to_string()
    } else {
        last_err
    })
}

fn aes_256_gcm_decrypt(key: &[u8], payload: &[u8]) -> Result<Vec<u8>, String> {
    use aes_gcm::aead::{Aead, KeyInit};
    let cipher = aes_gcm::Aes256Gcm::new_from_slice(key).map_err(|e| e.to_string())?;
    let nonce = aes_gcm::Nonce::from_slice(&payload[..12]);
    cipher
        .decrypt(nonce, &payload[12..])
        .map_err(|_| "AES-GCM authentication failed".to_string())
}

fn aes_128_cbc_decrypt(key: &[u8], payload: &[u8]) -> Result<Vec<u8>, String> {
    use cbc::cipher::{BlockDecryptMut, KeyIvInit};
    type Dec = cbc::Decryptor<aes::Aes128>;
    Dec::new(key.into(), &[b' '; 16].into())
        .decrypt_padded_vec_mut::<cbc::cipher::block_padding::Pkcs7>(payload)
        .map_err(|_| "AES-CBC decryption failed".to_string())
}

#[derive(Debug)]
pub struct ImportError(pub String);

impl ImportError {
    fn new<E: std::fmt::Display>(e: E) -> Self {
        Self(e.to_string())
    }
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ImportError {}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_gcm::aead::{Aead, KeyInit};

    fn chrome_db(dir: &Path, cookies: &[(&str, &str, &str, &[u8], i64, bool, bool, i64)], schema_version: i64) {
        let conn = Connection::open(dir.join("Network").join("Cookies")).unwrap();
        conn.execute("CREATE TABLE meta (key LONGVARCHAR NOT NULL UNIQUE PRIMARY KEY, value LONGVARCHAR)", []).unwrap();
        conn.execute("INSERT INTO meta VALUES ('version', ?1)", [schema_version.to_string()]).unwrap();
        conn.execute(
            "CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, encrypted_value BLOB, path TEXT, expires_utc INTEGER, is_secure INTEGER, is_httponly INTEGER, samesite INTEGER)",
            [],
        )
        .unwrap();
        for (host, name, value, enc, exp, sec, httponly, ss) in cookies {
            conn.execute(
                "INSERT INTO cookies VALUES (?1,?2,?3,?4,'/',?5,?6,?7,?8)",
                rusqlite::params![host, name, value, enc, exp, *sec as i64, *httponly as i64, ss],
            )
            .unwrap();
        }
    }

    fn gcm_cookie(key: &[u8; 32], host: &str, value: &str, schema_version: i64) -> Vec<u8> {
        let plaintext = if schema_version >= 24 {
            let mut p = Sha256::digest(host.as_bytes()).to_vec();
            p.extend_from_slice(value.as_bytes());
            p
        } else {
            value.as_bytes().to_vec()
        };
        let cipher = aes_gcm::Aes256Gcm::new(key.into());
        let nonce_bytes = [7u8; 12];
        let ct = cipher
            .encrypt(aes_gcm::Nonce::from_slice(&nonce_bytes), plaintext.as_slice())
            .unwrap();
        let mut out = b"v10".to_vec();
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ct);
        out
    }

    fn jar_header(jar: &CookieJar, url: &str) -> String {
        jar.get_cookie_header(&url::Url::parse(url).unwrap())
    }

    #[test]
    fn chrome_gcm_cookie_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Network")).unwrap();
        let key = [42u8; 32];
        let enc = gcm_cookie(&key, ".example.com", "session-value", 24);
        let expires = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 86_400) as i64
            * 1_000_000
            + CHROME_EPOCH_OFFSET_MICROS;
        chrome_db(
            dir.path(),
            &[(".example.com", "sid", "", &enc, expires, true, true, 1)],
            24,
        );
        let jar = CookieJar::new();
        let n = import_chrome_profile(
            dir.path(),
            &ChromeKey::Base64(key.to_vec()),
            &jar,
        )
        .unwrap();
        assert_eq!(n, 1);
        // Secure cookie: only sent over https; domain cookie: subdomains too.
        assert_eq!(jar_header(&jar, "https://sub.example.com/"), "sid=session-value");
        assert_eq!(jar_header(&jar, "http://example.com/"), "");
    }

    #[test]
    fn chrome_plaintext_and_expired() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Network")).unwrap();
        let future = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 86_400) as i64
            * 1_000_000
            + CHROME_EPOCH_OFFSET_MICROS;
        chrome_db(
            dir.path(),
            &[
                ("example.com", "plain", "v1", b"", future, false, false, 1),
                ("example.com", "dead", "v0", b"", 1, false, false, 1),
            ],
            20,
        );
        let jar = CookieJar::new();
        let n = import_chrome_profile(dir.path(), &ChromeKey::Peanuts, &jar).unwrap();
        // The expired row is skipped at read time, not imported.
        assert_eq!(n, 1);
        // Host-only cookie reaches the exact host; the long-expired row is dropped by the jar.
        assert_eq!(jar_header(&jar, "http://example.com/"), "plain=v1");
        assert_eq!(jar_header(&jar, "http://sub.example.com/"), "");
    }

    #[test]
    fn chrome_local_state_unwraps_key() {
        let dir = tempfile::tempdir().unwrap();
        let profile = dir.path().join("Default");
        std::fs::create_dir_all(profile.join("Network")).unwrap();
        let app_key = [9u8; 32];
        // Wrap the app key the way Chrome writes os_crypt.encrypted_key:
        // base64("v10" || AES-128-CBC(peanuts, iv=16 spaces, app_key)).
        use cbc::cipher::{BlockEncryptMut, KeyIvInit};
        type Enc = cbc::Encryptor<aes::Aes128>;
        let wrapped = Enc::new((&peanuts_key()[..16]).into(), &[b' '; 16].into())
            .encrypt_padded_vec_mut::<cbc::cipher::block_padding::Pkcs7>(&app_key);
        let mut blob = b"v10".to_vec();
        blob.extend_from_slice(&wrapped);
        let local_state = serde_json::json!({"os_crypt": {"encrypted_key": B64.encode(&blob)}});
        std::fs::write(dir.path().join("Local State"), local_state.to_string()).unwrap();

        let enc = gcm_cookie(&app_key, "site.test", "auto-key-works", 20);
        let expires = 50_000_000_000_000_000i64;
        chrome_db(
            &profile,
            &[("site.test", "k", "", &enc, expires, false, false, 1)],
            20,
        );
        let jar = CookieJar::new();
        let n = import_chrome_profile(&profile, &ChromeKey::Auto, &jar).unwrap();
        assert_eq!(n, 1);
        assert_eq!(jar_header(&jar, "http://site.test/"), "k=auto-key-works");
    }

    #[test]
    fn firefox_cookies_skip_containers() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("cookies.sqlite")).unwrap();
        conn.execute(
            "CREATE TABLE moz_cookies (name TEXT, value TEXT, host TEXT, path TEXT, expiry INTEGER, isSecure INTEGER, isHttpOnly INTEGER, sameSite INTEGER, schemeMap INTEGER, originAttributes TEXT)",
            [],
        )
        .unwrap();
        let future = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 86_400) as i64;
        conn.execute(
            "INSERT INTO moz_cookies VALUES ('ff','v','example.org','/',?1,0,0,1,0,'')",
            [future],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO moz_cookies VALUES ('container','v','example.org','/',?1,0,0,1,0,'userContextId=1')",
            [future],
        )
        .unwrap();
        drop(conn);
        let jar = CookieJar::new();
        let n = import_firefox_profile(dir.path(), &jar).unwrap();
        assert_eq!(n, 1);
        assert_eq!(jar_header(&jar, "http://example.org/"), "ff=v");
    }
}
