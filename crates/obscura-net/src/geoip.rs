//! Egress-IP identity lookup (port of camofox's `geoip` launch option).
//!
//! When traffic exits through a proxy, the browser's identity surfaces
//! (timezone, geolocation, locale, Accept-Language) should match the exit IP's
//! region or fingerprinting scripts see a mismatch. This module resolves the
//! identity once at startup by querying an IP-geo endpoint *through the proxy*,
//! so the service sees the egress address.
//!
//! The lookup is fail-open: any transport or parse error returns `None` and the
//! caller keeps its defaults, mirroring camofox's
//! `buildLaunchOptionsWithGeoipFallback` retry without geoip.

use serde_json::Value;
use std::time::Duration;

/// Default endpoint: ipwho.is is HTTPS, keyless, and returns country, IANA
/// timezone, coordinates, and the region's languages in one response.
/// `OBSCURA_GEOIP_ENDPOINT` overrides it.
pub const DEFAULT_ENDPOINT: &str = "https://ipwho.is/";

/// Matches camofox's `GEOIP_SETUP_TIMEOUT_MS`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Identity derived from the egress IP.
#[derive(Debug, Clone, PartialEq)]
pub struct EgressIdentity {
    pub ip: String,
    /// IANA timezone name, e.g. "Europe/Berlin".
    pub timezone: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    /// BCP-47 locale for the navigator surfaces, e.g. "de-DE".
    pub locale: Option<String>,
    /// Matching Accept-Language header value, e.g. "de-DE,de;q=0.9".
    pub accept_language: Option<String>,
}

/// Parse an ipwho.is-shaped response body. Returns None when the endpoint
/// reports failure (private/reserved egress IP, rate limit) or the body lacks
/// an IP address; each identity surface is optional individually.
fn parse_identity(body: &Value) -> Option<EgressIdentity> {
    if body.get("success").and_then(Value::as_bool) == Some(false) {
        return None;
    }
    let ip = body.get("ip").and_then(Value::as_str)?.to_string();
    let timezone = body
        .get("timezone")
        .and_then(|tz| tz.get("id").or_else(|| tz.get("name")))
        .and_then(Value::as_str)
        .or_else(|| body.get("timezone").and_then(Value::as_str))
        .filter(|tz| !tz.is_empty())
        .map(str::to_owned);
    let latitude = body.get("latitude").and_then(Value::as_f64);
    let longitude = body.get("longitude").and_then(Value::as_f64);
    // BCP-47 tag: first listed language + country code ("de" + "DE" -> "de-DE").
    // Accept either the `languages` CSV (ipwho.is) or `lang`/`language` string
    // (ipapi-style) so OBSCURA_GEOIP_ENDPOINT alternatives keep working.
    let country = body.get("country_code").and_then(Value::as_str).unwrap_or("");
    let language = body
        .get("languages")
        .and_then(Value::as_str)
        .or_else(|| body.get("lang").and_then(Value::as_str))
        .or_else(|| body.get("language").and_then(Value::as_str))
        .and_then(|languages| languages.split(',').next())
        .map(str::trim)
        .filter(|lang| !lang.is_empty() && lang.chars().all(|c| c.is_ascii_alphabetic()))
        .map(str::to_owned);
    let locale = language.as_deref().map(|lang| {
        if country.is_empty() { lang.to_string() } else { format!("{lang}-{country}") }
    });
    let accept_language = language.map(|lang| {
        if country.is_empty() { lang.clone() } else { format!("{lang}-{country},{lang};q=0.9") }
    });
    Some(EgressIdentity { ip, timezone, latitude, longitude, locale, accept_language })
}

/// Resolve the egress identity through `proxy` (a proxy URL such as
/// `http://user:pass@host:port` or `socks5://...`). `proxy` may be None for a
/// direct lookup (tests, or diagnosing the real exit IP). Always fails open.
pub async fn lookup_egress_identity(
    proxy: Option<&str>,
    endpoint: &str,
    timeout: Duration,
) -> Option<EgressIdentity> {
    let mut builder = reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::limited(3));
    if let Some(proxy) = proxy {
        builder = builder.proxy(reqwest::Proxy::all(proxy).ok()?);
    }
    let client = builder.build().ok()?;
    let response = client.get(endpoint).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    // reqwest's `json` feature is not enabled workspace-wide; parse the body.
    let body = serde_json::from_str::<Value>(&response.text().await.ok()?).ok()?;
    parse_identity(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_ipwho_is_response() {
        let body = json!({
            "ip": "203.0.113.7", "success": true, "country_code": "DE",
            "latitude": 52.52, "longitude": 13.405,
            "timezone": {"id": "Europe/Berlin"},
            "languages": "de,en,tr",
        });
        let identity = parse_identity(&body).unwrap();
        assert_eq!(identity.ip, "203.0.113.7");
        assert_eq!(identity.timezone.as_deref(), Some("Europe/Berlin"));
        assert_eq!(identity.latitude, Some(52.52));
        assert_eq!(identity.longitude, Some(13.405));
        assert_eq!(identity.locale.as_deref(), Some("de-DE"));
        assert_eq!(identity.accept_language.as_deref(), Some("de-DE,de;q=0.9"));
    }

    #[test]
    fn failed_lookup_body_yields_no_identity() {
        assert!(parse_identity(&json!({"success": false, "ip": null})).is_none());
        assert!(parse_identity(&json!({})).is_none());
    }

    #[test]
    fn missing_language_leaves_locale_unset() {
        let body = json!({"ip": "203.0.113.7", "success": true, "country_code": "JP",
            "timezone": {"id": "Asia/Tokyo"}});
        let identity = parse_identity(&body).unwrap();
        assert_eq!(identity.timezone.as_deref(), Some("Asia/Tokyo"));
        assert!(identity.locale.is_none());
        assert!(identity.accept_language.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unreachable_endpoint_fails_open() {
        // Port 1 is closed on loopback; the lookup must return None quickly
        // rather than error out of startup.
        let identity = lookup_egress_identity(
            None,
            "http://127.0.0.1:1/",
            Duration::from_millis(1500),
        ).await;
        assert!(identity.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lookup_reads_identity_from_a_local_endpoint() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::AsyncWriteExt;
            let body = r#"{"ip":"198.51.100.4","success":true,"country_code":"VN","latitude":10.82,"longitude":106.63,"timezone":{"id":"Asia/Ho_Chi_Minh"},"languages":"vi,en"}"#;
            let response = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body);
            let _ = socket.write_all(response.as_bytes()).await;
        });
        let identity = lookup_egress_identity(
            None,
            &format!("http://127.0.0.1:{port}/"),
            Duration::from_secs(5),
        ).await.unwrap();
        assert_eq!(identity.ip, "198.51.100.4");
        assert_eq!(identity.timezone.as_deref(), Some("Asia/Ho_Chi_Minh"));
        assert_eq!(identity.locale.as_deref(), Some("vi-VN"));
    }
}
