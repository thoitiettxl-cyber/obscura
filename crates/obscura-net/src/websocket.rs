//! Real WebSocket client transport for page-side `WebSocket` API.
//!
//! One connection is owned by a driver task (spawned by the caller with
//! [`WsSession::run`]). The ops layer sends payloads through an unbounded
//! channel and receives inbound events through another; the driver owns the
//! socket itself so protocol replies (pong, close echo) are emitted without
//! holding a JS-side borrow across an await point.

use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::{http, protocol::WebSocketConfig, Message};
use tokio_tungstenite::WebSocketStream;
use url::Url;

use crate::cookies::CookieJar;
use obscura_ssrf::{env_allows_private_network, is_forbidden_ip};

/// Cap on a single (re-assembled) WebSocket message, mirroring the response
/// body cap used elsewhere in the network stack.
pub const WS_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Time to keep reading after sending our Close frame, waiting for the peer's
/// close reply before giving up.
const CLOSE_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Debug, Error)]
pub enum WsError {
    #[error("invalid WebSocket URL: {0}")]
    Url(String),
    #[error("SSRF blocked: {0}")]
    Ssrf(String),
    #[error("proxy error: {0}")]
    Proxy(String),
    #[error("connect failed: {0}")]
    Connect(String),
    #[error("TLS error: {0}")]
    Tls(String),
    #[error("handshake failed: {0}")]
    Handshake(String),
    #[error("server rejected the upgrade: {0}")]
    Rejected(String),
}

/// Inbound event delivered to the page.
#[derive(Debug)]
pub enum WsInbound {
    Text(String),
    Binary(Vec<u8>),
    /// Terminal: peer or local close completed. `code`/`reason` describe the
    /// peer's close frame (1000/"" for a clean local close).
    Close(u16, String, bool),
    /// Terminal: transport or protocol error.
    Error(String),
}

/// Outbound payload the page wants on the wire.
#[derive(Debug)]
pub enum WsOutbound {
    Text(String),
    Binary(Vec<u8>),
    Close(u16, String),
}

/// Everything the caller needs to run one connection.
pub struct WsSession {
    socket: WebSocketStream<BoxStream>,
    /// `Sec-WebSocket-Protocol` the server negotiated, if any.
    pub protocol: String,
}

trait StreamDyn: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> StreamDyn for T {}
type BoxStream = Box<dyn StreamDyn>;

pub struct WsOptions<'a> {
    /// Browser-profile-style header values (already validated upstream).
    pub extra_headers: &'a [(String, String)],
    /// Sub-protocols requested by `new WebSocket(url, protocols)`.
    pub protocols: &'a [String],
    /// HTTP/HTTPS/SOCKS5 proxy URL, if configured.
    pub proxy: Option<&'a str>,
    /// Allow loopback/private destinations (SSRF bypass for local testing).
    pub allow_private: bool,
}

/// Connect, perform the opening handshake, and return the socket plus the
/// negotiated sub-protocol. Does not spawn anything; drive the session with
/// [`WsSession::run`] from the caller's executor.
///
/// Browsers tolerate servers that ignore `Sec-WebSocket-Protocol` entirely;
/// tungstenite reports that as a handshake error, so a connect that failed
/// solely on that ground is retried once without the header.
pub async fn connect(url: &Url, jar: Option<&CookieJar>, options: &WsOptions<'_>) ->
    Result<WsSession, WsError>
{
    match connect_once(url, jar, options).await {
        Err(WsError::Handshake(ref e))
            if !options.protocols.is_empty() && e.contains("no subprotocol") =>
        {
            let retry = WsOptions { protocols: &[], ..*options };
            connect_once(url, jar, &retry).await
        }
        result => result,
    }
}

async fn connect_once(url: &Url, jar: Option<&CookieJar>, options: &WsOptions<'_>) ->
    Result<WsSession, WsError>
{
    let secure = match url.scheme() {
        "ws" => false,
        "wss" => true,
        other => return Err(WsError::Url(format!("unsupported scheme '{other}'"))),
    };
    let host = url
        .host_str()
        .ok_or_else(|| WsError::Url("missing host".into()))?
        .to_string();
    let port = url.port_or_known_default().unwrap_or(if secure { 443 } else { 80 });

    ssrf_check(&host, port, options.allow_private).await?;

    let mut stream = open_tcp_stream(&host, port, options).await?;
    if secure {
        stream = tls_wrap(stream, &host).await?;
    }

    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|e| WsError::Url(e.to_string()))?;
    let headers = request.headers_mut();
    if !options.protocols.is_empty() {
        headers.insert(
            http::header::SEC_WEBSOCKET_PROTOCOL,
            http::HeaderValue::from_str(&options.protocols.join(", "))
                .map_err(|e| WsError::Url(e.to_string()))?,
        );
    }
    if let Some(jar) = jar {
        let cookie = jar.get_cookie_header_same_site(url);
        if !cookie.is_empty() {
            if let Ok(value) = http::HeaderValue::from_str(&cookie) {
                headers.insert(http::header::COOKIE, value);
            }
        }
    }
    for (name, value) in options.extra_headers {
        if let (Ok(name), Ok(value)) = (
            name.parse::<http::header::HeaderName>(),
            http::HeaderValue::from_str(value),
        ) {
            headers.insert(name, value);
        }
    }

    let config = WebSocketConfig::default()
        .max_message_size(Some(WS_MAX_MESSAGE_SIZE))
        .max_frame_size(Some(WS_MAX_MESSAGE_SIZE));
    let (socket, response) =
        tokio_tungstenite::client_async_with_config(request, stream, Some(config))
            .await
            .map_err(|e| match e {
                tungstenite::Error::Http(resp) => {
                    WsError::Rejected(format!("HTTP {}", resp.status()))
                }
                other => WsError::Handshake(other.to_string()),
            })?;
    if response.status() != http::StatusCode::SWITCHING_PROTOCOLS {
        return Err(WsError::Rejected(format!("HTTP {}", response.status())));
    }
    let protocol = response
        .headers()
        .get(http::header::SEC_WEBSOCKET_PROTOCOL)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    Ok(WsSession { socket, protocol })
}

async fn ssrf_check(host: &str, port: u16, allow_private: bool) -> Result<(), WsError> {
    if allow_private || env_allows_private_network() {
        return Ok(());
    }
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| WsError::Connect(format!("DNS lookup failed: {e}")))?;
    let mut saw_any = false;
    for addr in addrs {
        saw_any = true;
        if is_forbidden_ip(addr.ip()) {
            return Err(WsError::Ssrf(format!("{host}:{port} resolves to forbidden address")));
        }
    }
    if !saw_any {
        return Err(WsError::Connect(format!("DNS lookup returned no addresses for {host}")));
    }
    Ok(())
}

/// Open the raw TCP stream, tunneling through the configured proxy when set.
async fn open_tcp_stream(
    host: &str,
    port: u16,
    options: &WsOptions<'_>,
) -> Result<BoxStream, WsError> {
    let Some(proxy) = options.proxy else {
        let stream = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect((host, port)))
            .await
            .map_err(|_| WsError::Connect("connect timed out".into()))?
            .map_err(|e| WsError::Connect(e.to_string()))?;
        return Ok(Box::new(stream));
    };
    let proxy_url = Url::parse(proxy).map_err(|e| WsError::Proxy(e.to_string()))?;
    let proxy_host = proxy_url
        .host_str()
        .ok_or_else(|| WsError::Proxy("proxy URL has no host".into()))?;
    let proxy_port = proxy_url
        .port_or_known_default()
        .unwrap_or(8080);
    match proxy_url.scheme() {
        "http" | "https" => {
            let stream =
                tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect((proxy_host, proxy_port)))
                    .await
                    .map_err(|_| WsError::Proxy("proxy connect timed out".into()))?
                    .map_err(|e| WsError::Proxy(e.to_string()))?;
            // HTTPS proxies would need a TLS hop before CONNECT; the network
            // stack treats proxy URLs as opaque upstreams, so support the
            // common plaintext case and report the rest clearly.
            if proxy_url.scheme() == "https" {
                return Err(WsError::Proxy(
                    "https proxy endpoints are not supported for WebSocket".into(),
                ));
            }
            http_connect(stream, host, port, &proxy_url).await.map(|s| Box::new(s) as BoxStream)
        }
        "socks5" | "socks5h" => {
            let stream = if proxy_url.username().is_empty() {
                tokio_socks::tcp::Socks5Stream::connect(
                    (proxy_host, proxy_port),
                    (host, port),
                )
                .await
            } else {
                tokio_socks::tcp::Socks5Stream::connect_with_password(
                    (proxy_host, proxy_port),
                    (host, port),
                    proxy_url.username(),
                    proxy_url.password().unwrap_or(""),
                )
                .await
            };
            stream
                .map(|s| Box::new(s) as BoxStream)
                .map_err(|e| WsError::Proxy(e.to_string()))
        }
        other => Err(WsError::Proxy(format!(
            "unsupported proxy scheme '{other}' for WebSocket"
        ))),
    }
}

/// Issue an HTTP CONNECT through `stream` and verify the 2xx reply.
async fn http_connect(
    mut stream: TcpStream,
    host: &str,
    port: u16,
    proxy_url: &Url,
) -> Result<TcpStream, WsError> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let mut request = format!("CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n");
    if !proxy_url.username().is_empty() {
        use base64::Engine;
        let creds = format!(
            "{}:{}",
            proxy_url.username(),
            proxy_url.password().unwrap_or("")
        );
        let encoded = base64::engine::general_purpose::STANDARD.encode(creds);
        request.push_str(&format!("Proxy-Authorization: Basic {encoded}\r\n"));
    }
    request.push_str("\r\n");

    let mut reader = BufReader::new(&mut stream);
    tokio::time::timeout(CONNECT_TIMEOUT, async {
        reader.get_mut().write_all(request.as_bytes()).await?;
        let mut status = String::new();
        reader.read_line(&mut status).await?;
        // Status line: HTTP/1.1 200 ...
        let code = status
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        // Drain headers; they end at an empty line.
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await? == 0 {
                break;
            }
            if line == "\r\n" || line == "\n" {
                break;
            }
        }
        Ok::<u16, std::io::Error>(code)
    })
    .await
    .map_err(|_| WsError::Proxy("CONNECT timed out".into()))?
    .map_err(|e| WsError::Proxy(e.to_string()))
    .and_then(|code| {
        if (200..300).contains(&code) {
            Ok(())
        } else {
            Err(WsError::Proxy(format!("CONNECT failed: HTTP {code}")))
        }
    })?;
    Ok(stream)
}

/// Wrap `stream` in rustls with the same trust store family as the reqwest
/// path (webpki root certificates).
async fn tls_wrap(stream: BoxStream, host: &str) -> Result<BoxStream, WsError> {
    let mut roots = tokio_rustls::rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = tokio_rustls::rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name = tokio_rustls::rustls::pki_types::ServerName::try_from(host.to_string())
        .map_err(|e| WsError::Tls(format!("invalid server name: {e}")))?;
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let stream = tokio::time::timeout(CONNECT_TIMEOUT, connector.connect(name, stream))
        .await
        .map_err(|_| WsError::Tls("TLS handshake timed out".into()))?
        .map_err(|e| WsError::Tls(e.to_string()))?;
    Ok(Box::new(stream))
}

impl WsSession {
    /// Drive the socket until the peer closes, the transport fails, or all
    /// senders are dropped. Inbound events go to `events`; payloads from
    /// `writes` go on the wire. Pings are answered eagerly; an incoming Close
    /// frame is echoed before the task exits.
    pub async fn run(
        mut self,
        events: mpsc::UnboundedSender<WsInbound>,
        mut writes: mpsc::UnboundedReceiver<WsOutbound>,
    ) {
        let mut closing = false;
        let mut terminated = false;
        loop {
            let next = if closing {
                match tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, self.socket.next()).await {
                    Ok(next) => next,
                    Err(_) => break,
                }
            } else {
                tokio::select! {
                    next = self.socket.next() => next,
                    outbound = writes.recv() => match outbound {
                        Some(WsOutbound::Text(text)) => {
                            let _ = self.socket.send(Message::Text(text.into())).await;
                            continue;
                        }
                        Some(WsOutbound::Binary(bytes)) => {
                            let _ = self.socket.send(Message::Binary(bytes.into())).await;
                            continue;
                        }
                        Some(WsOutbound::Close(code, reason)) => {
                            let _ = self
                                .socket
                                .send(Message::Close(Some(CloseFrame {
                                    code: CloseCode::from(code),
                                    reason: reason.into(),
                                })))
                                .await;
                            closing = true;
                            continue;
                        }
                        // Every sender is gone (page navigated away or the
                        // resource was dropped): close politely and finish.
                        None => {
                            let _ = self.socket.close(None).await;
                            break;
                        }
                    },
                }
            };
            match next {
                Some(Ok(Message::Text(text))) => {
                    if events.send(WsInbound::Text(text.to_string())).is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Binary(bytes))) => {
                    if events.send(WsInbound::Binary(bytes.to_vec())).is_err() {
                        break;
                    }
                }
                // tungstenite already queues a pong for an incoming ping; send
                // an explicit reply anyway so a long quiet spell on the write
                // side still answers promptly.
                Some(Ok(Message::Ping(payload))) => {
                    let _ = self.socket.send(Message::Pong(payload)).await;
                }
                Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
                Some(Ok(Message::Close(frame))) => {
                    let (code, reason) = frame
                        .map(|f| (u16::from(f.code), f.reason.to_string()))
                        .unwrap_or((1005, String::new()));
                    // Echo the close frame back, then finish.
                    let _ = self.socket.close(frame_for(code, &reason)).await;
                    let _ = events.send(WsInbound::Close(code, reason, true));
                    terminated = true;
                    break;
                }
                Some(Err(error)) => {
                    let _ = events.send(WsInbound::Error(error.to_string()));
                    break;
                }
                None => break,
            }
        }
        // A close frame the peer never acknowledged, a transport error, or a
        // bare TCP drop all surface to the page as an unclean close (1006).
        if !terminated {
            let _ = events.send(WsInbound::Close(1006, String::new(), false));
        }
    }
}

fn frame_for(code: u16, reason: &str) -> Option<CloseFrame> {
    (code != 1005).then(|| CloseFrame {
        code: CloseCode::from(code),
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accept one connection, echo back every text/binary message, echo the
    /// close handshake, then finish.
    async fn echo_server(listener: tokio::net::TcpListener) {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_hdr_async(
            stream,
            |_request: &http::Request<()>, mut response: http::Response<()>| {
                response
                    .headers_mut()
                    .insert(http::header::SEC_WEBSOCKET_PROTOCOL, "echo".parse().unwrap());
                Ok(response)
            },
        )
        .await
        .unwrap();
        while let Some(Ok(message)) = socket.next().await {
            match message {
                Message::Text(_) | Message::Binary(_) => {
                    socket.send(message).await.unwrap();
                }
                Message::Close(frame) => {
                    let _ = socket.send(Message::Close(frame)).await;
                    // Keep reading until the peer finishes the close handshake
                    // and drops the TCP connection: dropping early can turn
                    // into a reset that swallows the echoed close frame.
                    while let Some(Ok(_)) = socket.next().await {}
                    break;
                }
                _ => {}
            }
        }
    }

    fn options<'a>(headers: &'a [(String, String)], protocols: &'a [String]) -> WsOptions<'a> {
        WsOptions { extra_headers: headers, protocols, proxy: None, allow_private: true }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn echo_roundtrip_text_binary_close() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(echo_server(listener));

        let url = Url::parse(&format!("ws://127.0.0.1:{port}/chat?x=1")).unwrap();
        let headers = [("Origin".to_string(), "http://example.test".to_string())];
        let protocols = vec!["echo".to_string()];
        let session = connect(&url, None, &options(&headers, &protocols)).await.unwrap();

        let (events_tx, mut events) = mpsc::unbounded_channel();
        let (write_tx, write_rx) = mpsc::unbounded_channel();
        tokio::spawn(session.run(events_tx, write_rx));

        write_tx.send(WsOutbound::Text("hello".into())).unwrap();
        write_tx.send(WsOutbound::Binary(vec![1, 2, 3])).unwrap();
        write_tx.send(WsOutbound::Close(1000, "done".into())).unwrap();

        let mut events_seen = Vec::new();
        for _ in 0..3 {
            events_seen.push(
                tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
                    .await
                    .expect("timed out waiting for event")
                    .expect("channel closed early"),
            );
        }
        match &events_seen[0] {
            WsInbound::Text(t) => assert_eq!(t, "hello"),
            other => panic!("expected text echo, got {other:?}"),
        }
        match &events_seen[1] {
            WsInbound::Binary(b) => assert_eq!(b, &[1, 2, 3]),
            other => panic!("expected binary echo, got {other:?}"),
        }
        match &events_seen[2] {
            WsInbound::Close(code, reason, clean) => {
                assert_eq!(*code, 1000);
                assert_eq!(reason, "done");
                assert!(*clean);
            }
            other => panic!("expected close, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handshake_failure_reports_rejection() {
        // A plain HTTP responder: accept the connection, answer 403.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            let _ = stream
                .write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                .await;
        });
        let url = Url::parse(&format!("ws://127.0.0.1:{port}/")).unwrap();
        let headers = [];
        let Err(error) = connect(&url, None, &options(&headers, &[])).await else {
            panic!("handshake unexpectedly succeeded");
        };
        assert!(matches!(error, WsError::Rejected(_) | WsError::Handshake(_)), "{error}");
    }

    #[tokio::test]
    async fn ssrf_blocks_private_target_by_default() {
        let url = Url::parse("ws://127.0.0.1:1/").unwrap();
        let headers = [];
        let private = WsOptions { extra_headers: &headers, protocols: &[], proxy: None, allow_private: false };
        // Loopback is forbidden unless --allow-private-network is set; the env
        // flag may already permit it on developer machines, so only assert on
        // the strict result.
        std::env::remove_var("OBSCURA_ALLOW_PRIVATE_NETWORK");
        let Err(error) = connect(&url, None, &private).await else {
            panic!("private-network connect unexpectedly succeeded");
        };
        assert!(matches!(error, WsError::Ssrf(_)), "{error}");
    }
}
