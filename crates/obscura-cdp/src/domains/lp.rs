use serde_json::{json, Value};

use crate::dispatch::CdpContext;

pub async fn handle(
    method: &str,
    _params: &Value,
    ctx: &mut CdpContext,
    session_id: &Option<String>,
) -> Result<Value, String> {
    match method {
        "getMarkdown" => {
            let page = ctx.get_session_page_mut(session_id).ok_or("No page")?;
            Ok(json!({ "markdown": page.markdown() }))
        }
        _ => Err(format!("Unknown LP method: {}", method)),
    }
}
