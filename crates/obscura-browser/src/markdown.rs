//! DOM → Markdown conversion backed by the vendored `obscura-html2md` state
//! machine. The adapter snapshots `DomTree` into an owned arena because the
//! `Dom` trait borrows strings from `self` while `DomTree` keeps nodes behind
//! a `RefCell`. Hidden subtrees (computed `display: none` or
//! `visibility: hidden|collapse`) are dropped during the snapshot, matching the
//! reader expectation that a markdown dump contains visible content only.

use std::collections::HashMap;

use obscura_dom::tree::{DomTree, NodeData, NodeId};
use obscura_html2md::{Converter, Dom, NodeKind, Options};
use obscura_js::runtime::MarkdownStyle;

/// Convert `dom` to markdown, starting at `<body>` when present.
pub(crate) fn render(dom: &DomTree, styles: &HashMap<NodeId, MarkdownStyle>) -> String {
    let Some(snapshot) = Snapshot::build(dom, styles) else {
        return String::new();
    };
    Converter::new(Options {
        default_code_language: Some("text".to_owned()),
        ..Options::default()
    })
    .convert(&snapshot, snapshot.root)
}

struct SnapNode {
    tag: Option<String>,
    text: Option<String>,
    attrs: Vec<(String, String)>,
    control: Option<String>,
    block: bool,
    boundary: bool,
    first: Option<usize>,
    next: Option<usize>,
}

struct Snapshot {
    nodes: Vec<SnapNode>,
    root: usize,
}

impl Snapshot {
    fn build(dom: &DomTree, styles: &HashMap<NodeId, MarkdownStyle>) -> Option<Self> {
        let root_id = dom
            .query_selector("body")
            .ok()
            .flatten()
            .unwrap_or_else(|| dom.document());
        let mut nodes: Vec<SnapNode> = Vec::new();
        // Emitted child index per parent, for first/next sibling links.
        let mut last_child: HashMap<usize, usize> = HashMap::new();
        let mut stack: Vec<(NodeId, Option<usize>)> = vec![(root_id, None)];
        let mut root = None;
        while let Some((nid, parent)) = stack.pop() {
            let Some(node) = dom.get_node(nid) else {
                continue;
            };
            let style = styles.get(&nid);
            let display = style.map_or("", |s| s.display.as_str());
            let visibility = style.map_or("", |s| s.visibility.as_str());
            if matches!(display, "none") || matches!(visibility, "hidden" | "collapse") {
                continue;
            }
            let idx = nodes.len();
            let (tag, text, attrs) = match &node.data {
                NodeData::Element { name, attrs, .. } => (
                    Some(name.local.to_string()),
                    None,
                    attrs
                        .iter()
                        .map(|a| (a.name.local.to_string(), a.value.clone()))
                        .collect(),
                ),
                NodeData::Text { contents } => (None, Some(contents.clone()), Vec::new()),
                _ => (None, None, Vec::new()),
            };
            let (block, boundary) = match (tag.as_deref(), display) {
                (_, "" | "none" | "contents" | "inline" | "inline flow") => (false, false),
                (Some(tag), display) if display.starts_with("inline") => (
                    false,
                    matches!(tag, "input" | "textarea" | "select" | "button")
                        || display.contains("list-item"),
                ),
                _ => (true, false),
            };
            nodes.push(SnapNode {
                tag,
                text,
                attrs,
                control: dom.form_control_state(nid).and_then(|state| state.value),
                block,
                boundary,
                first: None,
                next: None,
            });
            match parent {
                Some(parent) => match last_child.insert(parent, idx) {
                    Some(previous) => nodes[previous].next = Some(idx),
                    None => nodes[parent].first = Some(idx),
                },
                None => root = Some(idx),
            }
            // Push children reversed so the stack pops them in document order.
            let mut child = node.last_child;
            while let Some(child_id) = child {
                let previous = dom
                    .borrow_node(child_id)
                    .and_then(|n| n.prev_sibling);
                stack.push((child_id, Some(idx)));
                child = previous;
            }
        }
        root.map(|root| Self { nodes, root })
    }
}

impl Dom for Snapshot {
    type NodeId = usize;

    fn node_kind(&self, node: usize) -> NodeKind<'_> {
        let snap = &self.nodes[node];
        if let Some(tag) = &snap.tag {
            NodeKind::Element(tag)
        } else if let Some(text) = &snap.text {
            NodeKind::Text(text)
        } else if self.root == node {
            NodeKind::Document
        } else {
            NodeKind::Other
        }
    }

    fn first_child(&self, node: usize) -> Option<usize> {
        self.nodes[node].first
    }

    fn next_sibling(&self, node: usize) -> Option<usize> {
        self.nodes[node].next
    }

    fn attribute(&self, node: usize, name: &str) -> Option<&str> {
        self.nodes[node]
            .attrs
            .iter()
            .find(|(attr, _)| attr == name)
            .map(|(_, value)| value.as_str())
    }

    fn control_value(&self, node: usize) -> Option<&str> {
        self.nodes[node].control.as_deref()
    }

    fn has_block_layout(&self, node: usize) -> bool {
        self.nodes[node].block
    }

    fn has_text_boundary(&self, node: usize) -> bool {
        self.nodes[node].boundary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obscura_dom::parse_html;

    fn render_html(html: &str) -> String {
        let dom = parse_html(html);
        render(&dom, &HashMap::new())
    }

    #[test]
    fn headings_paragraphs_and_inline() {
        let md = render_html(
            "<html><body><h1>Title</h1><h2>Sub</h2><p>Hello <strong>world</strong> \
             <a href='https://x.test/'>link</a> <em>em</em></p></body></html>",
        );
        assert!(md.contains("# Title"), "missing H1: {md}");
        assert!(md.contains("## Sub"), "missing H2: {md}");
        assert!(md.contains("**world**"), "missing strong: {md}");
        assert!(md.contains("*em*"), "missing em: {md}");
        assert!(md.contains("[link](https://x.test/)"), "missing link: {md}");
    }

    #[test]
    fn lists_and_tables() {
        let md = render_html(
            "<html><body><ul><li>A</li><li>B</li></ul><ol><li>X</li></ol>\
             <table><tr><th>H</th><th>J</th></tr><tr><td>one</td><td>two</td></tr></table></body></html>",
        );
        assert!(md.contains("- A"), "missing unordered item: {md}");
        assert!(md.contains("1. X"), "missing ordered item: {md}");
        assert!(md.contains("| H | J |"), "missing header row: {md}");
        assert!(md.contains("| one | two |"), "missing body row: {md}");
    }

    #[test]
    fn skips_script_and_style() {
        let md = render_html(
            "<html><body><p>Text</p><script>alert(1)</script>\
             <style>body{color:red}</style></body></html>",
        );
        assert!(md.contains("Text"), "missing visible text: {md}");
        assert!(!md.contains("alert"), "leaked script content: {md}");
        assert!(!md.contains("color:red"), "leaked style content: {md}");
    }

    #[test]
    fn deep_dom_does_not_recurse() {
        let mut html = String::from("<html><body>");
        html.push_str(&"<span>".repeat(12000));
        html.push_str("deep leaf");
        html.push_str(&"</span>".repeat(12000));
        html.push_str("</body></html>");
        assert_eq!(render_html(&html), "deep leaf");
    }

    #[test]
    fn deep_tables_do_not_recurse() {
        let mut html = String::from("<html><body>");
        html.push_str(&"<table><tr><td>".repeat(180));
        html.push_str("leaf");
        html.push_str(&"</td></tr></table>".repeat(180));
        html.push_str("</body></html>");
        let md = render_html(&html);
        assert!(md.contains("leaf"), "missing leaf: {md}");
    }

    #[test]
    fn input_value_comes_from_form_state() {
        let dom = parse_html("<html><body><input id='i' value='default'></body></html>");
        let input = dom.query_selector("#i").unwrap().unwrap();
        dom.update_form_control_state(input, |state| state.value = Some("typed".to_owned()));
        let md = render(&dom, &HashMap::new());
        assert!(md.contains("typed"), "missing control value: {md}");
    }

    #[test]
    fn display_none_subtree_is_dropped() {
        let dom = parse_html(
            "<html><body><p id='keep'>shown</p><div id='hide'><p>gone</p></div></body></html>",
        );
        let hidden = dom.query_selector("#hide").unwrap().unwrap();
        let styles = HashMap::from([(
            hidden,
            MarkdownStyle {
                display: "none".to_owned(),
                visibility: String::new(),
            },
        )]);
        let md = render(&dom, &styles);
        assert!(md.contains("shown"), "missing visible text: {md}");
        assert!(!md.contains("gone"), "leaked hidden content: {md}");
    }

    #[test]
    fn stylesheet_block_boundary_splits_inline_elements() {
        let dom = parse_html(
            "<html><body><p><span id='a'>first</span><span>second</span></p></body></html>",
        );
        let blocked = dom.query_selector("#a").unwrap().unwrap();
        let styles = HashMap::from([(
            blocked,
            MarkdownStyle {
                display: "block".to_owned(),
                visibility: String::new(),
            },
        )]);
        let md = render(&dom, &styles);
        assert!(md.contains("first\n"), "missing block boundary: {md}");
    }
}
