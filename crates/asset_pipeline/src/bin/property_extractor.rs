//! Offline oracle extractor — Step 2.5 light properties.
//!
//! Reads the decompiled release-jar `Blocks.java` (quarantined oracle under
//! `~/.minecraft-decomp/`, never part of the build) and harvests per block:
//!   - light emission classification (constant / lit-conditional / property-driven)
//!   - `.noOcclusion()`  → `canOcclude == false`
//!   - `TransparentBlock` construction → the `propagatesSkylightDown == true` class
//!     override that makes the glass family dampen by 0 rather than 1
//!
//! Output: checked-in `assets/registry/block_properties.json`. The registry baker
//! consumes ONLY this JSON — normal builds and CI stay fully hermetic; this bin is
//! the one non-hermetic step and runs offline by design.
//!
//! Usage:
//!   cargo run -p asset_pipeline --bin property_extractor <path/to/Blocks.java>
//!
//! Every record may carry provenance; emission values trace back to
//! `Blocks.java:<line>` (see SPECS.md §Registry v1).

use serde_json::{json, Value as J};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::Path;

const DYED_COLORS: [&str; 16] = [
    "white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray",
    "cyan", "purple", "blue", "brown", "green", "red", "black",
];

// ---------------------------------------------------------------------------
// text scanning helpers
// ---------------------------------------------------------------------------

fn ident_at(src: &str, start: usize) -> String {
    let b = src.as_bytes();
    let mut end = start;
    while end < b.len() && (b[end].is_ascii_alphanumeric() || b[end] == b'_') {
        end += 1;
    }
    src[start..end].to_string()
}

/// Index of the ')' matching the '(' located at byte `open`.
fn balanced_end(src: &str, open: usize) -> Option<usize> {
    let b = src.as_bytes();
    let mut depth = 0usize;
    for (i, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Content of the first balanced `(...)` in `s`.
fn capture_first_paren(s: &str) -> Option<String> {
    let open = s.find('(')?;
    let close = balanced_end(s, open)?;
    Some(s[open + 1..close].to_string())
}

fn number(s: &str) -> Result<Option<u8>, String> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(None);
    }
    if t.chars().all(|c| c.is_ascii_digit()) {
        return Ok(Some(t.parse::<u8>().map_err(|e| e.to_string())?));
    }
    Ok(None)
}

const REG_MARKER: &str = "\n   public static final ";

fn reg_bodies(src: &str) -> Vec<(usize, String)> {
    let mut spans: Vec<usize> = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = src[cursor..].find(REG_MARKER) {
        let abs = cursor + rel + 1; // statement text begins right after the newline
        spans.push(abs);
        cursor = abs + REG_MARKER.len() - 1;
    }
    let mut out = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        let end = spans.get(i + 1).copied().unwrap_or(src.len());
        out.push((s + 1, src[*s..end].to_string()));
    }
    out
}

/// Block name from the registration FIELD (left side of `=`). Field names track the
/// canonical block ids far more reliably than the id constants (e.g. CAVE_VINES
/// registers under `BlockItemIds.GLOW_BERRY_CROP`).
fn reg_name(body: &str) -> Option<String> {
    let eq = body.find(" = ")?;
    let head = body[..eq].trim_end();
    let name = head.rsplit(' ').next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_lowercase())
    }
}

/// Final registered ids produced by one registration statement: wrapper forms
/// (`ColorCollection` / `WeatheringCopperCollection`) expand to their member ids.
fn reg_ids(body: &str, base: &str) -> Vec<String> {
    if body.contains("ColorCollection.registerBlocks(") {
        let base = base.strip_prefix("dyed_").unwrap_or(base);
        return DYED_COLORS.iter().map(|c| format!("{c}_{base}")).collect();
    }
    if body.contains("WeatheringCopperCollection.registerBlocks(") {
        return vec![
            base.to_string(),
            format!("exposed_{base}"),
            format!("weathered_{base}"),
            format!("oxidized_{base}"),
        ];
    }
    vec![base.to_string()]
}

/// Collect `static Properties <name>(...) { ... }` helper bodies; their call sites
/// are substituted into registration bodies so hidden chains (e.g. `candleProperties`)
/// still expose `lightLevel(...)` / `.noOcclusion()` tokens.
fn collect_helpers(src: &str) -> BTreeMap<String, String> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    let mut cursor = 0usize;
    while cursor < src.len() {
        let from = match src[cursor..].find("static Properties ") {
            Some(r) => cursor + r + "static Properties ".len(),
            None => break,
        };
        let name = ident_at(src, from);
        if name.is_empty() {
            break;
        }
        let Some(paren) = src.find('(').map(|_| from + name.len()).filter(|&p| {
            // first '(' immediately following the helper name
            p < src.len() && src.as_bytes()[p] == b'('
        }) else {
            break;
        };
        let Some(param_close) = balanced_end(src, paren) else { break };
        let scan_end = src.len().min(param_close + 4_000);
        let Some(bopen_rel) = src[param_close..scan_end].find('{') else {
            cursor = param_close;
            continue;
        };
        let bopen = param_close + bopen_rel;
        let mut depth = 0usize;
        let mut bclose = None;
        for (i, &c) in src.as_bytes().iter().enumerate().skip(bopen) {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        bclose = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(bclose) = bclose else { break };
        let inner = src[bopen + 1..bclose].trim();
        let expr = inner
            .strip_prefix("return ")
            .map_or(inner, |r| r.trim())
            .strip_suffix(';')
            .map_or(inner, |r| r.trim());
        out.insert(name, expr.to_string());
        cursor = bclose;
    }
    out
}

fn substitute_body(body: &str, helpers: &BTreeMap<String, String>) -> String {
    let mut body = body.to_string();
    while let Some(at) = body.find("wallVariant(") {
        let open_idx = at + "wallVariant".len();
        let Some(close) = balanced_end(&body, open_idx) else { break };
        body.replace_range(at..close + 1, "");
    }
    for (name, expr) in helpers {
        let call_token = format!("{name}(");
        while let Some(at) = body.find(&call_token) {
            let open = at + name.len();
            let Some(close) = balanced_end(&body, open) else { break };
            body.replace_range(at..close + 1, &format!("({expr})"));
        }
    }
    body
}

// ---------------------------------------------------------------------------
// classification
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum OutRec {
    /// Omit the block from the emission table (baker defaults to constant 0).
    Omit,
    Resolved(J),
    /// `Properties.ofLegacyCopy(PARENT)` — resolve after pass 1.
    Legacy(String),
}

fn classify_reg(body: &str) -> Result<OutRec, Vec<String>> {
    let mut rec = OutRec::Omit;

    if body.contains("ofLegacyCopy(") {
        let at = body.find("ofLegacyCopy(").unwrap() + 13;
        let parent = ident_at(body, at);
        if !parent.is_empty() {
            rec = OutRec::Legacy(parent.to_lowercase());
        }
    }

    if let Some(ll) = body.find("lightLevel(") {
        let open = ll + "lightLevel".len();
        let Some(close) = balanced_end(body, open) else {
            return Err(vec![format!(
                "unbalanced lightLevel(...) near {:?}",
                &body[..body.len().min(160)]
            )]);
        };
        let expr = body[open + 1..close].trim().to_string();
        match classify_expr(&expr) {
            Ok(Some(r)) => {
                rec = OutRec::Resolved(r);
            }
            Ok(None) => {
                return Err(vec![format!(
                    "unresolved lightLevel ({:?}) — map it in property_extractor.rs",
                    expr
                )]);
            }
            Err(e) => {
                return Err(vec![format!("bad lightLevel ({e}): {expr}")]);
            }
        }
    }
    Ok(rec)
}

fn parse_cases(s: &str) -> Result<J, String> {
    let mut arr: Vec<J> = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = s[cursor..].find("case ") {
        let p = cursor + rel + 5;
        let end = s[p..].find(';').map(|r| p + r).unwrap_or(s.len());
        let seg = &s[p..end];
        if let Some(arrow) = seg.find("->") {
            if let (Ok(Some(k)), Ok(Some(v))) = (number(&seg[..arrow]), number(&seg[arrow + 2..])) {
                arr.push(json!({"stage": k, "value": v}));
            }
        }
        cursor = end + 1.min(s.len() - end);
    }
    Ok(J::Array(arr))
}

fn classify_expr(expr: &str) -> Result<Option<J>, String> {
    let e = expr.trim();

    // constant: `state -> N`
    if let Some(rest) = e.strip_prefix("state ->") {
        if let Some(v) = number(rest)? {
            return Ok(Some(json!({"kind": "constant", "value": v})));
        }
    }

    // lit-conditional helper: litBlockEmission(N)
    if e.starts_with("litBlockEmission(") && !e.contains("switch") {
        if let Some(v) = number(&capture_first_paren(e).unwrap_or_default())? {
            return Ok(Some(json!({"kind": "lit", "value": v})));
        }
    }

    // weathering-copper switch (Blocks.java:4538): UNWEATHERED/EXPOSED/WEATHERED/OXIDIZED
    if e.contains("litBlockEmission(switch") {
        return Ok(Some(json!({"kind": "weathering_lit", "cases": parse_cases(e)?})));
    }

    // CandleBlock.java:42 — `LIT ? 3 * CANDLES : 0`
    if e.contains("CandleBlock.LIGHT_EMISSION") {
        return Ok(Some(json!({"kind": "lit_scale", "prop": "lit", "base": 3, "per": 3,
                              "count_prop": "candles"})));
    }

    // SeaPickleBlock — `isDead ? 0 : 3 + 3 * PICKLES` (isDead == !WATERLOGGED)
    if e.contains("SeaPickleBlock.isDead") {
        return Ok(Some(json!({"kind": "alive_scale", "alive_prop": "waterlogged",
                              "add": 3, "per": 3, "count_prop": "pickles"})));
    }

    // RespawnAnchorBlock.java:195 — floor(CHARGE / 4 * N); registration passes N=15
    if e.contains("getScaledChargeLevel") {
        return Ok(Some(json!({"kind": "prop_scale", "prop": "charge", "per": 15, "denom": 4})));
    }

    // LightBlock — emission equals its `level` property (0..15)
    if e.contains("LightBlock.LIGHT_EMISSION") {
        return Ok(Some(json!({"kind": "prop_value", "prop": "level"})));
    }

    // VaultState — VaultState$LightLevel: HALF_LIT=6 (inactive), LIT=12
    if e.contains("VaultState") {
        return Ok(Some(json!({"kind": "prop_map", "prop": "vault_state",
            "map": {"inactive": 6, "active": 12, "unlocking": 12, "ejecting": 12}})));
    }

    // TrialSpawnerState — per-state lightLevel from the enum constructor table
    if e.contains("TrialSpawnerState") {
        return Ok(Some(json!({"kind": "prop_map", "prop": "trial_spawner_state",
            "map": {"inactive": 0, "waiting_for_players": 4, "active": 8,
                    "waiting_for_reward_ejection": 8, "ejecting_reward": 8, "cooldown": 0}})));
    }

    // CaveVines.emission(N) — CaveVines.java:54: `BERRIES ? N : 0`
    if e.contains("CaveVines") {
        let inner = capture_first_paren(e).unwrap_or_default();
        if let Some(v) = number(&inner)? {
            return Ok(Some(json!({"kind": "lit", "prop": "berries", "value": v})));
        }
    }

    // GlowLichenBlock.emission(N) — hasAnyFace-conditional; v1 glow_lichen is a single
    // collapsed Complex row, so bake as a constant (documented approximation)
    if e.contains("GlowLichenBlock") {
        let inner = capture_first_paren(e).unwrap_or_default();
        if let Some(v) = number(&inner)? {
            return Ok(Some(json!({"kind": "constant", "value": v})));
        }
    }

    Ok(None)
}

/// Per-id emission expansion: `weathering_lit` cases resolve each stage to that
/// variant's own lit value (ids are ordered plain / exposed / weathered / oxidized);
/// everything else shares one record.
fn expand_per_id(rec: &OutRec, ids: &[String]) -> Vec<(String, OutRec)> {
    if let OutRec::Resolved(r) = rec {
        if r.get("kind").and_then(|k| k.as_str()) == Some("weathering_lit") {
            let cases = r.get("cases").and_then(|c| c.as_array()).cloned().unwrap_or_default();
            let stage_value = |stage: u8| -> u8 {
                cases
                    .iter()
                    .find(|c| c.get("stage").and_then(|s| s.as_u64()) == Some(stage as u64))
                    .and_then(|c| c.get("value").and_then(|v| v.as_u64()))
                    .and_then(|v| u8::try_from(v).ok())
                    .unwrap_or(0)
            };
            return ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    (
                        id.clone(),
                        OutRec::Resolved(json!({"kind": "lit", "value": stage_value(i as u8 + 1)})),
                    )
                })
                .collect();
        }
    }
    ids.iter().map(|id| (id.clone(), rec.clone())).collect()
}

/// True when the body constructs a block class that (transitively) overrides
/// `propagatesSkylightDown == true`: the TransparentBlock family (glass, grates)
/// and StainedGlassBlock, which extends TransparentBlock. The `Half` prefix
/// collision is excluded explicitly.
fn constructs_transparent(body: &str) -> bool {
    const TOKENS: [&str; 2] = ["TransparentBlock", "StainedGlassBlock"];
    for token in TOKENS {
        let mut cursor = 0usize;
        while let Some(rel) = body[cursor..].find(token) {
            let at = cursor + rel;
            let pre = &body[..at];
            let excluded = match token {
                "TransparentBlock" => pre.ends_with("Half"),
                _ => pre.ends_with("StainedGlassPane"),
            };
            if !excluded {
                return true;
            }
            cursor = at + token.len();
        }
    }
    false
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let Some(src_path) = args.get(1).map(String::as_str) else {
        eprintln!("usage: property_extractor <Blocks.java> [out.json]");
        std::process::exit(2);
    };
    let src = fs::read_to_string(src_path).expect("read Blocks.java");
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "assets/registry/block_properties.json".to_string());

    let helpers = collect_helpers(&src);

    // pass 1 — harvest records (own or legacy) + occlusion/skylight sets
    let mut records: BTreeMap<String, OutRec> = BTreeMap::new();
    let mut no_occlusion: BTreeSet<String> = BTreeSet::new();
    let mut skylight: BTreeSet<String> = BTreeSet::new();
    let mut cull_classes: BTreeMap<String, String> = BTreeMap::new();
    let mut diagnostics: Vec<String> = vec![];

    for (decl_line, raw) in reg_bodies(&src) {
        let Some(base) = reg_name(&raw) else { continue };
        let ids = reg_ids(&raw, &base);
        let body = substitute_body(&raw, &helpers);

        match classify_reg(&body) {
            Ok(rec) => {
                let decl = format!("Blocks.java:{}", decl_line);
                for (id, mut rec) in expand_per_id(&rec, &ids) {
                    if let OutRec::Resolved(ref mut r) = rec {
                        if let J::Object(m) = r {
                            m.entry("decl".to_string())
                                .or_insert_with(|| json!(decl));
                        }
                    }
                    records.insert(id, rec);
                }
            }
            Err(d) => {
                for msg in d {
                    diagnostics.push(format!("Blocks.java:{decl_line}: {msg}"));
                }
            }
        }

        if body.contains(".noOcclusion()") {
            no_occlusion.extend(ids.iter().cloned());
        }
        if constructs_transparent(&body) {
            skylight.extend(ids.iter().cloned());
        }

        let is_leaves = body.contains("LeavesBlock")
            || body.contains("ParticleLeavesBlock")
            || body.contains("leavesProperties(")
            || base.ends_with("_leaves")
            || base == "leaves";
        let is_water = base == "water" || base == "flowing_water";
        let is_lava = base == "lava" || base == "flowing_lava";
        let is_half_transparent = body.contains("TransparentBlock")
            || body.contains("HalfTransparentBlock")
            || body.contains("IceBlock")
            || body.contains("FrostedIceBlock")
            || body.contains("StainedGlassBlock")
            || body.contains("TintedGlassBlock")
            || body.contains("SlimeBlock")
            || body.contains("HoneyBlock");

        if is_leaves {
            for id in &ids {
                cull_classes.insert(id.clone(), "leaves".to_string());
            }
        } else if is_water {
            for id in &ids {
                cull_classes.insert(id.clone(), "water".to_string());
            }
        } else if is_lava {
            for id in &ids {
                cull_classes.insert(id.clone(), "lava".to_string());
            }
        } else if is_half_transparent {
            for id in &ids {
                cull_classes.insert(id.clone(), id.clone());
            }
        }

        // provenance patching happens after legacy resolution below
        let _ = decl_line;
    }

    // pass 2 — resolve legacy-copy chains
    let mut emission: BTreeMap<String, (String, J)> = BTreeMap::new(); // id -> (decl-ish note, rec)
    for (id, rec) in &records {
        match rec {
            OutRec::Omit => {}
            OutRec::Resolved(r) => {
                emission.insert(id.clone(), (String::new(), r.clone()));
            }
            OutRec::Legacy(parent) => {
                let mut seen: BTreeSet<String> = BTreeSet::new();
                let mut cur = parent.clone();
                let origin = id.clone();
                loop {
                    if !seen.insert(cur.clone()) {
                        diagnostics.push(format!("{origin}: legacy-copy cycle at {cur}"));
                        break;
                    }
                    if !cull_classes.contains_key(&origin) {
                        if let Some(cc) = cull_classes.get(&cur).cloned() {
                            cull_classes.insert(origin.clone(), cc);
                        }
                    }
                    match records.get(&cur) {
                        Some(OutRec::Resolved(r)) => {
                            if !matches!(
                                r.get("kind").and_then(|k| k.as_str()),
                                Some("skip") | Some("omit")
                            ) {
                                let mut r = r.clone();
                                if let J::Object(m) = &mut r {
                                    m.insert(
                                        "inherit_from".to_string(),
                                        json!(format!("Blocks.java:{cur}")),
                                    );
                                }
                                emission.insert(origin.clone(), (String::new(), r));
                            }
                            break;
                        }
                        Some(OutRec::Legacy(parent)) => cur = parent.clone(),
                        Some(OutRec::Omit) => break, // parent has no emission → child has none either
                        None => {
                            diagnostics.push(format!("{origin}: legacy-copy parent {cur} not found in Blocks.java"));
                            break;
                        }
                    }
                }
            }
        }
    }

    let emit_map: BTreeMap<String, J> = emission
        .into_iter()
        .map(|(id, (_, r))| (id, r))
        .collect();

    let no_occlusion_n = no_occlusion.len();
    let skylight_n = skylight.len();
    let cull_classes_n = cull_classes.len();
    let json = json!({
        "version": 1,
        "generated_by": "property_extractor (the only non-hermetic pipeline step; baker reads this JSON)",
        "source": {
            "path": src_path,
            "oracle": "release jar 26.3 decompile (Vineflower), quarantined in ~/.minecraft-decomp/"
        },
        "emission": emit_map,
        "no_occlusion": no_occlusion.into_iter().collect::<Vec<_>>(),
        "skylight": skylight.into_iter().collect::<Vec<_>>(),
        "cull_classes": cull_classes,
    });

    if let Some(parent) = Path::new(&out_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&out_path, serde_json::to_string_pretty(&json).unwrap() + "\n")
        .expect("write block_properties.json");

    println!(
        "block_properties.json: {} emission entries, {} noOcclusion, {} skylight, {} cull_classes; {} problems",
        emit_map.len(),
        no_occlusion_n,
        skylight_n,
        cull_classes_n,
        diagnostics.len()
    );
    for w in &diagnostics {
        eprintln!("warning: {w}");
    }
    if !diagnostics.is_empty() {
        std::process::exit(1); // every unresolved expression must be hand-mapped
    }
}
