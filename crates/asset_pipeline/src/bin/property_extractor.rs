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

fn parse_weathering_copper_prefixes(decomp_dir: &Path) -> Vec<String> {
    let coll_path = decomp_dir.join("WeatheringCopperCollection.java");
    if let Ok(src) = fs::read_to_string(&coll_path) {
        if let Some(at) = src.find("PREFIXES =") {
            let slice = &src[at..src.len().min(at + 500)];
            let mut prefixes = Vec::new();
            let mut in_str = false;
            let mut cur = String::new();
            for c in slice.chars() {
                if c == '"' {
                    if in_str {
                        prefixes.push(cur.clone());
                        cur.clear();
                        in_str = false;
                    } else {
                        in_str = true;
                    }
                } else if in_str {
                    cur.push(c);
                }
                if prefixes.len() == 8 {
                    break;
                }
            }
            if prefixes.len() == 8 {
                return prefixes;
            }
        }
    }
    eprintln!(
        "\n❌ [FATAL ERROR] Required oracle file WeatheringCopperCollection.java not found at {:?} or failed to parse PREFIXES!\n\
         Decompile from release jar via Vineflower into ~/.minecraft-decomp/src/ to continue.\n",
        coll_path
    );
    std::process::exit(1);
}

fn parse_copper_block_special_names(decomp_dir: &Path) -> Vec<String> {
    let ids_path = decomp_dir.join("BlockItemIds.java");
    if let Ok(src) = fs::read_to_string(&ids_path) {
        if let Some(at) = src.find("COPPER_BLOCK_SPECIAL_NAMES") {
            let slice = &src[at..src.len().min(at + 300)];
            let mut names = Vec::new();
            let mut in_str = false;
            let mut cur = String::new();
            for c in slice.chars() {
                if c == '"' {
                    if in_str {
                        names.push(cur.clone());
                        cur.clear();
                        in_str = false;
                    } else {
                        in_str = true;
                    }
                } else if in_str {
                    cur.push(c);
                }
                if names.len() == 4 {
                    break;
                }
            }
            if names.len() == 4 {
                return names;
            }
        }
    }
    eprintln!(
        "\n❌ [FATAL ERROR] Required oracle file BlockItemIds.java not found at {:?} or failed to parse COPPER_BLOCK_SPECIAL_NAMES!\n\
         Decompile from release jar via Vineflower into ~/.minecraft-decomp/src/ to continue.\n",
        ids_path
    );
    std::process::exit(1);
}

/// Detects blocks that are currently classified as `solid` in `render_layers` but whose
/// model textures contain transparent/cutout pixels (alpha < 255).
/// In vanilla 26.3, these blocks are CUTOUT via `byTransparency` on their quad sprites,
/// but their Properties were constructed via helper methods (like `torchProperties()`)
/// or custom Block classes without an inline `.noOcclusion()` call.
/// Flagging them in the coverage report ensures transparency for complex-mesh re-derivation.
fn detect_helper_undershoots(
    registry_blocks: &BTreeSet<String>,
    render_layers: &BTreeMap<String, String>,
) -> Vec<String> {
    let Ok(manifest_str) = fs::read_to_string("assets/textures/texture_manifest.json") else {
        return Vec::new();
    };
    let Ok(manifest_val): Result<J, _> = serde_json::from_str(&manifest_str) else {
        return Vec::new();
    };
    let Some(textures) = manifest_val.get("textures").and_then(|t| t.as_object()) else {
        return Vec::new();
    };
    let Ok(mip0_bytes) = fs::read("assets/textures/block_layers_mip0_16x16.rgba8") else {
        return Vec::new();
    };

    let mut layer_has_alpha: BTreeMap<String, bool> = BTreeMap::new();
    for (name, entry) in textures {
        let Some(base_layer) = entry.get("base_layer").and_then(|l| l.as_u64()) else { continue };
        let offset = (base_layer as usize) * 1024;
        let mut has_alpha = false;
        if offset + 1024 <= mip0_bytes.len() {
            for pixel in mip0_bytes[offset..offset + 1024].chunks_exact(4) {
                if pixel[3] < 255 {
                    has_alpha = true;
                    break;
                }
            }
        }
        layer_has_alpha.insert(name.clone(), has_alpha);
    }

    let mut undershoots = Vec::new();
    for block in registry_blocks {
        let is_solid = render_layers.get(block).map(|s| s.as_str()).unwrap_or("solid") == "solid";
        if !is_solid {
            continue;
        }

        let bs_path = format!("assets/blockstates/{}.json", block);
        let Ok(bs_str) = fs::read_to_string(&bs_path) else { continue };
        let Ok(bs_val): Result<J, _> = serde_json::from_str(&bs_str) else { continue };

        let mut model_names = BTreeSet::new();
        if let Some(variants) = bs_val.get("variants").and_then(|v| v.as_object()) {
            for v in variants.values() {
                if let Some(arr) = v.as_array() {
                    for item in arr {
                        if let Some(m) = item.get("model").and_then(|s| s.as_str()) {
                            model_names.insert(m.replace("minecraft:block/", "").replace("block/", ""));
                        }
                    }
                } else if let Some(m) = v.get("model").and_then(|s| s.as_str()) {
                    model_names.insert(m.replace("minecraft:block/", "").replace("block/", ""));
                }
            }
        }
        if let Some(multipart) = bs_val.get("multipart").and_then(|m| m.as_array()) {
            for part in multipart {
                if let Some(apply) = part.get("apply") {
                    if let Some(arr) = apply.as_array() {
                        for item in arr {
                            if let Some(m) = item.get("model").and_then(|s| s.as_str()) {
                                model_names.insert(m.replace("minecraft:block/", "").replace("block/", ""));
                            }
                        }
                    } else if let Some(m) = apply.get("model").and_then(|s| s.as_str()) {
                        model_names.insert(m.replace("minecraft:block/", "").replace("block/", ""));
                    }
                }
            }
        }

        let mut has_cutout_tex = false;
        for m_name in model_names {
            let mut cur_model = m_name;
            for _ in 0..8 {
                let m_path = format!("assets/models/block/{}.json", cur_model);
                let Ok(m_str) = fs::read_to_string(&m_path) else { break; };
                let Ok(m_val): Result<J, _> = serde_json::from_str(&m_str) else { break; };
                if let Some(tex_obj) = m_val.get("textures").and_then(|t| t.as_object()) {
                    for t_val in tex_obj.values() {
                        let t_str_opt = if let Some(s) = t_val.as_str() {
                            Some(s.to_string())
                        } else if let Some(s) = t_val.get("sprite").and_then(|s| s.as_str()) {
                            Some(s.to_string())
                        } else {
                            None
                        };
                        if let Some(t_str) = t_str_opt {
                            if !t_str.starts_with('#') {
                                let clean_t = t_str.replace("minecraft:block/", "").replace("block/", "");
                                if layer_has_alpha.get(&clean_t).copied().unwrap_or(false) {
                                    has_cutout_tex = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if has_cutout_tex {
                    break;
                }
                if let Some(parent) = m_val.get("parent").and_then(|p| p.as_str()) {
                    cur_model = parent.replace("minecraft:block/", "").replace("block/", "");
                } else {
                    break;
                }
            }
            if has_cutout_tex {
                break;
            }
        }

        if has_cutout_tex {
            undershoots.push(block.clone());
        }
    }

    undershoots
}

/// Block name from the registration FIELD (left side of `=`). Field names track the
/// canonical block ids far more reliably than the id constants (e.g. CAVE_VINES
/// registers under `BlockItemIds.GLOW_BERRY_CROP`).
fn reg_name(body: &str) -> Option<String> {
    let eq = body.find(" = ")?;
    let head = body[..eq].trim_end();
    let name = head.rsplit(' ').next()?.trim();
    if name.is_empty() {
        return None;
    }
    let lower = name.to_lowercase();
    if lower == "potted_azalea" {
        return Some("potted_azalea_bush".to_string());
    }
    if lower == "potted_flowering_azalea" {
        return Some("potted_flowering_azalea_bush".to_string());
    }
    Some(lower)
}

/// Final registered ids produced by one registration statement: wrapper forms
/// (`ColorCollection` / `WeatheringCopperCollection`) expand to their member ids.
fn reg_ids(
    body: &str,
    base: &str,
    copper_prefixes: &[String],
    copper_special: &[String],
) -> Vec<String> {
    if body.contains("ColorCollection.registerBlocks(") || body.contains("ColorCollection.zipMap(") {
        let base = base.strip_prefix("dyed_").unwrap_or(base);
        return DYED_COLORS.iter().map(|c| format!("{c}_{base}")).collect();
    }
    if body.contains("WeatheringCopperCollection.registerBlocks(") {
        return copper_prefixes
            .iter()
            .enumerate()
            .map(|(i, prefix)| {
                let name = if base == "copper_block" {
                    &copper_special[i % copper_special.len()]
                } else {
                    base
                };
                format!("{prefix}{name}")
            })
            .collect();
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

    // VaultState (Blocks.java:236474) — VaultState.java:18-23, VaultState$LightLevel.java: HALF_LIT=6 (inactive), LIT=12
    if e.contains("VaultState") || e.contains("VaultBlock.STATE") {
        return Ok(Some(json!({"kind": "prop_map", "prop": "vault_state",
            "map": {"inactive": 6, "active": 12, "unlocking": 12, "ejecting": 12}})));
    }

    // TrialSpawnerState (Blocks.java:236066) — TrialSpawnerState.java:33-37 constructor table
    if e.contains("TrialSpawnerState") || e.contains("TrialSpawnerBlock.STATE") {
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
                        OutRec::Resolved(json!({"kind": "lit", "value": stage_value(((i % 4) as u8) + 1)})),
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

    let decomp_dir = Path::new(&src_path).parent().unwrap_or_else(|| Path::new("."));
    let copper_prefixes = parse_weathering_copper_prefixes(decomp_dir);
    let copper_special = parse_copper_block_special_names(decomp_dir);

    // pass 1 — harvest records (own or legacy) + occlusion/skylight sets
    let mut all_harvested_ids: BTreeSet<String> = BTreeSet::new();
    let mut records: BTreeMap<String, OutRec> = BTreeMap::new();
    let mut no_occlusion: BTreeSet<String> = BTreeSet::new();
    let mut no_collision: BTreeSet<String> = BTreeSet::new();
    let mut skylight: BTreeSet<String> = BTreeSet::new();
    let mut cull_classes: BTreeMap<String, String> = BTreeMap::new();
    let mut render_layers: BTreeMap<String, String> = BTreeMap::new();
    let mut diagnostics: Vec<String> = vec![];

    for (decl_line, raw) in reg_bodies(&src) {
        let Some(base) = reg_name(&raw) else { continue };
        let ids = reg_ids(&raw, &base, &copper_prefixes, &copper_special);
        all_harvested_ids.extend(ids.iter().cloned());
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
        // Vanilla collision-shape absence (`Properties.noCollision()`) — used by the
        // baker to zero is_solid. Distinct from noOcclusion (render-facing: glass is
        // noOcclusion but MUST still collide). Blocks.java decompile is the oracle;
        // e.g. leaf_litter (Blocks.java:4676-4680) is noCollision(), not a cross.
        if body.contains(".noCollision()") {
            no_collision.extend(ids.iter().cloned());
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

        // Render layer harvest:
        // - leaves -> cutout_mipped (LAYER_CUTOUT_MIPPED = 1)
        // - translucent (stained glass, stained glass panes, ice, frosted ice, tinted glass, slime, honey, water) -> translucent (LAYER_TRANSLUCENT = 3)
        // - cutout (copper grates, glass, glass panes, doors, trapdoors, other noOcclusion) -> cutout (LAYER_CUTOUT = 2)
        // - default -> solid (LAYER_SOLID = 0)
        let is_translucent = is_water
            || base == "bubble_column"
            || body.contains("StainedGlassBlock")
            || body.contains("StainedGlassPaneBlock")
            || body.contains("IceBlock")
            || body.contains("FrostedIceBlock")
            || body.contains("TintedGlassBlock")
            || body.contains("SlimeBlock")
            || body.contains("HoneyBlock");

        if is_leaves {
            for id in &ids {
                render_layers.insert(id.clone(), "cutout_mipped".to_string());
            }
        } else if is_translucent {
            for id in &ids {
                render_layers.insert(id.clone(), "translucent".to_string());
            }
        } else if body.contains(".noOcclusion()") {
            for id in &ids {
                render_layers.insert(id.clone(), "cutout".to_string());
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
                    if !render_layers.contains_key(&origin) {
                        if let Some(rl) = render_layers.get(&cur).cloned() {
                            render_layers.insert(origin.clone(), rl);
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
    let no_collision_n = no_collision.len();
    let skylight_n = skylight.len();
    let cull_classes_n = cull_classes.len();
    let render_layers_n = render_layers.len();

    // Coverage report: scan registry blockstates from assets/blockstates
    let blockstates_dir = Path::new("assets/blockstates");
    let mut registry_blocks: BTreeSet<String> = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(blockstates_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    registry_blocks.insert(stem.to_string());
                }
            }
        }
    }

    let mut count_cutout_mipped = 0usize;
    let mut count_cutout = 0usize;
    let mut count_translucent = 0usize;
    let mut count_solid = 0usize;

    for b in &registry_blocks {
        match render_layers.get(b).map(|s| s.as_str()) {
            Some("cutout_mipped") => count_cutout_mipped += 1,
            Some("cutout") => count_cutout += 1,
            Some("translucent") => count_translucent += 1,
            _ => count_solid += 1,
        }
    }

    let mut covered_blocks: BTreeSet<String> = BTreeSet::new();
    covered_blocks.extend(emit_map.keys().cloned());
    covered_blocks.extend(no_occlusion.iter().cloned());
    covered_blocks.extend(no_collision.iter().cloned());
    covered_blocks.extend(skylight.iter().cloned());
    covered_blocks.extend(cull_classes.keys().cloned());
    covered_blocks.extend(render_layers.keys().cloned());

    let unharvested_from_blocks_java: Vec<String> = registry_blocks
        .iter()
        .filter(|b| !all_harvested_ids.contains(*b))
        .cloned()
        .collect();

    let absent_from_all_maps: Vec<String> = registry_blocks
        .iter()
        .filter(|b| !covered_blocks.contains(*b))
        .cloned()
        .collect();

    println!("=================== EXTRACTOR COVERAGE REPORT ===================");
    println!("Registry blocks scanned from assets/blockstates: {}", registry_blocks.len());
    println!("Blocks harvested from Blocks.java: {}", all_harvested_ids.len());
    println!("Blocks with non-default properties (covered by at least 1 map): {}", covered_blocks.len());
    println!("Blocks absent from Blocks.java harvest ({}): {:?}", unharvested_from_blocks_java.len(), unharvested_from_blocks_java);
    println!("Blocks with standard opaque defaults (absent from all property maps): {}", absent_from_all_maps.len());
    println!(
        "Render layers breakdown (of {} registry blocks): {} solid, {} cutout_mipped, {} cutout, {} translucent",
        registry_blocks.len(),
        count_solid,
        count_cutout_mipped,
        count_cutout,
        count_translucent
    );
    let flagged_helper_undershoots = detect_helper_undershoots(&registry_blocks, &render_layers);
    println!(
        "Helper-class undershoots flagged (solid in registry but model has cutout/alpha textures): {} blocks",
        flagged_helper_undershoots.len()
    );
    println!("=================================================================");

    let json = json!({
        "version": 1,
        "generated_by": "property_extractor (the only non-hermetic pipeline step; baker reads this JSON)",
        "source": {
            "path": src_path,
            "oracle": "release jar 26.3 decompile (Vineflower), quarantined in ~/.minecraft-decomp/"
        },
        "emission": emit_map,
        "no_occlusion": no_occlusion.into_iter().collect::<Vec<_>>(),
        "no_collision": no_collision.into_iter().collect::<Vec<_>>(),
        "skylight": skylight.into_iter().collect::<Vec<_>>(),
        "cull_classes": cull_classes,
        "render_layers": render_layers,
        "coverage_report": {
            "total_registry_blocks": registry_blocks.len(),
            "harvested_blocks": all_harvested_ids.len(),
            "unharvested_blocks": unharvested_from_blocks_java,
            "blocks_with_custom_properties": covered_blocks.len(),
            "blocks_with_opaque_defaults": absent_from_all_maps.len(),
            "render_layers_breakdown": {
                "solid": count_solid,
                "cutout_mipped": count_cutout_mipped,
                "cutout": count_cutout,
                "translucent": count_translucent,
                "flagged_helper_undershoots_count": flagged_helper_undershoots.len(),
                "flagged_helper_undershoots": flagged_helper_undershoots,
            },
        },
    });

    if let Some(parent) = Path::new(&out_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&out_path, serde_json::to_string_pretty(&json).unwrap() + "\n")
        .expect("write block_properties.json");

    println!(
        "block_properties.json: {} emission entries, {} noOcclusion, {} noCollision, {} skylight, {} cull_classes, {} non-solid render_layers; {} problems",
        emit_map.len(),
        no_occlusion_n,
        no_collision_n,
        skylight_n,
        cull_classes_n,
        render_layers_n,
        diagnostics.len()
    );
    for w in &diagnostics {
        eprintln!("warning: {w}");
    }
    if !diagnostics.is_empty() {
        std::process::exit(1); // every unresolved expression must be hand-mapped
    }
}
