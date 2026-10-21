//! Offline Block Registry Baker — Step 2 (`assets/registry/block_registry.bin`)
//!
//! Parses `assets/blockstates/*.json` + `assets/models/block/*.json` at build time and
//! compiles a flat, zero-allocation binary catalog for `crates/engine`. The engine never
//! parses model JSON; it memory-loads one array of 32-byte `BlockStateEntry` rows plus a
//! block index.
//!
//! Provenance rules:
//! - Dynamic tint lists derive from `BlockColors::createDefault()` @ 26.3-snapshot-10
//!   (quarantined oracle `~/.minecraft-decomp/`), constants re-verified on 2026-10-22:
//!   birch 0x80A755, spruce 0x619961, lily-pad in-world 0x208030.
//! - Mesh-type classification is structural (resolved model elements + parent family).
//! - `is_opaque` / `is_solid` / light emission / light filter derive from the checked-in
//!   oracle table `assets/registry/block_properties.json` (extracted offline from the
//!   release-jar `Blocks.java` decompile by `property_extractor.rs`). Emission is
//!   per-state: LIT-conditioned emitters (`litBlockEmission` λ in Blocks.java) bake
//!   exact per-variant rows; single-row Complex/Empty blocks collapse to always-lit
//!   (documented v1 approximation). Dampening derives from the release-jar default
//!   formula + class overrides; `is_opaque` now means canOcclude. See SPECS.md.
//! - v1 approximation: canonical-cube rows bake only the FIRST resolved element
//!   (grass_block side overlay is a second element and is dropped — documented).
//! - v1 approximation: `uv_rot` is the model face's own `rotation` field; blockstate
//!   rotation *twist* (uvlock re-alignment) is refined in Step 3 golden tests.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Face / state layout (mirrors SPECS.md §BlockStateEntry)
// ---------------------------------------------------------------------------

pub const FACE_UP: usize = 0; // +Y
pub const FACE_DOWN: usize = 1; // -Y
pub const FACE_SOUTH: usize = 2; // +Z
pub const FACE_NORTH: usize = 3; // -Z
pub const FACE_EAST: usize = 4; // +X
pub const FACE_WEST: usize = 5; // -X

pub const TINT_NONE: u8 = 0;
pub const TINT_GRASS: u8 = 1;
pub const TINT_FOLIAGE: u8 = 2;
pub const TINT_WATER: u8 = 3;
/// Fourth dynamic colormap (`dry_foliage`). Registry rows can hold values > 3; the
/// mesher encodes it into Word 1 spare bit 0 when packing the vertex (see SPECS.md).
pub const TINT_DRY_FOLIAGE: u8 = 4;

pub const MESH_EMPTY: u8 = 0;
pub const MESH_CUBE: u8 = 1;
pub const MESH_CROSS: u8 = 2;
pub const MESH_FLUID: u8 = 3;
pub const MESH_COMPLEX: u8 = 4;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FaceInfo {
    pub base_layer: u16, // texture array layer 0..1842
    pub tint_mode: u8,   // TINT_*
    pub uv_rot: u8,      // 0..3 (90° steps)
} // 4 bytes

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockStateEntry {
    pub faces: [FaceInfo; 6], // 24 B, order: UP, DOWN, SOUTH, NORTH, EAST, WEST
    pub mesh_type: u8,
    pub is_opaque: u8,
    pub is_solid: u8,
    pub light_emission: u8, // 0..15 block light emitted
    pub light_filter: u8,   // 0..15 light attenuation per voxel
    pub pad: [u8; 3],
} // 32 bytes, 2 rows per 64B cache line

impl BlockStateEntry {
    pub fn write_le(&self, out: &mut Vec<u8>) {
        for f in &self.faces {
            out.extend_from_slice(&f.base_layer.to_le_bytes());
            out.push(f.tint_mode);
            out.push(f.uv_rot);
        }
        out.push(self.mesh_type);
        out.push(self.is_opaque);
        out.push(self.is_solid);
        out.push(self.light_emission);
        out.push(self.light_filter);
        out.extend_from_slice(&self.pad);
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockIndexEntry {
    pub name_off: u32,
    pub name_len: u32,
    pub state_start: u32,
    pub state_count: u32,
    pub mesh_type: u8, // classification of the default variant
    pub has_variants: u8,
    pub pad: u16,
} // 24 bytes

impl BlockIndexEntry {
    pub fn write_le(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.name_off.to_le_bytes());
        out.extend_from_slice(&self.name_len.to_le_bytes());
        out.extend_from_slice(&self.state_start.to_le_bytes());
        out.extend_from_slice(&self.state_count.to_le_bytes());
        out.push(self.mesh_type);
        out.push(self.has_variants);
        out.extend_from_slice(&self.pad.to_le_bytes());
    }
}

pub const HEADER_BYTES: usize = 32;
/// Header is `magic + 7 u32 words`: magic, version, block_count, state_count,
/// names_len, reserved, reserved. (46 bytes of stable engine ABI, padded to 32.)

// ---------------------------------------------------------------------------
// Model store & resolution
// ---------------------------------------------------------------------------

const CUBE_FAMILY: &[&str] = &[
    "cube",
    "cube_all",
    "cube_column",
    "cube_column_horizontal",
    "cube_bottom_top",
    "cube_directional",
    "orientable",
    "orientable_with_bottom",
    "leaves",
    "tinted_leaves",
];

const CROSS_FAMILY: &[&str] = &[
    "cross",
    "tinted_cross",
    "tinted_cross_emissive",
    "flower_pot_cross",
    "tinted_flower_pot_cross",
];

struct Models {
    by_name: BTreeMap<String, Value>,
}

#[derive(Clone, Debug)]
struct ResolvedModel {
    /// First ancestor (child→root order) that declares `elements`.
    elements: Vec<Value>,
    /// Merged texture-variable map, child overrides parent (leaf values may still be refs).
    textures: BTreeMap<String, String>,
    /// Parent chain names (bottom-up: root ancestor last).
    chain: Vec<String>,
}

fn strip_ref(s: &str) -> &str {
    let s = s.strip_prefix("minecraft:").unwrap_or(s);
    s.strip_prefix("block/").unwrap_or(s)
}

impl Models {
    fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut by_name = BTreeMap::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("bad model file name")?
                .to_string();
            let text = std::fs::read_to_string(&path)?;
            let json: Value = serde_json::from_str(&text)?;
            by_name.insert(name, json);
        }
        Ok(Self { by_name })
    }

    fn resolve(&self, start: &str) -> Option<ResolvedModel> {
        let mut cur = start.to_string();
        let mut textures = BTreeMap::new();
        let mut elements: Option<Vec<Value>> = None;
        let mut chain: Vec<String> = Vec::new();

        for _ in 0..16 {
            let json = self.by_name.get(strip_ref(&cur))?.clone();
            if let Some(Value::Object(obj)) = Some(&json) {
                if let Some(Value::Object(tex)) = obj.get("textures") {
                    for (k, v) in tex {
                        match v {
                            Value::String(s) => {
                                textures.insert(k.clone(), s.to_string());
                            }
                            // Quirk 1 (26.3): stained glass etc. use
                            // `{"force_translucent": true, "sprite": "..."}` refs.
                            Value::Object(o) => {
                                if let Some(sp) = o.get("sprite").and_then(|s| s.as_str()) {
                                    textures.insert(k.clone(), sp.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if elements.is_none() {
                    if let Some(Value::Array(els)) = obj.get("elements") {
                        elements = Some(els.clone());
                    }
                }
                chain.push(strip_ref(&cur).to_string());
                match obj.get("parent").and_then(|p| p.as_str()) {
                    Some(p) => cur = strip_ref(p).to_string(),
                    None => break,
                }
            } else {
                return None;
            }
        }
        Some(ResolvedModel {
            elements: elements.unwrap_or_default(),
            textures,
            chain,
        })
    }

    /// Follow `#var` chains (or the bare-name quirk used by e.g. `heavy_core.json`)
    /// down to an absolute texture name, stripping namespace prefixes.
    fn deref_texture(&self, tex: &Value, vars: &BTreeMap<String, String>) -> Option<String> {
        let mut cur = match tex {
            Value::String(s) => s.clone(),
            Value::Object(o) => o.get("sprite")?.as_str()?.to_string(),
            _ => return None,
        };
        for _ in 0..10 {
            if let Some(rest) = cur.strip_prefix('#') {
                cur = vars.get(rest)?.clone();
            } else if !cur.contains('/') {
                if let Some(v) = vars.get(&cur) {
                    cur = v.clone();
                } else {
                    return Some(strip_ref(&cur).to_string());
                }
            } else {
                return Some(strip_ref(&cur).to_string());
            }
        }
        None
    }
}

// Face data extracted from a resolved element's face map, keyed by cullface.
struct FaceData {
    texture: Value,
    rotation: i32,
}

fn face_map_of_element(elem: &Value) -> BTreeMap<String, FaceData> {
    let mut out = BTreeMap::new();
    if let Some(Value::Object(faces)) = elem.get("faces") {
        for (fname, fval) in faces {
            // Cullface when present; otherwise the face KEY is the direction
            // (full-height models like piston_base's inset north face omit it).
            let dir = fval
                .get("cullface")
                .and_then(|c| c.as_str())
                .unwrap_or(fname.as_str());
            let rot = fval.get("rotation").and_then(|r| r.as_i64()).unwrap_or(0) as i32;
            out.insert(
                dir.to_string(),
                FaceData {
                    texture: fval.get("texture").cloned().unwrap_or(Value::Null),
                    rotation: rot,
                },
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Rotation math (validated: oak_log axis=x ends on ±X, axis=z ends on ±Z)
// ---------------------------------------------------------------------------

const DIR_VEC: [[i32; 3]; 6] = [
    [0, 1, 0],  // UP
    [0, -1, 0], // DOWN
    [0, 0, 1],  // SOUTH
    [0, 0, -1], // NORTH
    [1, 0, 0],  // EAST
    [-1, 0, 0], // WEST
];

const FACE_NAMES: [&str; 6] = ["up", "down", "south", "north", "east", "west"];

fn rot_x(v: [i32; 3], steps: i32) -> [i32; 3] {
    let mut v = v;
    for _ in 0..(steps.rem_euclid(4)) {
        v = [v[0], -v[2], v[1]]; // Rx90: +Y→+Z, +Z→-Y
    }
    v
}

fn rot_y(v: [i32; 3], steps: i32) -> [i32; 3] {
    let mut v = v;
    for _ in 0..(steps.rem_euclid(4)) {
        v = [v[2], v[1], -v[0]]; // Ry90: +Z→+X, +X→-Z
    }
    v
}

fn dir_index(v: [i32; 3]) -> Option<usize> {
    DIR_VEC.iter().position(|d| *d == v)
}

/// Blockstate rotation order: Rx(x) applied first, then Ry(y).
/// `x` and `y` are in DEGREES (90/180/270); convert to quarter-turns.
/// Returns model-index for each world-index.
fn inverse_face_map(x_deg: i32, y_deg: i32) -> [usize; 6] {
    let (x, y) = (x_deg / 90, y_deg / 90);
    let mut table = [usize::MAX; 6];
    for (model_idx, dir) in DIR_VEC.iter().enumerate() {
        let world = rot_y(rot_x(*dir, x), y);
        if let Some(w) = dir_index(world) {
            table[w] = model_idx;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Tint sources (provenance: BlockColors::createDefault, see module doc)
// ---------------------------------------------------------------------------

const GRASS_TINT_BLOCKS: &[&str] = &[
    "fern",
    "short_grass",
    "potted_fern",
    "bush",
    "tall_grass",
    "large_fern",
    "grass_block",
    "sugar_cane", // PENDING-VERIFY: sugarCane() may not be pure grass palette
];

const FOLIAGE_TINT_BLOCKS: &[&str] = &[
    "oak_leaves",
    "jungle_leaves",
    "acacia_leaves",
    "dark_oak_leaves",
    "vine",
    "mangrove_leaves",
];

const WATER_TINT_BLOCKS: &[&str] = &["water", "bubble_column", "water_cauldron"];

const DRY_FOLIAGE_TINT_BLOCKS: &[&str] = &["leaf_litter"];

/// Blocks whose first tint layer is BLANK (untinted): petals stay natural color.
const BLANK_FIRST_BLOCKS: &[&str] = &["pink_petals", "wildflowers"];

#[derive(Debug, Clone, Copy)]
enum TintDecision {
    Mode(u8),
    Swap(u32), // replace texture layer with pre-tinted layer
}

fn tint_decision(block: &str) -> TintDecision {
    // Fixed tints bake to dedicated pre-tinted layers (tint_mode = 0).
    match block {
        "birch_leaves" => return TintDecision::Swap(1840),
        "spruce_leaves" => return TintDecision::Swap(1841),
        "lily_pad" => return TintDecision::Swap(1842),
        _ => {}
    }
    if GRASS_TINT_BLOCKS.contains(&block) {
        return TintDecision::Mode(TINT_GRASS);
    }
    if FOLIAGE_TINT_BLOCKS.contains(&block) {
        return TintDecision::Mode(TINT_FOLIAGE);
    }
    if WATER_TINT_BLOCKS.contains(&block) {
        return TintDecision::Mode(TINT_WATER);
    }
    if DRY_FOLIAGE_TINT_BLOCKS.contains(&block) {
        return TintDecision::Mode(TINT_DRY_FOLIAGE);
    }
    TintDecision::Mode(TINT_NONE) // incl. BLANK_FIRST_BLOCKS (petals): no tint in v1
}

// ---------------------------------------------------------------------------
// Verified block properties (light emission / dampening / occlusion)
// ---------------------------------------------------------------------------
// Values live in `assets/registry/block_properties.json`, machine-extracted from
// the decompiled release-jar `Blocks.java` by `property_extractor.rs` (the one
// non-hermetic step; run it offline). This baker reads the JSON only.
//
// Derivation provenance (release jar, 26.3):
// - `BlockBehaviour.getLightDampening` default (BlockBehaviour.java:274-280):
//     solidRender ? 15 : propagatesSkylightDown ? 0 : 1
//   where solidRender == canOcclude && occlusion shape is a full block
//   (BlockBehaviour$BlockStateBase.java:161-162).
// - Class overrides: LeavesBlock.java:80 (=1), TintedGlassBlock.java:15-17 (=15),
//   TransparentBlock.java:30-32 / WaterloggedTransparentBlock / StainedGlassBlock
//   extend it (propagatesSkylightDown => 0). `IceBlock`/`HalfTransparentBlock`
//   have NO overrides: ice/frosted_ice ride the default path → 1, while
//   packed_ice/blue_ice (occluding full prisms) hit solidRender → 15.
// - Fluids: `LiquidBlock` (Blocks.java:246) has no dampening override and both
//   water and lava register `.liquid()` (canOcclude=false) with
//   `LiquidBlock.propagatesSkylightDown=false` (LiquidBlock.java:94) → 1 each.
// - The engine-side `max(1, dampening)` step is `LightEngine.getOpacity`
//   (LightEngine.java:79-81); the 16-gate of `getLightDampeningInto`
//   (LightEngine.java:52-62) only engages `useShapeForLightOcclusion` blocks
//   (stairs/panes — the complex-skip set) and is noted for Step 2.5.

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Emission {
    Constant { value: u8, #[serde(default)] decl: String },
    Lit { value: u8, #[serde(default)] prop: String, #[serde(default)] decl: String },
    LitScale {
        base: u8,
        per: u8,
        #[serde(default)]
        prop: String,
        count_prop: String,
        #[serde(default)]
        decl: String,
    },
    AliveScale {
        alive_prop: String,
        add: u8,
        per: u8,
        count_prop: String,
        #[serde(default)]
        decl: String,
    },
    PropValue { prop: String, #[serde(default)] decl: String },
    PropScale { prop: String, per: u8, denom: u8, #[serde(default)] decl: String },
    PropMap { prop: String, map: BTreeMap<String, u8>, #[serde(default)] decl: String },
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct BlockProps {
    #[serde(default)]
    pub emission: BTreeMap<String, Emission>,
    #[serde(default)]
    pub no_occlusion: BTreeSet<String>,
    #[serde(default)]
    pub skylight: BTreeSet<String>,
}

impl BlockProps {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    fn is_fluid(block: &str) -> bool {
        let b = block.strip_prefix("flowing_").unwrap_or(block);
        matches!(b, "water" | "lava" | "bubble_column")
    }

    fn val_u8(vals: &BTreeMap<String, String>, key: &str) -> u8 {
        vals.get(key).and_then(|v| v.parse::<u8>().ok()).unwrap_or(0)
    }

    fn is_true(vals: &BTreeMap<String, String>, key: &str) -> bool {
        vals.get(key).map(|v| v == "true").unwrap_or(false)
    }

    /// Per-state block-light emission. `collapsed = true` bakes the always-lit
    /// representative value for single-row (Complex/Empty) approximations, as
    /// documented in SPECS.md §Registry v1.
    pub fn emission_row(&self, block: &str, vals: &BTreeMap<String, String>, collapsed: bool) -> u8 {
        let rec = self
            .emission
            .get(block)
            .or_else(|| self.emission.get(block.strip_prefix("flowing_").unwrap_or(block)));
        let Some(rec) = rec else { return 0 };
        match rec {
            Emission::Constant { value, .. } => *value,
            Emission::Lit { value, prop, .. } => {
                if collapsed {
                    return *value;
                }
                let key = if prop.is_empty() { "lit" } else { prop };
                if Self::is_true(vals, key) { *value } else { 0 }
            }
            Emission::LitScale { base, per, prop, count_prop, .. } => {
                let count = Self::val_u8(vals, count_prop).max(1);
                if collapsed {
                    return *base;
                }
                let key = if prop.is_empty() { "lit" } else { prop };
                if Self::is_true(vals, key) {
                    base.saturating_add(per.saturating_mul(count - 1))
                } else {
                    0
                }
            }
            Emission::AliveScale { alive_prop, add, per, count_prop, .. } => {
                let count = Self::val_u8(vals, count_prop).max(1);
                if collapsed {
                    return add.saturating_add(per.saturating_mul(1));
                }
                if Self::is_true(vals, alive_prop) {
                    add.saturating_add(per.saturating_mul(count))
                } else {
                    0
                }
            }
            Emission::PropValue { prop, .. } => {
                if collapsed { 0 } else { Self::val_u8(vals, prop) }
            }
            Emission::PropScale { prop, per, denom, .. } => {
                if collapsed {
                    0
                } else {
                    let v = Self::val_u8(vals, prop) as u32;
                    ((v * *per as u32) / (*denom as u32)) as u8
                }
            }
            Emission::PropMap { prop, map, .. } => {
                if collapsed {
                    map.values().copied().max().unwrap_or(0)
                } else {
                    map.get(vals.get(prop).map(String::as_str).unwrap_or("")).copied().unwrap_or(0)
                }
            }
        }
    }

    /// Light dampening (attenuation per voxel) via the release-jar derivation.
    pub fn light_filter(&self, block: &str, mesh_type: u8) -> u8 {
        if block == "tinted_glass" {
            return 15; // TintedGlassBlock.java:15-17 explicit override
        }
        if Self::is_fluid(block) {
            return 1; // LiquidBlock.java relationships above
        }
        let full = mesh_type == MESH_CUBE;
        let occluding = !self.no_occlusion.contains(block);
        if full && occluding {
            return 15; // solidRender (BlockBehaviour$BlockStateBase.java:161-162)
        }
        if !full || self.skylight.contains(block) {
            return 0; // shape does not stop skylight (torch/pane/glass/grate or empty)
        }
        1 // full shape, non-occluding: leaves, ice family, slime/honey
    }

    /// `is_opaque` now means canOcclude (guesstimated as a full prism without
    /// `noOcclusion`), matching how dampening derives it. Mesher culling reads this.
    pub fn is_opaque(&self, block: &str, mesh_type: u8) -> u8 {
        u8::from(mesh_type == MESH_CUBE && !self.no_occlusion.contains(block))
    }
}

fn is_solid(mesh_type: u8) -> u8 {
    match mesh_type {
        MESH_EMPTY => 0,
        MESH_CROSS | MESH_FLUID => 0,
        _ => 1,
    }
}

// ---------------------------------------------------------------------------
// Variant key expansion
// ---------------------------------------------------------------------------

/// Parse one variant key into ordered (prop, value) assignments.
/// Tokens without '=' continue the previous assignment's value list (OR-group).
fn parse_key(key: &str) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for tok in key.split(',') {
        let tok = tok.trim();
        if let Some((k, v)) = tok.split_once('=') {
            out.push((k.trim().to_string(), vec![v.trim().to_string()]));
        } else if let Some(last) = out.last_mut() {
            last.1.push(tok.to_string());
        }
    }
    out
}

fn canonical_combo(assign: &[(String, String)]) -> String {
    let sorted: BTreeMap<&str, &str> = assign
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    sorted
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// Bake
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct Baked {
    /// sorted by block name
    pub blocks: BTreeMap<String, BakedBlock>,
    pub warnings: Vec<String>,
    pub cube_states: usize,
    pub cross_states: usize,
    pub fluid_states: usize,
    pub complex_states: usize,
    pub empty_states: usize,
}

#[derive(Debug)]
pub struct BakedBlock {
    pub mesh_type: u8,
    pub states: Vec<BlockStateEntry>,
}

pub fn bake(
    blockstates_dir: &Path,
    models_dir: &Path,
    manifest_path: &Path,
    props: &BlockProps,
) -> Result<Baked, Box<dyn std::error::Error>> {
    let models = Models::load(models_dir)?;
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(manifest_path)?)?;
    let mut layers: BTreeMap<String, u32> = BTreeMap::new();
    if let Some(Value::Object(tex)) = manifest.get("textures") {
        for (k, v) in tex {
            if let Some(b) = v.get("base_layer").and_then(|b| b.as_u64()) {
                layers.insert(k.clone(), b as u32);
            }
        }
    }
    let layer_of = |name: &str| -> Option<u16> { layers.get(name).map(|l| *l as u16) };

    let mut blocks: BTreeMap<String, BakedBlock> = BTreeMap::new();
    let mut warnings: Vec<String> = Vec::new();

    let mut state_files: Vec<(String, Value)> = Vec::new();
    for entry in std::fs::read_dir(blockstates_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let json: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        state_files.push((name, json));
    }
    state_files.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, bs) in state_files {
        let default_ref = if let Some(variants) = bs.get("variants") {
            variants
                .as_object()
                .and_then(|v| v.get("").or_else(|| v.values().next()))
                .map(variant_obj)
                .and_then(|e| e.get("model").and_then(|m| m.as_str()))
                .map(|s| s.to_string())
        } else {
            None
        };

        let kind = if matches!(
            name.as_str(),
            "water" | "lava" | "flowing_water" | "flowing_lava" | "bubble_column"
        ) {
            MESH_FLUID
        } else if let Some(model_ref) = &default_ref {
            match models.resolve(model_ref) {
                Some(rm) => {
                    let family_hit = |list: &[&str]| {
                        rm.chain
                            .iter()
                            .any(|c| list.contains(&c.as_str()))
                    };
                    if family_hit(CROSS_FAMILY) || rm.chain.iter().any(|c| c.contains("cross")) {
                        MESH_CROSS
                    } else if family_hit(CUBE_FAMILY)
                        || canonical_full_cube_elements(&rm.elements)
                    {
                        MESH_CUBE
                    } else if rm.elements.is_empty() {
                        MESH_EMPTY
                    } else {
                        MESH_COMPLEX
                    }
                }
                None => {
                    warnings.push(format!("{}: unresolved default model {:?}", name, model_ref));
                    MESH_COMPLEX
                }
            }
        } else {
            MESH_EMPTY // no variants/multipart at all (air family)
        };

        let is_multipart = bs.get("multipart").is_some();
        let mesh_type = if is_multipart { MESH_COMPLEX } else { kind };

        let mut states: Vec<BlockStateEntry> = Vec::new();

        if mesh_type == MESH_CUBE || mesh_type == MESH_CROSS || mesh_type == MESH_FLUID {
            if mesh_type == MESH_FLUID {
                // Fluids: all faces carry the still-fluid texture; shader animates via
                // the timeline table keyed on this base layer.
                let tex = match layer_of(if name == "lava" || name == "flowing_lava" {
                    "lava_still"
                } else {
                    "water_still"
                }) {
                    Some(l) => l,
                    None => return Err(format!("fluid texture missing for {}", name).into()),
                };
                let tint = if name.starts_with("water") { TINT_WATER } else { TINT_NONE };
                let mut entry = BlockStateEntry::default();
                entry.faces = [FaceInfo { base_layer: tex, tint_mode: tint, uv_rot: 0 }; 6];
                entry.mesh_type = MESH_FLUID;
                entry.is_opaque = 0;
                entry.is_solid = 0;
                let empty = BTreeMap::new();
                entry.light_emission = props.emission_row(&name, &empty, false);
                entry.light_filter = props.light_filter(&name, MESH_FLUID);
                states.push(entry);
            } else if let Some(Value::Object(variants)) = bs.get("variants") {
                // Expand property combinations for mesh-relevant states.
                let mut combo_map: BTreeMap<String, VariantValue> = BTreeMap::new();
                let mut variant_props: Vec<(String, Vec<String>)> = Vec::new();
                for (k, v) in variants {
                    let assigns = parse_key(k);
                    // register canonical combos per assignment-set
                    for set in expand_assignments(&assigns) {
                        let canon = canonical_combo(&set);
                        let vv = VariantValue::from_entry(v);
                        combo_map.insert(canon, vv.clone());
                    }
                    for (pk, mut vals) in assigns {
                        if let Some(slot) = variant_props.iter_mut().find(|(k, _)| *k == pk) {
                            for val in vals.drain(..) {
                                if !slot.1.contains(&val) {
                                    slot.1.push(val);
                                }
                            }
                        } else {
                            variant_props.push((pk, vals));
                        }
                    }
                }

                let count = variant_props
                    .iter()
                    .map(|(_, v)| v.len())
                    .product::<usize>()
                    .max(1);
                if count > 100_000 {
                    return Err(format!("{}: variant explosion ({})", name, count).into());
                }
                if variant_props.is_empty() {
                    let vv = variants
                        .get("")
                        .or_else(|| variants.values().next())
                        .map(VariantValue::from_entry)
                        .ok_or_else(|| format!("{}: no resolvable variant", name))?;
                    let empty = BTreeMap::new();
                    states.push(cube_state_row(
                        &models, &name, &vv, mesh_type, &layer_of, props, &empty,
                    )?);
                } else {
                    for combo in combos(&variant_props) {
                        match combo_map.get(&canonical_combo(&combo)) {
                            Some(vv) => {
                                let vals: BTreeMap<String, String> =
                                    combo.iter().cloned().collect();
                                states.push(cube_state_row(
                                    &models, &name, vv, mesh_type, &layer_of, props, &vals,
                                )?)
                            }
                            None => {
                                return Err(format!(
                                    "{}: variant combo {:?} has no matching key (coverage)",
                                    name, combo
                                )
                                .into())
                            }
                        }
                    }
                }
            }
        } else {
            // EMPTY / COMPLEX: single collapsed row, no faces baked. State-dependent
            // light emitters collapse to their always-lit value (SPECS approximation)
            // until per-state expansion lands for complex blocks.
            let mut entry = BlockStateEntry::default();
            entry.mesh_type = mesh_type;
            entry.is_opaque = props.is_opaque(&name, mesh_type);
            entry.is_solid = is_solid(mesh_type);
            let empty = BTreeMap::new();
            entry.light_emission = props.emission_row(&name, &empty, true);
            entry.light_filter = props.light_filter(&name, mesh_type);
            states.push(entry);
        }

        if states.is_empty() {
            return Err(format!("{}: no states emitted for mesh type {}", name, mesh_type).into());
        }

        blocks.insert(
            name.clone(),
            BakedBlock {
                mesh_type,
                states,
            },
        );
    }

    let mut baked = Baked {
        blocks,
        warnings,
        ..Default::default()
    };
    for b in baked.blocks.values() {
        for s in &b.states {
            match s.mesh_type {
                MESH_CUBE => baked.cube_states += 1,
                MESH_CROSS => baked.cross_states += 1,
                MESH_FLUID => baked.fluid_states += 1,
                MESH_COMPLEX => baked.complex_states += 1,
                _ => baked.empty_states += 1,
            }
        }
    }
    Ok(baked)
}

#[derive(Clone, Debug)]
struct VariantValue {
    model: String,
    x: i32,
    y: i32,
    uvlock: bool,
}

impl VariantValue {
    fn from_entry(v: &Value) -> Self {
        let e = variant_obj(v);
        Self {
            model: e.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string(),
            x: e.get("x").and_then(|x| x.as_i64()).unwrap_or(0) as i32,
            y: e.get("y").and_then(|y| y.as_i64()).unwrap_or(0) as i32,
            uvlock: e.get("uvlock").and_then(|u| u.as_bool()).unwrap_or(false),
        }
    }
}

/// Variants may be dict-form or array-form (randomized list); use the first entry.
fn variant_obj(e: &Value) -> &Value {
    if e.is_array() {
        e.get(0).unwrap_or(&Value::Null)
    } else {
        e
    }
}

/// Expand OR-groups: each (prop, [v1, v2]) becomes one set per value.
fn expand_assignments(assigns: &[(String, Vec<String>)]) -> Vec<Vec<(String, String)>> {
    let mut out: Vec<Vec<(String, String)>> = vec![Vec::new()];
    for (k, vals) in assigns {
        let mut next = Vec::new();
        for base in &out {
            for v in vals {
                let mut set = base.clone();
                set.push((k.clone(), v.clone()));
                next.push(set);
            }
        }
        out = next;
    }
    out
}

fn combos(props: &[(String, Vec<String>)]) -> Vec<Vec<(String, String)>> {
    let mut out: Vec<Vec<(String, String)>> = vec![Vec::new()];
    for (k, vals) in props {
        let mut next = Vec::new();
        for base in &out {
            for v in vals {
                let mut set = base.clone();
                set.push((k.clone(), v.clone()));
                next.push(set);
            }
        }
        out = next;
    }
    out
}

fn canonical_full_cube_elements(elements: &[Value]) -> bool {
    let is_full_with_cull = |e: &Value| -> bool {
        e.get("from").map(v_is_unit_origin) == Some(true)
            && e.get("to").map(v_is_unit_max) == Some(true)
            && e
                .get("faces")
                .and_then(|f: &Value| f.as_object().map(|o| o.values().count()))
                .unwrap_or(0)
                >= 6
            && e.get("faces")
                .and_then(|f: &Value| f.as_object())
                .map(|o| {
                    o.values().all(|fv| {
                        fv.get("cullface").and_then(|c| c.as_str()).is_some() || false
                    })
                })
                .unwrap_or(false)
    };

    // Case 1: every element is a full 16³ prism with directional cullfaces
    // (the multi-full-element stacked family — grass_block overlay).
    if !elements.is_empty() && elements.iter().all(is_full_with_cull) {
        return true;
    }

    // Case 2: one covering full element (slime/honey have an inner decoration
    // box inside a full outer shell). Matches shape-based occlusion: the block
    // occupies the whole voxel, so the mesher may treat it as CUBE. Interior
    // texture elements are dropped (same documented approximation as
    // grass_block overlays).
    elements.iter().any(is_full_with_cull)
}

fn v_is_unit_origin(v: &Value) -> bool {
    v.as_array()
        .map(|a| a.iter().all(|n| n.as_f64() == Some(0.0)))
        .unwrap_or(false)
}

fn v_is_unit_max(v: &Value) -> bool {
    v.as_array()
        .map(|a| a.iter().all(|n| n.as_f64() == Some(16.0)))
        .unwrap_or(false)
}

/// Build one (possibly rotated) state row for a CUBE/CROSS state.
#[allow(clippy::too_many_arguments)]
fn cube_state_row(
    models: &Models,
    block: &str,
    vv: &VariantValue,
    mesh_type: u8,
    layer_of: &dyn Fn(&str) -> Option<u16>,
    block_props: &BlockProps,
    vals: &BTreeMap<String, String>,
) -> Result<BlockStateEntry, Box<dyn std::error::Error>> {
    let mut entry = BlockStateEntry::default();
    entry.mesh_type = mesh_type;
    entry.is_opaque = block_props.is_opaque(block, mesh_type);
    entry.is_solid = is_solid(mesh_type);
    entry.light_emission = block_props.emission_row(block, vals, false);
    entry.light_filter = block_props.light_filter(block, mesh_type);

    let tint = tint_decision(block);
    let tint_mode = match tint {
        TintDecision::Mode(m) => m,
        TintDecision::Swap(_) => TINT_NONE, // baked to pre-tinted layer, no shader tint
    };

    let rm = models
        .resolve(&vv.model)
        .ok_or_else(|| format!("{}: cannot resolve model {}", block, vv.model))?;

    if mesh_type == MESH_CROSS {
        // Cross blocks: all faces carry the single cross texture; mesher renders 2 quads.
        // NB: cross element faces have no cullface — read any face value directly.
        let mut first_face_tex: Option<Value> = None;
        for e in &rm.elements {
            if let Some(Value::Object(faces)) = e.get("faces") {
                const PREF: [&str; 4] = ["north", "south", "east", "west"];
                let pick = PREF
                    .iter()
                    .find_map(|p| faces.get(*p))
                    .map(|f| f.get("texture").cloned().unwrap_or(Value::Null));
                if let Some(t) = pick {
                    first_face_tex = Some(t);
                    break;
                }
            }
        }
        let raw_tex =
            first_face_tex.ok_or_else(|| format!("{}: cross model without face textures", block))?;
        let tex = models
            .deref_texture(&raw_tex, &rm.textures)
            .ok_or_else(|| format!("{}: unresolved cross texture", block))?;
        let layer =
            layer_of(&tex).ok_or_else(|| format!("{}: texture {} not in manifest", block, tex))?;
        entry.faces = [FaceInfo { base_layer: layer, tint_mode, uv_rot: 0 }; 6];
        return Ok(entry);
    }

    // CUBE: inverse-rotate world faces onto model cullfaces, resolve texture per face.
    let inv = inverse_face_map(vv.x, vv.y);
    // collect all element faces with cullfaces (merge across elements for safety)
    let mut face_map: BTreeMap<String, FaceData> = BTreeMap::new();
    if let Some(first) = rm.elements.first() {
        face_map = face_map_of_element(first);
    }
    if face_map.len() < 6 {
        // merge faces from remaining elements (canonical cubes should have one)
        for e in &rm.elements {
            for (k, v) in face_map_of_element(e) {
                face_map.entry(k).or_insert(v);
            }
        }
    }
    for (w_idx, m_idx) in inv.iter().enumerate() {
        if *m_idx == usize::MAX {
            return Err(format!("{}: rotation {} {} dropped face {}", block, vv.x, vv.y, w_idx).into());
        }
        let face_name = FACE_NAMES[*m_idx];
        let fd = face_map
            .get(face_name)
            .ok_or_else(|| format!("{}: model {} lacks cullface {}", block, vv.model, face_name))?;
        let tex = models
            .deref_texture(&fd.texture, &rm.textures)
            .ok_or_else(|| format!("{}: unresolved texture for cullface {}", block, face_name))?;
        let layer = layer_of(&tex)
            .ok_or_else(|| format!("{}: texture {} not in manifest (golden: unresolved == 0)", block, tex))?;

        // If the block has a fixed tint baked offline, point the face at the pre-tinted layer.
        let final_layer = match tint {
            TintDecision::Swap(swap) => swap as u16,
            _ => layer,
        };
        let uv_rot = if fd.rotation % 90 == 0 {
            (fd.rotation.rem_euclid(360) / 90) as u8
        } else {
            0 // non-orthogonal face rotation: not on canonical cube parents
        };
        entry.faces[w_idx] = FaceInfo {
            base_layer: final_layer,
            tint_mode,
            uv_rot,
        };
    }
    Ok(entry)
}

/// Serialize to the flat binary layout. Deterministic (all maps are sorted).
pub fn emit(baked: &Baked) -> Vec<u8> {
    let mut names = String::new();
    let mut index = Vec::new();
    let mut state_rows = Vec::new();
    let mut state_count = 0u32;
    for (name, block) in &baked.blocks {
        let off = names.len() as u32;
        names.push_str(name);
        index.push(BlockIndexEntry {
            name_off: off,
            name_len: name.len() as u32,
            state_start: state_count,
            state_count: block.states.len() as u32,
            mesh_type: block.mesh_type,
            has_variants: u8::from(block.states.len() > 1),
            pad: 0,
        });
        state_count += block.states.len() as u32;
        for s in &block.states {
            s.write_le(&mut state_rows);
        }
    }

    let mut out = Vec::with_capacity(baked.blocks.len() * 64);
    out.extend_from_slice(b"VXRG"); // magic
    out.extend_from_slice(&1u32.to_le_bytes()); // version
    out.extend_from_slice(&(baked.blocks.len() as u32).to_le_bytes());
    out.extend_from_slice(&state_count.to_le_bytes());
    out.extend_from_slice(&(names.len() as u32).to_le_bytes()); // names blob length
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved 2
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved: magic + 7 u32 = 32 B
    debug_assert_eq!(out.len(), HEADER_BYTES);
    out.extend_from_slice(names.as_bytes());
    for e in &index {
        e.write_le(&mut out);
    }
    out.extend_from_slice(&state_rows);
    out
}

pub fn write_registry(assets: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let props = BlockProps::load(&assets.join("registry/block_properties.json"))
        .map_err(|e| {
            format!(
                "block_properties.json missing/broken ({}); run `cargo run -p asset_pipeline \
                 --bin property_extractor ~/.minecraft-decomp/src/Blocks.java` offline first",
                e
            )
        })?;
    let baked = bake(
        &assets.join("blockstates"),
        &assets.join("models/block"),
        &assets.join("textures/texture_manifest.json"),
        &props,
    )?;
    for w in &baked.warnings {
        eprintln!("registry warning: {}", w);
    }
    let out_dir = assets.join("registry");
    std::fs::create_dir_all(&out_dir)?;
    let blob = emit(&baked);
    std::fs::write(out_dir.join("block_registry.bin"), &blob)?;
    Ok(format!(
        "blocks={} states={} (cube={} cross={} fluid={} complex={} empty={}) bytes={}",
        baked.blocks.len(),
        baked.cube_states + baked.cross_states + baked.fluid_states + baked.complex_states + baked.empty_states,
        baked.cube_states,
        baked.cross_states,
        baked.fluid_states,
        baked.complex_states,
        baked.empty_states,
        blob.len()
    ))
}

fn main() {
    let assets = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets".to_string());
    match write_registry(Path::new(&assets)) {
        Ok(summary) => {
            println!("Block Registry Bake Complete: {}", summary);
        }
        Err(e) => {
            eprintln!("Block Registry Bake FAILED: {}", e);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Golden tests (run against the real extracted data)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assets_dir() -> PathBuf {
        // crates/asset_pipeline/src/bin -> repo root is 3 levels up
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets")
            .canonicalize()
            .expect("assets dir")
    }

    fn baked() -> Baked {
        let dir = assets_dir();
        let props = BlockProps::load(&dir.join("registry/block_properties.json"))
            .expect("block_properties.json (run property_extractor offline)");
        bake(
            &dir.join("blockstates"),
            &dir.join("models/block"),
            &dir.join("textures/texture_manifest.json"),
            &props,
        )
        .expect("bake must succeed on real data (golden: unresolved == 0)")
    }

    fn manifest_layers() -> BTreeMap<String, u32> {
        let dir = assets_dir();
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("textures/texture_manifest.json")).unwrap())
                .unwrap();
        let mut out = BTreeMap::new();
        if let Some(Value::Object(tex)) = manifest.get("textures") {
            for (k, v) in tex {
                if let Some(b) = v.get("base_layer").and_then(|b| b.as_u64()) {
                    out.insert(k.clone(), b as u32);
                }
            }
        }
        out
    }

    #[test]
    fn golden_coverage_and_classification() {
        let b = baked();
        assert!(b.warnings.is_empty(), "no warnings: {:?}", b.warnings);
        assert!(b.blocks.len() > 1_000, "{} block states files", b.blocks.len());
        assert!(b.cube_states > 380, "{} cube states", b.cube_states);
        assert!(b.cross_states > 80, "{} cross states", b.cross_states);
        for name in ["stone", "oak_log", "water", "pink_petals", "leaf_litter"] {
            assert!(b.blocks.contains_key(name), "missing block {}", name);
        }
    }

    #[test]
    fn golden_stone_faces_match_manifest() {
        let b = baked();
        let layers = manifest_layers();
        let stone_layer = layers["stone"] as u16;
        let entry = &b.blocks["stone"].states[0];
        assert_eq!(entry.mesh_type, MESH_CUBE);
        for (i, face) in entry.faces.iter().enumerate() {
            assert_eq!(face.base_layer, stone_layer, "stone face {}", i);
            assert_eq!(face.tint_mode, TINT_NONE);
        }
    }

    #[test]
    fn golden_oak_log_axis_end_caps() {
        let b = baked();
        let log = &b.blocks["oak_log"];
        assert_eq!(log.mesh_type, MESH_CUBE);
        assert_eq!(log.states.len(), 3, "axis x/y/z");
        let top = manifest_layers()["oak_log_top"] as u16;
        let side = manifest_layers()["oak_log"] as u16;

        let axis_x = &log.states[0];
        assert_eq!(axis_x.faces[FACE_EAST].base_layer, top, "axis x: east end");
        assert_eq!(axis_x.faces[FACE_WEST].base_layer, top, "axis x: west end");
        assert_eq!(axis_x.faces[FACE_UP].base_layer, side);
        assert_eq!(axis_x.faces[FACE_SOUTH].base_layer, side);

        let axis_z = &log.states[2];
        assert_eq!(axis_z.faces[FACE_SOUTH].base_layer, top, "axis z: south end");
        assert_eq!(axis_z.faces[FACE_NORTH].base_layer, top, "axis z: north end");

        let axis_y = &log.states[1];
        assert_eq!(axis_y.faces[FACE_UP].base_layer, top, "axis y: up end");
        assert_eq!(axis_y.faces[FACE_DOWN].base_layer, top, "axis y: down end");
        for f in [FACE_NORTH, FACE_SOUTH, FACE_EAST, FACE_WEST] {
            assert_eq!(axis_y.faces[f].base_layer, side);
        }
    }

    #[test]
    fn golden_tint_decisions() {
        let b = baked();
        let water = &b.blocks["water"].states[0];
        assert_eq!(water.mesh_type, MESH_FLUID);
        for f in &water.faces {
            assert_eq!(f.tint_mode, TINT_WATER);
        }
        assert_eq!(water.light_filter, 1);

        assert_eq!(b.blocks["oak_leaves"].states[0].faces[FACE_UP].tint_mode, TINT_FOLIAGE);
        assert_eq!(
            b.blocks["birch_leaves"].states[0].faces[FACE_UP].base_layer,
            1840,
            "birch pre-tinted layer"
        );
        assert_eq!(
            b.blocks["spruce_leaves"].states[0].faces[FACE_UP].base_layer,
            1841
        );
        assert_eq!(b.blocks["pink_petals"].states[0].faces[FACE_UP].tint_mode, TINT_NONE,
            "petals are NOT grass-tinted (BlockColors: BLANK first layer)");
        let grass = &b.blocks["grass_block"];
        assert_eq!(grass.states[0].faces[FACE_UP].tint_mode, TINT_GRASS);
    }

    #[test]
    fn golden_light_release_values() {
        let b = baked();

        // LIT-conditioned emitters are per-state (Blocks.java litBlockEmission).
        // furnace: facing(4) × lit(2) → half the rows emit 13, half 0.
        let rows = &b.blocks["furnace"].states;
        assert_eq!(rows.len(), 8);
        assert_eq!(rows.iter().filter(|s| s.light_emission == 13).count(), 4, "furnace lit rows");
        assert_eq!(rows.iter().filter(|s| s.light_emission == 0).count(), 4, "furnace unlit rows");
        assert!(rows.iter().all(|s| s.light_filter == 15));

        // redstone_lamp is ONE block, LITλ=15 (no separate *_on block exists).
        let rows = &b.blocks["redstone_lamp"].states;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.iter().filter(|s| s.light_emission == 15).count(), 1);
        assert_eq!(rows.iter().filter(|s| s.light_emission == 0).count(), 1);

        // Complex LIT emitters collapse to always-lit (SPECS approximation): torches,
        // campfires. Collapse value equals the lit emission.
        assert_eq!(b.blocks["torch"].states[0].light_emission, 14);
        assert_eq!(b.blocks["redstone_torch"].states[0].light_emission, 7);
        assert_eq!(b.blocks["campfire"].states[0].light_emission, 15);
        assert_eq!(b.blocks["campfire"].states[0].light_filter, 0);

        // Fluids: water AND lava dampen by 1 (LiquidBlock default path; the old
        // "lava blocks light" value was a myth — it merely emits 15).
        assert_eq!(b.blocks["lava"].states[0].light_emission, 15);
        assert_eq!(b.blocks["lava"].states[0].light_filter, 1);
        assert_eq!(b.blocks["water"].states[0].light_emission, 0);

        // Ice family split: ice/frosted ride the default path (1); packed/blue are
        // occluding full prisms (solidRender → 15).
        assert_eq!(b.blocks["ice"].states[0].light_filter, 1);
        assert_eq!(b.blocks["frosted_ice"].states[0].light_filter, 1);
        assert_eq!(b.blocks["packed_ice"].states[0].light_filter, 15);
        assert_eq!(b.blocks["blue_ice"].states[0].light_filter, 15);

        // Glass family: propagatesSkylightDown class override → 0 and non-occluding.
        for n in ["glass", "white_stained_glass", "glass_pane"] {
            let st = &b.blocks[n].states[0];
            assert_eq!(st.light_filter, 0, "{n}");
            assert_eq!(st.is_opaque, 0, "{n}");
        }
        assert_eq!(b.blocks["tinted_glass"].states[0].light_filter, 15);

        // Full-shape non-occluding: slime/honey dampen 1; leaves do too.
        assert_eq!(b.blocks["slime_block"].states[0].light_filter, 1);
        assert_eq!(b.blocks["slime_block"].states[0].is_opaque, 0);
        assert_eq!(b.blocks["honey_block"].states[0].light_filter, 1);
        let leaves = &b.blocks["oak_leaves"].states[0];
        assert_eq!(leaves.light_filter, 1);
        assert_eq!(leaves.is_opaque, 0);

        // Fire collapses to 15; opaque cubes keep 15/15.
        assert_eq!(b.blocks["fire"].states[0].light_emission, 15);
        assert_eq!(b.blocks["glowstone"].states[0].light_emission, 15);
        assert_eq!(b.blocks["glowstone"].states[0].light_filter, 15);
        assert_eq!(b.blocks["stone"].states[0].light_filter, 15);
    }

    #[test]
    fn golden_binary_layout_roundtrip() {
        let b = baked();
        let blob = emit(&b);
        // header
        assert_eq!(&blob[0..4], b"VXRG");
        let le_u32 = |off: usize| u32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
        assert_eq!(le_u32(4), 1, "version");
        assert_eq!(le_u32(8) as usize, b.blocks.len());
        // names blob occupies 32..32+len; index entries follow it
        let names_len = le_u32(16) as usize;
        let idx_off = 32 + names_len;
        // first index entry points at the alphabetically first block with zero offset
        assert_eq!(le_u32(idx_off) as usize, 0, "first name offset");
    }
}
