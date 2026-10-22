//! Single `fs::read` offline block registry reader (`assets/registry/block_registry.bin`), no serde/JSON.
//!
//! Provides engine-side access to the baked catalog of 32-byte `BlockStateEntry` rows
//! and 20-byte `BlockIndexEntry` records. The engine loads this directly into flat arrays
//! without parsing JSON.

use std::path::Path;

pub const FACE_UP: usize = 0; // +Y
pub const FACE_DOWN: usize = 1; // -Y
pub const FACE_SOUTH: usize = 2; // +Z
pub const FACE_NORTH: usize = 3; // -Z
pub const FACE_EAST: usize = 4; // +X
pub const FACE_WEST: usize = 5; // -X

pub const MESH_EMPTY: u8 = 0;
pub const MESH_CUBE: u8 = 1;
pub const MESH_CROSS: u8 = 2;
pub const MESH_FLUID: u8 = 3;
pub const MESH_COMPLEX: u8 = 4;

pub const HEADER_BYTES: usize = 32;
pub const INDEX_ENTRY_BYTES: usize = 20;
pub const STATE_ENTRY_BYTES: usize = 32;
pub const MAGIC: &[u8; 4] = b"VXRG";

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FaceInfo {
    pub base_layer: u16, // texture array layer
    pub tint_mode: u8,   // TINT_*
    pub uv_rot: u8,      // 0..3 (90° steps)
}

pub const CULL_OPAQUE: u8 = 0;
pub const CULL_LEAVES: u8 = 1;
pub const CULL_WATER: u8 = 2;
pub const CULL_LAVA: u8 = 3;
pub const CULL_TRANSPARENT_START: u8 = 4;

pub const LAYER_SOLID: u8 = 0;
pub const LAYER_CUTOUT_MIPPED: u8 = 1;
pub const LAYER_CUTOUT: u8 = 2;
pub const LAYER_TRANSLUCENT: u8 = 3;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockStateEntry {
    pub faces: [FaceInfo; 6],
    pub mesh_type: u8,
    pub is_opaque: u8,
    pub is_solid: u8,
    pub light_emission: u8, // 0..15 block light emitted
    pub light_filter: u8,   // 0..15 light attenuation per voxel
    pub pad: [u8; 3],       // pad[0] = cull_class, pad[1] = render_layer
}

impl BlockStateEntry {
    #[inline(always)]
    pub fn cull_class(&self) -> u8 {
        self.pad[0]
    }

    #[inline(always)]
    pub fn render_layer(&self) -> u8 {
        self.pad[1]
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockIndexEntry {
    pub name_off: u32,
    pub name_len: u32,
    pub state_start: u32,
    pub state_count: u32,
    pub mesh_type: u8,
    pub has_variants: u8,
    pub pad: u16,
}

#[derive(Debug)]
pub enum RegistryError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    CorruptHeader,
    UnexpectedEof { expected: usize, actual: usize },
    InvalidUtf8,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::InvalidMagic => write!(f, "invalid registry magic (expected 'VXRG')"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported registry version: {}", v),
            Self::CorruptHeader => write!(f, "corrupt registry header"),
            Self::UnexpectedEof { expected, actual } => {
                write!(f, "unexpected EOF: expected {} bytes, got {}", expected, actual)
            }
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in block names blob"),
        }
    }
}

impl std::error::Error for RegistryError {}

impl From<std::io::Error> for RegistryError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// In-memory catalog loaded from `block_registry.bin`.
pub struct BlockRegistry {
    names: String,
    blocks: Vec<BlockIndexEntry>,
    states: Vec<BlockStateEntry>,
}

impl BlockRegistry {
    /// Loads the binary registry from a raw byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RegistryError> {
        if bytes.len() < HEADER_BYTES {
            return Err(RegistryError::CorruptHeader);
        }

        if &bytes[0..4] != MAGIC {
            return Err(RegistryError::InvalidMagic);
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 {
            return Err(RegistryError::UnsupportedVersion(version));
        }

        let block_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let state_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let names_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;

        let expected_total = HEADER_BYTES + names_len + block_count * INDEX_ENTRY_BYTES + state_count * STATE_ENTRY_BYTES;
        if bytes.len() < expected_total {
            return Err(RegistryError::UnexpectedEof {
                expected: expected_total,
                actual: bytes.len(),
            });
        }

        let names_start = HEADER_BYTES;
        let names_end = names_start + names_len;
        let names_str = std::str::from_utf8(&bytes[names_start..names_end])
            .map_err(|_| RegistryError::InvalidUtf8)?
            .to_string();

        let index_start = names_end;
        let mut blocks = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let off = index_start + i * INDEX_ENTRY_BYTES;
            let name_off = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let name_len = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            let state_start = u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap());
            let state_count_entry = u32::from_le_bytes(bytes[off + 12..off + 16].try_into().unwrap());
            let mesh_type = bytes[off + 16];
            let has_variants = bytes[off + 17];
            let pad = u16::from_le_bytes(bytes[off + 18..off + 20].try_into().unwrap());

            blocks.push(BlockIndexEntry {
                name_off,
                name_len,
                state_start,
                state_count: state_count_entry,
                mesh_type,
                has_variants,
                pad,
            });
        }

        let states_start = index_start + block_count * INDEX_ENTRY_BYTES;
        let mut states = Vec::with_capacity(state_count);
        for i in 0..state_count {
            let off = states_start + i * STATE_ENTRY_BYTES;
            let mut faces = [FaceInfo::default(); 6];
            for f in 0..6 {
                let f_off = off + f * 4;
                let base_layer = u16::from_le_bytes(bytes[f_off..f_off + 2].try_into().unwrap());
                let tint_mode = bytes[f_off + 2];
                let uv_rot = bytes[f_off + 3];
                faces[f] = FaceInfo {
                    base_layer,
                    tint_mode,
                    uv_rot,
                };
            }

            let mesh_type = bytes[off + 24];
            let is_opaque = bytes[off + 25];
            let is_solid = bytes[off + 26];
            let light_emission = bytes[off + 27];
            let light_filter = bytes[off + 28];
            let pad = [bytes[off + 29], bytes[off + 30], bytes[off + 31]];

            states.push(BlockStateEntry {
                faces,
                mesh_type,
                is_opaque,
                is_solid,
                light_emission,
                light_filter,
                pad,
            });
        }

        Ok(Self {
            names: names_str,
            blocks,
            states,
        })
    }

    /// Loads the binary registry from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Returns the block name for a given index entry.
    #[inline(always)]
    pub fn get_block_name(&self, block: &BlockIndexEntry) -> &str {
        let start = block.name_off as usize;
        let end = start + block.name_len as usize;
        &self.names[start..end]
    }

    /// Finds a block by name using binary search (index entries are stored alphabetically).
    pub fn get_block(&self, name: &str) -> Option<&BlockIndexEntry> {
        self.blocks
            .binary_search_by(|probe| self.get_block_name(probe).cmp(name))
            .ok()
            .map(|idx| &self.blocks[idx])
    }

    /// Retrieves state by raw state id.
    #[inline(always)]
    pub fn get_state(&self, state_id: u32) -> Option<&BlockStateEntry> {
        self.states.get(state_id as usize)
    }

    /// Direct state reference without bounds check (hot path for mesher/lighting).
    #[inline(always)]
    pub fn get_state_unchecked(&self, state_id: u32) -> &BlockStateEntry {
        &self.states[state_id as usize]
    }

    #[inline(always)]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    #[inline(always)]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline(always)]
    pub fn light_emission(&self, state_id: u32) -> u8 {
        self.states
            .get(state_id as usize)
            .map(|s| s.light_emission)
            .unwrap_or(0)
    }

    #[inline(always)]
    pub fn light_filter(&self, state_id: u32) -> u8 {
        self.states
            .get(state_id as usize)
            .map(|s| s.light_filter)
            .unwrap_or(0)
    }

    #[inline(always)]
    pub fn is_opaque(&self, state_id: u32) -> bool {
        self.states
            .get(state_id as usize)
            .map(|s| s.is_opaque != 0)
            .unwrap_or(false)
    }

    #[inline(always)]
    pub fn is_solid(&self, state_id: u32) -> bool {
        self.states
            .get(state_id as usize)
            .map(|s| s.is_solid != 0)
            .unwrap_or(false)
    }
}
