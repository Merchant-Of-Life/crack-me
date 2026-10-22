//! 4,096 B/section packed light storage and chunk column structures.
//!
//! Pinned invariants (SPECS.md):
//! - Section layout: 4,096 bytes per 16³ section, one byte per voxel: `(sky << 4) | (block & 0x0F)`.
//! - Section voxel indexing: `(y << 8) | (z << 4) | x` (X fastest, Y slowest).
//! - World cover: 24 sections covering Y in `[-64, 320)`, `MIN_Y = -64`.
//! - `get_raw_brightness`: `max(blockLight, skyLight - skyDampen)` (LevelLightEngine.java:142-146).

pub const SECTION_SIZE: usize = 16;
pub const SECTION_VOXELS: usize = 4096;
pub const SECTION_COUNT: usize = 24;
pub const MIN_Y: i32 = -64;
pub const MAX_Y: i32 = 320; // MIN_Y + SECTION_COUNT * SECTION_SIZE

/// Computes the 1D voxel index within a 16³ section.
///
/// Convention: `(y << 8) | (z << 4) | x` (X fastest, Z middle, Y slowest).
#[inline(always)]
pub fn section_voxel_index(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < SECTION_SIZE && y < SECTION_SIZE && z < SECTION_SIZE);
    (y << 8) | (z << 4) | x
}

/// Decodes a 1D section index into `(x, y, z)` section coordinates.
#[inline(always)]
pub fn section_index_to_coords(idx: usize) -> (usize, usize, usize) {
    debug_assert!(idx < SECTION_VOXELS);
    let x = idx & 0x0F;
    let z = (idx >> 4) & 0x0F;
    let y = (idx >> 8) & 0x0F;
    (x, y, z)
}

/// Exactly 4,096 bytes storing 4-bit sky light and 4-bit block light per voxel.
///
/// Storage byte: `(sky << 4) | (block & 0x0F)`.
#[repr(C, align(1))]
#[derive(Clone, PartialEq, Eq)]
pub struct LightSection {
    pub raw: [u8; SECTION_VOXELS],
}

impl Default for LightSection {
    fn default() -> Self {
        Self::new()
    }
}

impl LightSection {
    /// Creates a section with all lights cleared to 0.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            raw: [0u8; SECTION_VOXELS],
        }
    }

    #[inline(always)]
    pub fn get_block_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.get_block_light_idx(section_voxel_index(x, y, z))
    }

    #[inline(always)]
    pub fn set_block_light(&mut self, x: usize, y: usize, z: usize, val: u8) {
        self.set_block_light_idx(section_voxel_index(x, y, z), val);
    }

    #[inline(always)]
    pub fn get_sky_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.get_sky_light_idx(section_voxel_index(x, y, z))
    }

    #[inline(always)]
    pub fn set_sky_light(&mut self, x: usize, y: usize, z: usize, val: u8) {
        self.set_sky_light_idx(section_voxel_index(x, y, z), val);
    }

    #[inline(always)]
    pub fn get_block_light_idx(&self, idx: usize) -> u8 {
        self.raw[idx] & 0x0F
    }

    #[inline(always)]
    pub fn set_block_light_idx(&mut self, idx: usize, val: u8) {
        self.raw[idx] = (self.raw[idx] & 0xF0) | (val & 0x0F);
    }

    #[inline(always)]
    pub fn get_sky_light_idx(&self, idx: usize) -> u8 {
        (self.raw[idx] >> 4) & 0x0F
    }

    #[inline(always)]
    pub fn set_sky_light_idx(&mut self, idx: usize, val: u8) {
        self.raw[idx] = (self.raw[idx] & 0x0F) | ((val & 0x0F) << 4);
    }

    #[inline(always)]
    pub fn get_packed_idx(&self, idx: usize) -> u8 {
        self.raw[idx]
    }

    #[inline(always)]
    pub fn set_packed_idx(&mut self, idx: usize, packed: u8) {
        self.raw[idx] = packed;
    }

    /// Calculates raw brightness at the given section voxel according to
    /// vanilla `LevelLightEngine.java:142-146`:
    ///
    /// `max(blockLight, skyLight - skyDampen)`
    ///
    /// # Arguments
    /// * `sky_dampen` - Caller-supplied ambient darkening (e.g. night, rain, thunder,
    ///   or dimension ambient reduction), NOT the voxel's own `light_filter`.
    #[inline(always)]
    pub fn get_raw_brightness(&self, x: usize, y: usize, z: usize, sky_dampen: u8) -> u8 {
        self.get_raw_brightness_idx_with_dampen(section_voxel_index(x, y, z), sky_dampen)
    }

    /// Index-based `get_raw_brightness`.
    #[inline(always)]
    pub fn get_raw_brightness_idx(&self, idx: usize) -> u8 {
        self.get_raw_brightness_idx_with_dampen(idx, 0)
    }

    /// Index-based `get_raw_brightness` with explicit ambient darkening parameter.
    #[inline(always)]
    pub fn get_raw_brightness_idx_with_dampen(&self, idx: usize, sky_dampen: u8) -> u8 {
        let block = self.get_block_light_idx(idx);
        let sky = self.get_sky_light_idx(idx);
        let dimmed_sky = sky.saturating_sub(sky_dampen);
        block.max(dimmed_sky)
    }
}

/// A vertical column of 24 light sections covering Y ∈ [-64, 320).
pub struct ChunkLightColumn {
    pub sections: Box<[LightSection; SECTION_COUNT]>,
}

impl Default for ChunkLightColumn {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ChunkLightColumn {
    fn clone(&self) -> Self {
        Self {
            sections: self.sections.clone(),
        }
    }
}

impl ChunkLightColumn {
    pub fn new() -> Self {
        // Allocate zeroed on heap to avoid 96 KiB stack frame
        let sections = vec![LightSection::new(); SECTION_COUNT]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("size mismatch"));
        Self { sections }
    }

    #[inline(always)]
    pub fn section_for_y(y: i32) -> Option<(usize, usize)> {
        if !(MIN_Y..MAX_Y).contains(&y) {
            return None;
        }
        let rel_y = (y - MIN_Y) as usize;
        let s = rel_y >> 4;
        let local_y = rel_y & 0x0F;
        Some((s, local_y))
    }

    #[inline(always)]
    pub fn get_block_light(&self, x: usize, y: i32, z: usize) -> u8 {
        if y >= MAX_Y {
            return 0;
        }
        match Self::section_for_y(y) {
            Some((s, local_y)) => self.sections[s].get_block_light(x, local_y, z),
            None => 0, // y < MIN_Y
        }
    }

    #[inline(always)]
    pub fn set_block_light(&mut self, x: usize, y: i32, z: usize, val: u8) {
        if let Some((s, local_y)) = Self::section_for_y(y) {
            self.sections[s].set_block_light(x, local_y, z, val);
        }
    }

    #[inline(always)]
    pub fn get_sky_light(&self, x: usize, y: i32, z: usize) -> u8 {
        if y >= MAX_Y {
            return 15; // Vanilla padding section above MAX_Y is full sky (SPECS Step 2.5)
        }
        match Self::section_for_y(y) {
            Some((s, local_y)) => self.sections[s].get_sky_light(x, local_y, z),
            None => 0, // y < MIN_Y
        }
    }

    #[inline(always)]
    pub fn set_sky_light(&mut self, x: usize, y: i32, z: usize, val: u8) {
        if let Some((s, local_y)) = Self::section_for_y(y) {
            self.sections[s].set_sky_light(x, local_y, z, val);
        }
    }

    #[inline(always)]
    pub fn get_packed_light(&self, x: usize, y: i32, z: usize) -> u8 {
        if y >= MAX_Y {
            return 0xF0; // sky 15, block 0
        }
        match Self::section_for_y(y) {
            Some((s, local_y)) => self.sections[s].get_packed_idx(section_voxel_index(x, local_y, z)),
            None => 0, // y < MIN_Y
        }
    }

    #[inline(always)]
    pub fn set_packed_light(&mut self, x: usize, y: i32, z: usize, packed: u8) {
        if let Some((s, local_y)) = Self::section_for_y(y) {
            self.sections[s].set_packed_idx(section_voxel_index(x, local_y, z), packed);
        }
    }

    /// Calculates raw brightness at world voxel `(x, y, z)`.
    ///
    /// # Arguments
    /// * `sky_dampen` - Caller-supplied ambient darkening (weather/void), NOT the voxel's light_filter.
    #[inline(always)]
    pub fn get_raw_brightness(&self, x: usize, y: i32, z: usize, sky_dampen: u8) -> u8 {
        if y >= MAX_Y {
            return 15u8.saturating_sub(sky_dampen);
        }
        match Self::section_for_y(y) {
            Some((s, local_y)) => self.sections[s].get_raw_brightness(x, local_y, z, sky_dampen),
            None => 0, // y < MIN_Y
        }
    }
}

/// A vertical column of voxel state IDs (24 sections of 4,096 voxels each).
pub struct ChunkVoxelColumn {
    pub sections: Box<[[u16; SECTION_VOXELS]; SECTION_COUNT]>,
}

impl ChunkVoxelColumn {
    pub fn new(default_state: u16) -> Self {
        let section = [default_state; SECTION_VOXELS];
        let sections = vec![section; SECTION_COUNT]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("size mismatch"));
        Self { sections }
    }

    #[inline(always)]
    pub fn get_state(&self, x: usize, y: i32, z: usize) -> u16 {
        match ChunkLightColumn::section_for_y(y) {
            Some((s, local_y)) => self.sections[s][section_voxel_index(x, local_y, z)],
            None => 0,
        }
    }

    #[inline(always)]
    pub fn set_state(&mut self, x: usize, y: i32, z: usize, state_id: u16) {
        if let Some((s, local_y)) = ChunkLightColumn::section_for_y(y) {
            let idx = section_voxel_index(x, local_y, z);
            self.sections[s][idx] = state_id;
        }
    }
}
