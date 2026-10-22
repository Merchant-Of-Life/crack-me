//! Pure CPU Greedy Mesher for 16³ voxel sections.
//!
//! Converts chunk sections and light fields into compact 64-bit PackedVoxelVertex
//! triangle buffers with 4-corner AO, sealed-corner light blending, quad diagonal flip,
//! and 8-neighbor halo boundary access according to SPECS.md Step 3.

use crate::light::{
    ChunkLightColumn, ChunkVoxelColumn, MAX_Y, MIN_Y, SECTION_COUNT, SECTION_SIZE,
};

pub const HALO_SIZE: usize = 18;
pub const HALO_VOXELS: usize = HALO_SIZE * HALO_SIZE * HALO_SIZE; // 5,832

/// Padded 18×18×18 halo buffer covering local section coordinates x, y, z ∈ [-1, 16].
/// Indexing: `halo_index(x + 1, y + 1, z + 1) = ((y + 1) * 324) + ((z + 1) * 18) + (x + 1)`.
pub struct SectionHalo {
    pub voxels: Box<[u16; HALO_VOXELS]>,
    pub light: Box<[u8; HALO_VOXELS]>, // packed (sky << 4) | (block & 0x0F)
}

impl Default for SectionHalo {
    fn default() -> Self {
        Self::new()
    }
}

impl SectionHalo {
    pub fn new() -> Self {
        let voxels = vec![0u16; HALO_VOXELS]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("size mismatch"));
        let light = vec![0u8; HALO_VOXELS]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("size mismatch"));
        Self { voxels, light }
    }

    #[inline(always)]
    pub fn halo_index(hx: usize, hy: usize, hz: usize) -> usize {
        debug_assert!(hx < HALO_SIZE && hy < HALO_SIZE && hz < HALO_SIZE);
        hy * 324 + hz * 18 + hx
    }

    /// Access voxel state at section-relative coordinate x, y, z ∈ [-1, 16].
    #[inline(always)]
    pub fn get_voxel(&self, x: i32, y: i32, z: i32) -> u16 {
        debug_assert!((-1..=16).contains(&x));
        debug_assert!((-1..=16).contains(&y));
        debug_assert!((-1..=16).contains(&z));
        self.voxels[Self::halo_index((x + 1) as usize, (y + 1) as usize, (z + 1) as usize)]
    }

    /// Access packed light at section-relative coordinate x, y, z ∈ [-1, 16].
    #[inline(always)]
    pub fn get_packed_light(&self, x: i32, y: i32, z: i32) -> u8 {
        debug_assert!((-1..=16).contains(&x));
        debug_assert!((-1..=16).contains(&y));
        debug_assert!((-1..=16).contains(&z));
        self.light[Self::halo_index((x + 1) as usize, (y + 1) as usize, (z + 1) as usize)]
    }

    #[inline(always)]
    pub fn get_sky_light(&self, x: i32, y: i32, z: i32) -> u8 {
        (self.get_packed_light(x, y, z) >> 4) & 0x0F
    }

    #[inline(always)]
    pub fn get_block_light(&self, x: i32, y: i32, z: i32) -> u8 {
        self.get_packed_light(x, y, z) & 0x0F
    }

    #[inline(always)]
    pub fn set_voxel(&mut self, x: i32, y: i32, z: i32, val: u16) {
        debug_assert!((-1..=16).contains(&x));
        debug_assert!((-1..=16).contains(&y));
        debug_assert!((-1..=16).contains(&z));
        self.voxels[Self::halo_index((x + 1) as usize, (y + 1) as usize, (z + 1) as usize)] = val;
    }

    #[inline(always)]
    pub fn set_packed_light(&mut self, x: i32, y: i32, z: i32, val: u8) {
        debug_assert!((-1..=16).contains(&x));
        debug_assert!((-1..=16).contains(&y));
        debug_assert!((-1..=16).contains(&z));
        self.light[Self::halo_index((x + 1) as usize, (y + 1) as usize, (z + 1) as usize)] = val;
    }

    /// Populates this halo buffer with voxels and light for `section_idx` (0..24),
    /// pulling from the center chunk column and all 8 horizontal neighbor columns.
    ///
    /// Clamps:
    /// - y < MIN_Y (-64): air_state_id, packed light 0x00 (sky 0, block 0)
    /// - y >= MAX_Y (320): air_state_id, packed light 0xF0 (sky 15, block 0)
    pub fn populate(
        &mut self,
        section_idx: usize,
        center_voxels: &ChunkVoxelColumn,
        center_light: &ChunkLightColumn,
        neighbors: &HorizontalNeighbors,
        air_state_id: u16,
    ) {
        assert!(section_idx < SECTION_COUNT, "section_idx must be < SECTION_COUNT");
        let base_y = MIN_Y + (section_idx as i32) * (SECTION_SIZE as i32);

        for rel_y in -1..=16 {
            let world_y = base_y + rel_y;

            if world_y < MIN_Y {
                // y < -64 void floor clamp: air, sky 0 / block 0
                for rel_z in -1..=16 {
                    for rel_x in -1..=16 {
                        self.set_voxel(rel_x, rel_y, rel_z, air_state_id);
                        self.set_packed_light(rel_x, rel_y, rel_z, 0x00);
                    }
                }
                continue;
            }

            if world_y >= MAX_Y {
                // y >= 320 sky ceiling clamp: air, sky 15 / block 0
                for rel_z in -1..=16 {
                    for rel_x in -1..=16 {
                        self.set_voxel(rel_x, rel_y, rel_z, air_state_id);
                        self.set_packed_light(rel_x, rel_y, rel_z, 0xF0);
                    }
                }
                continue;
            }

            // Interior section slice: rel_x ∈ 0..16, rel_z ∈ 0..16 from center
            for rel_z in 0..16 {
                for rel_x in 0..16 {
                    let v = center_voxels.get_state(rel_x as usize, world_y, rel_z as usize);
                    let l = center_light.get_packed_light(rel_x as usize, world_y, rel_z as usize);
                    self.set_voxel(rel_x, rel_y, rel_z, v);
                    self.set_packed_light(rel_x, rel_y, rel_z, l);
                }
            }

            // Cardinal West (rel_x = -1, rel_z ∈ 0..16) -> West neighbor at x = 15
            for rel_z in 0..16 {
                let v = neighbors.west.0.get_state(15, world_y, rel_z as usize);
                let l = neighbors.west.1.get_packed_light(15, world_y, rel_z as usize);
                self.set_voxel(-1, rel_y, rel_z, v);
                self.set_packed_light(-1, rel_y, rel_z, l);
            }

            // Cardinal East (rel_x = 16, rel_z ∈ 0..16) -> East neighbor at x = 0
            for rel_z in 0..16 {
                let v = neighbors.east.0.get_state(0, world_y, rel_z as usize);
                let l = neighbors.east.1.get_packed_light(0, world_y, rel_z as usize);
                self.set_voxel(16, rel_y, rel_z, v);
                self.set_packed_light(16, rel_y, rel_z, l);
            }

            // Cardinal North (rel_z = -1, rel_x ∈ 0..16) -> North neighbor at z = 15
            for rel_x in 0..16 {
                let v = neighbors.north.0.get_state(rel_x as usize, world_y, 15);
                let l = neighbors.north.1.get_packed_light(rel_x as usize, world_y, 15);
                self.set_voxel(rel_x, rel_y, -1, v);
                self.set_packed_light(rel_x, rel_y, -1, l);
            }

            // Cardinal South (rel_z = 16, rel_x ∈ 0..16) -> South neighbor at z = 0
            for rel_x in 0..16 {
                let v = neighbors.south.0.get_state(rel_x as usize, world_y, 0);
                let l = neighbors.south.1.get_packed_light(rel_x as usize, world_y, 0);
                self.set_voxel(rel_x, rel_y, 16, v);
                self.set_packed_light(rel_x, rel_y, 16, l);
            }

            // Diagonal NW (rel_x = -1, rel_z = -1) -> NW neighbor at (15, 15)
            let v_nw = neighbors.nw.0.get_state(15, world_y, 15);
            let l_nw = neighbors.nw.1.get_packed_light(15, world_y, 15);
            self.set_voxel(-1, rel_y, -1, v_nw);
            self.set_packed_light(-1, rel_y, -1, l_nw);

            // Diagonal NE (rel_x = 16, rel_z = -1) -> NE neighbor at (0, 15)
            let v_ne = neighbors.ne.0.get_state(0, world_y, 15);
            let l_ne = neighbors.ne.1.get_packed_light(0, world_y, 15);
            self.set_voxel(16, rel_y, -1, v_ne);
            self.set_packed_light(16, rel_y, -1, l_ne);

            // Diagonal SW (rel_x = -1, rel_z = 16) -> SW neighbor at (15, 0)
            let v_sw = neighbors.sw.0.get_state(15, world_y, 0);
            let l_sw = neighbors.sw.1.get_packed_light(15, world_y, 0);
            self.set_voxel(-1, rel_y, 16, v_sw);
            self.set_packed_light(-1, rel_y, 16, l_sw);

            // Diagonal SE (rel_x = 16, rel_z = 16) -> SE neighbor at (0, 0)
            let v_se = neighbors.se.0.get_state(0, world_y, 0);
            let l_se = neighbors.se.1.get_packed_light(0, world_y, 0);
            self.set_voxel(16, rel_y, 16, v_se);
            self.set_packed_light(16, rel_y, 16, l_se);
        }
    }
}

/// All 8 horizontal neighbor columns required for meshing a section.
pub struct HorizontalNeighbors<'a> {
    pub west: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),   // -X
    pub east: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),   // +X
    pub north: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),  // -Z
    pub south: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),  // +Z
    pub nw: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),     // -X, -Z
    pub ne: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),     // +X, -Z
    pub sw: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),     // -X, +Z
    pub se: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),     // +X, +Z
}

/// Optional wrapper enforcing the WAIT-FOR-8 contract (SPECS Step 3 §3).
/// A section meshes ONLY when all 8 horizontal columns exist; returns `None` if any is missing.
#[derive(Default)]
pub struct OptionalHorizontalNeighbors<'a> {
    pub west: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
    pub east: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
    pub north: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
    pub south: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
    pub nw: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
    pub ne: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
    pub sw: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
    pub se: Option<(&'a ChunkVoxelColumn, &'a ChunkLightColumn)>,
}

impl<'a> OptionalHorizontalNeighbors<'a> {
    /// Returns `Some(HorizontalNeighbors)` if and only if all 8 neighbors are present.
    pub fn as_complete(&self) -> Option<HorizontalNeighbors<'a>> {
        Some(HorizontalNeighbors {
            west: self.west?,
            east: self.east?,
            north: self.north?,
            south: self.south?,
            nw: self.nw?,
            ne: self.ne?,
            sw: self.sw?,
            se: self.se?,
        })
    }
}
