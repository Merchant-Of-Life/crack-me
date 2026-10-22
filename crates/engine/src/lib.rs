use bytemuck::{Pod, Zeroable};

pub mod bfs;
pub mod light;
pub mod mesher;
pub mod registry;
pub mod render;

pub use bfs::{recompute_light_column, LightComputeResult, NeighborChunkLights, CARDINAL_DIRECTIONS};
pub use light::{
    section_index_to_coords, section_voxel_index, ChunkLightColumn, ChunkVoxelColumn, LightSection,
    MAX_Y, MIN_Y, SECTION_COUNT, SECTION_SIZE, SECTION_VOXELS,
};
pub use mesher::{
    compute_corner, emit_quad, mesh_section, mesh_section_into, mesh_section_with_neighbors,
    should_cull_face, FaceMergeKey, HorizontalNeighbors, MeshResult, OptionalHorizontalNeighbors,
    SectionHalo, HALO_SIZE, HALO_VOXELS,
};
pub use registry::{
    BlockIndexEntry, BlockRegistry, BlockStateEntry, FaceInfo, RegistryError, CULL_LAVA,
    CULL_LEAVES, CULL_OPAQUE, CULL_TRANSPARENT_START, CULL_WATER, FACE_DOWN, FACE_EAST,
    FACE_NORTH, FACE_SOUTH, FACE_UP, FACE_WEST, LAYER_CUTOUT, LAYER_CUTOUT_MIPPED, LAYER_SOLID,
    LAYER_TRANSLUCENT, MESH_COMPLEX, MESH_CROSS, MESH_CUBE, MESH_EMPTY, MESH_FLUID,
};

/// Exactly 64-bit packed voxel vertex according to SPECS.md
///
/// Word 0:
/// Bits 0..4 (5b): Local X (0..=16)
/// Bits 5..9 (5b): Local Y (0..=16)
/// Bits 10..14 (5b): Local Z (0..=16)
/// Bits 15..17 (3b): Normal ID (0..5: +Y, -Y, +Z, -Z, +X, -X)
/// Bits 18..19 (2b): Tint mode (0..3: None, Grass, Foliage, Water)
/// Bits 20..31 (12b): Spare / UV rotation / flags
///
/// Word 1:
/// Bits 0..3 (4b): Block Light (0..15)
/// Bits 4..7 (4b): Sky Light (0..15)
/// Bits 8..9 (2b): Ambient Occlusion (0..3)
/// Bits 10..20 (11b): Base Texture Layer (0..2047)
/// Bits 21..31 (11b): Spare
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct PackedVoxelVertex {
    pub word0: u32,
    pub word1: u32,
}

impl PackedVoxelVertex {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn new(
        x: u32,
        y: u32,
        z: u32,
        normal_id: u32,
        tint_mode: u32,
        flags: u32,
        block_light: u32,
        sky_light: u32,
        ao: u32,
        base_layer: u32,
    ) -> Self {
        debug_assert!(x <= 16, "X coordinate must be <= 16");
        debug_assert!(y <= 16, "Y coordinate must be <= 16");
        debug_assert!(z <= 16, "Z coordinate must be <= 16");
        debug_assert!(normal_id < 6, "Normal ID must be in 0..5");
        debug_assert!(tint_mode < 4, "Tint mode must be in 0..3");
        debug_assert!(flags < 4096, "Flags must be < 4096");
        debug_assert!(block_light < 16, "Block light must be < 16");
        debug_assert!(sky_light < 16, "Sky light must be < 16");
        debug_assert!(ao < 4, "AO must be in 0..3");
        debug_assert!(base_layer < 2048, "Base layer must be < 2048");

        let word0 = (x & 0x1F)
            | ((y & 0x1F) << 5)
            | ((z & 0x1F) << 10)
            | ((normal_id & 0x07) << 15)
            | ((tint_mode & 0x03) << 18)
            | ((flags & 0x0FFF) << 20);

        let word1 = (block_light & 0x0F)
            | ((sky_light & 0x0F) << 4)
            | ((ao & 0x03) << 8)
            | ((base_layer & 0x07FF) << 10);

        Self { word0, word1 }
    }

    #[inline(always)]
    pub fn x(&self) -> u32 {
        self.word0 & 0x1F
    }

    #[inline(always)]
    pub fn y(&self) -> u32 {
        (self.word0 >> 5) & 0x1F
    }

    #[inline(always)]
    pub fn z(&self) -> u32 {
        (self.word0 >> 10) & 0x1F
    }

    #[inline(always)]
    pub fn normal_id(&self) -> u32 {
        (self.word0 >> 15) & 0x07
    }

    #[inline(always)]
    pub fn tint_mode(&self) -> u32 {
        (self.word0 >> 18) & 0x03
    }

    #[inline(always)]
    pub fn flags(&self) -> u32 {
        (self.word0 >> 20) & 0x0FFF
    }

    #[inline(always)]
    pub fn block_light(&self) -> u32 {
        self.word1 & 0x0F
    }

    #[inline(always)]
    pub fn sky_light(&self) -> u32 {
        (self.word1 >> 4) & 0x0F
    }

    #[inline(always)]
    pub fn ao(&self) -> u32 {
        (self.word1 >> 8) & 0x03
    }

    #[inline(always)]
    pub fn base_layer(&self) -> u32 {
        (self.word1 >> 10) & 0x07FF
    }

    #[inline(always)]
    pub fn set_dry_foliage(&mut self, val: bool) {
        if val {
            self.word1 |= 1 << 21;
        } else {
            self.word1 &= !(1 << 21);
        }
    }

    #[inline(always)]
    pub fn dry_foliage(&self) -> bool {
        (self.word1 & (1 << 21)) != 0
    }

    #[inline(always)]
    pub fn set_registry_tint_mode(&mut self, tint_mode: u8) {
        if tint_mode == 4 {
            self.word0 &= !(0x03 << 18);
            self.set_dry_foliage(true);
        } else {
            self.word0 = (self.word0 & !(0x03 << 18)) | (((tint_mode as u32) & 0x03) << 18);
            self.set_dry_foliage(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_oracle_registry() -> BlockRegistry {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let path = Path::new(&manifest_dir).join("../../assets/registry/block_registry.bin");
        BlockRegistry::from_file(&path).expect("block_registry.bin must load from assets/registry/")
    }

    fn state_id_of(reg: &BlockRegistry, name: &str) -> u16 {
        let block = reg.get_block(name).unwrap_or_else(|| panic!("block '{}' not found in registry", name));
        block.state_start as u16
    }

    #[test]
    fn test_vertex_packing_roundtrip_64bit() {
        assert_eq!(std::mem::size_of::<PackedVoxelVertex>(), 8);
        assert_eq!(std::mem::align_of::<PackedVoxelVertex>(), 4);

        let v = PackedVoxelVertex::new(16, 8, 16, 4, 2, 512, 14, 15, 3, 1842);
        assert_eq!(v.x(), 16);
        assert_eq!(v.y(), 8);
        assert_eq!(v.z(), 16);
        assert_eq!(v.normal_id(), 4);
        assert_eq!(v.tint_mode(), 2);
        assert_eq!(v.flags(), 512);
        assert_eq!(v.block_light(), 14);
        assert_eq!(v.sky_light(), 15);
        assert_eq!(v.ao(), 3);
        assert_eq!(v.base_layer(), 1842);
    }

    #[test]
    fn test_light_section_size_and_indexing() {
        assert_eq!(std::mem::size_of::<LightSection>(), 4096);
        assert_eq!(std::mem::align_of::<LightSection>(), 1);

        let mut sec = LightSection::new();
        // Check roundtrip for coordinates
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let idx = section_voxel_index(x, y, z);
                    let (dx, dy, dz) = section_index_to_coords(idx);
                    assert_eq!((x, y, z), (dx, dy, dz));

                    let block_val = ((x + y + z) % 16) as u8;
                    let sky_val = ((x * 2 + y + z * 3) % 16) as u8;
                    sec.set_block_light(x, y, z, block_val);
                    sec.set_sky_light(x, y, z, sky_val);

                    assert_eq!(sec.get_block_light(x, y, z), block_val);
                    assert_eq!(sec.get_sky_light(x, y, z), sky_val);
                }
            }
        }
    }

    #[test]
    fn test_get_raw_brightness_invariant() {
        let mut sec = LightSection::new();
        // LevelLightEngine.java:142-146: max(blockLight, skyLight - skyDampen)
        sec.set_block_light(0, 0, 0, 8);
        sec.set_sky_light(0, 0, 0, 15);

        // Broad daylight (sky_dampen = 0)
        assert_eq!(sec.get_raw_brightness(0, 0, 0, 0), 15);
        // Dimming by 4 -> sky becomes 11, max(8, 11) = 11
        assert_eq!(sec.get_raw_brightness(0, 0, 0, 4), 11);
        // Night / heavy storm (sky_dampen = 11) -> sky becomes 4, max(8, 4) = 8
        assert_eq!(sec.get_raw_brightness(0, 0, 0, 11), 8);
        // Complete darkness (sky_dampen = 15) -> sky becomes 0, block light dominates = 8
        assert_eq!(sec.get_raw_brightness(0, 0, 0, 15), 8);

        // When block light is 0 and sky is 10
        sec.set_block_light(1, 1, 1, 0);
        sec.set_sky_light(1, 1, 1, 10);
        assert_eq!(sec.get_raw_brightness(1, 1, 1, 3), 7);
        assert_eq!(sec.get_raw_brightness(1, 1, 1, 12), 0);
    }

    #[test]
    fn test_registry_loading_and_properties() {
        let reg = load_oracle_registry();
        assert_eq!(reg.block_count(), 1288);
        assert_eq!(reg.state_count(), 1812);

        // acacia_button is row 0
        let b0 = reg.get_block("acacia_button").unwrap();
        assert_eq!(b0.state_start, 0);

        // air is registered and has 0 emission, 0 filter
        let air = reg.get_block("air").unwrap();
        let air_state = reg.get_state(air.state_start).unwrap();
        assert_eq!(air_state.light_emission, 0);
        assert_eq!(air_state.light_filter, 0);
        assert_eq!(air_state.is_opaque, 0);

        // stone has filter 15
        let stone = reg.get_block("stone").unwrap();
        let stone_state = reg.get_state(stone.state_start).unwrap();
        assert_eq!(stone_state.light_filter, 15);
        assert_eq!(stone_state.is_opaque, 1);
    }

    #[test]
    fn test_added_6d_furnace_lit_vs_unlit() {
        let reg = load_oracle_registry();
        let furnace = reg.get_block("furnace").unwrap();
        assert_eq!(furnace.state_count, 8); // 4 facing x 2 lit

        let states: Vec<_> = (furnace.state_start..furnace.state_start + furnace.state_count)
            .map(|id| reg.get_state(id).unwrap())
            .collect();

        let lit_count = states.iter().filter(|s| s.light_emission == 13).count();
        let unlit_count = states.iter().filter(|s| s.light_emission == 0).count();
        assert_eq!(lit_count, 4, "4 lit furnace states with emission 13");
        assert_eq!(unlit_count, 4, "4 unlit furnace states with emission 0");
    }

    #[test]
    fn test_added_6a_lava_lake_attenuate_by_destination() {
        let reg = load_oracle_registry();
        let air_id = state_id_of(&reg, "air");
        let stone_id = state_id_of(&reg, "stone");
        let lava_id = state_id_of(&reg, "lava");

        // Verify registry invariants first:
        assert_eq!(reg.light_emission(lava_id as u32), 15);
        assert_eq!(reg.light_filter(lava_id as u32), 1);
        assert_eq!(reg.light_filter(stone_id as u32), 15);
        assert_eq!(reg.light_filter(air_id as u32), 0);

        let mut voxels = ChunkVoxelColumn::new(air_id);
        // Create a 2x2 lava lake at y = 0, surrounded by stone below (y = -1) and walls (x=0,3, z=0,3)
        // Basin at y = 0: x in 1..=2, z in 1..=2 are lava
        for x in 1..=2 {
            for z in 1..=2 {
                voxels.set_state(x, 0, z, lava_id);
            }
        }
        // Place stone floor at y = -1
        for x in 0..4 {
            for z in 0..4 {
                voxels.set_state(x, -1, z, stone_id);
            }
        }
        // Place stone wall on the east at x = 3, y = 0
        for z in 1..=2 {
            voxels.set_state(3, 0, z, stone_id);
        }

        let res = recompute_light_column(&voxels, &reg, None, None);
        let light = res.light;

        // 1. All lava cells are 15 (own emission = 15; neighbor lava attenuates 15-1=14 < 15)
        for x in 1..=2 {
            for z in 1..=2 {
                assert_eq!(light.get_block_light(x, 0, z), 15, "lava cell at ({}, 0, {}) must be 15", x, z);
            }
        }

        // 2. Air neighbor above lava at y = 1 gets 14 (destination air attenuation = max(1, 0) = 1)
        for x in 1..=2 {
            for z in 1..=2 {
                assert_eq!(light.get_block_light(x, 1, z), 14, "air above lava at ({}, 1, {}) must be 14", x, z);
            }
        }

        // 3. Air neighbor to the west at x = 0, y = 0 gets 14
        assert_eq!(light.get_block_light(0, 0, 1), 14, "air neighbor next to lava must be 14");

        // 4. Stone neighbor at x = 3, y = 0 gets 0 (proves attenuate-by-destination:
        // stone destination filter is 15 -> 15 - 15 = 0; light does NOT leak into stone)
        assert_eq!(light.get_block_light(3, 0, 1), 0, "stone neighbor at (3, 0, 1) must be 0");
        assert_eq!(light.get_block_light(1, -1, 1), 0, "stone floor below lava must be 0");
    }

    #[test]
    fn test_added_6b_idempotence() {
        let reg = load_oracle_registry();
        let air_id = state_id_of(&reg, "air");
        let stone_id = state_id_of(&reg, "stone");
        let torch_id = state_id_of(&reg, "torch");

        let mut voxels = ChunkVoxelColumn::new(air_id);
        voxels.set_state(8, 64, 8, torch_id);
        voxels.set_state(8, 63, 8, stone_id);

        let run1 = recompute_light_column(&voxels, &reg, None, None);
        let run2 = recompute_light_column(&voxels, &reg, None, Some(&run1.light));

        // Exactly byte-identical across all 24 sections
        for s in 0..SECTION_COUNT {
            assert_eq!(
                run1.light.sections[s].raw,
                run2.light.sections[s].raw,
                "section {} must be byte-identical on recomputation",
                s
            );
        }
        // Since run2 passed run1 as prev_light and no bytes changed, dirty_mask must be 0!
        assert_eq!(run2.dirty_mask, 0, "idempotent recompute must produce dirty_mask == 0");
    }

    #[test]
    fn test_added_6c_dirty_mask_flags_only_changed_sections() {
        let reg = load_oracle_registry();
        let air_id = state_id_of(&reg, "air");
        let torch_id = state_id_of(&reg, "torch");
        let stone_id = state_id_of(&reg, "stone");

        // Start with a clear world (air everywhere down to bedrock at y = -64)
        let mut voxels = ChunkVoxelColumn::new(air_id);
        // Roof across section 10 (y = 96..111) to block skylight from sections below
        for x in 0..16 {
            for z in 0..16 {
                voxels.set_state(x, 100, z, stone_id);
            }
        }

        let base_res = recompute_light_column(&voxels, &reg, None, None);

        // Now place a torch in section 2 (y = -32, inside section 2 which spans y = -32..-17)
        // Torch light = 14; with max range 14 blocks, light will stay entirely within section 2
        // (y in -32..-18)
        let (torch_sec, _) = ChunkLightColumn::section_for_y(-25).unwrap();
        assert_eq!(torch_sec, 2);
        voxels.set_state(8, -25, 8, torch_id);

        let edit_res = recompute_light_column(&voxels, &reg, None, Some(&base_res.light));

        // Torch light changed section 2
        assert_ne!(edit_res.dirty_mask & (1 << torch_sec), 0, "section 2 must be flagged dirty");
        // Sections above section 10 (e.g. section 15) did not change at all
        assert_eq!(edit_res.dirty_mask & (1 << 15), 0, "section 15 was untouched and must not be dirty");
    }

    #[test]
    fn test_ice_family_splitting_and_glass_transparency() {
        let reg = load_oracle_registry();
        let air_id = state_id_of(&reg, "air");
        let ice_id = state_id_of(&reg, "ice");
        let frosted_ice_id = state_id_of(&reg, "frosted_ice");
        let packed_ice_id = state_id_of(&reg, "packed_ice");
        let blue_ice_id = state_id_of(&reg, "blue_ice");
        let glass_id = state_id_of(&reg, "glass");
        let tinted_glass_id = state_id_of(&reg, "tinted_glass");

        assert_eq!(reg.light_filter(ice_id as u32), 1);
        assert_eq!(reg.light_filter(frosted_ice_id as u32), 1);
        assert_eq!(reg.light_filter(packed_ice_id as u32), 15);
        assert_eq!(reg.light_filter(blue_ice_id as u32), 15);
        assert_eq!(reg.light_filter(glass_id as u32), 0);
        assert_eq!(reg.light_filter(tinted_glass_id as u32), 15);

        let stone_id = state_id_of(&reg, "stone");
        let mut voxels = ChunkVoxelColumn::new(air_id);

        // Ice shaft at (1, 1): surrounded by stone at (0, 1), (2, 1), (1, 0), (1, 2)
        // so no horizontal sunlight enters from neighboring columns.
        for y in 98..=100 {
            voxels.set_state(0, y, 1, stone_id);
            voxels.set_state(2, y, 1, stone_id);
            voxels.set_state(1, y, 0, stone_id);
            voxels.set_state(1, y, 2, stone_id);
            voxels.set_state(1, y, 1, ice_id);
        }

        // Column (5, 5): 1 layer of packed_ice at y = 100
        voxels.set_state(5, 100, 5, packed_ice_id);

        // Column (10, 10): 10 layers of clear glass at y = 90..=100
        for y in 90..=100 {
            voxels.set_state(10, y, 10, glass_id);
        }

        // Column (15, 15): 1 layer of tinted_glass at y = 100
        voxels.set_state(15, 100, 15, tinted_glass_id);

        let res = recompute_light_column(&voxels, &reg, None, None);
        let light = res.light;

        // Clear glass: direct sky passes through all 10 layers with ZERO attenuation
        for y in 90..=100 {
            assert_eq!(light.get_sky_light(10, y, 10), 15, "inside clear glass at y={y} must be 15");
        }
        assert_eq!(light.get_sky_light(10, 89, 10), 15, "air beneath clear glass must be 15");

        // Packed ice: dampening = 15 -> direct sky stopped at y=101; entering packed ice at y=100 gets 15-15=0
        assert_eq!(light.get_sky_light(5, 100, 5), 0, "inside packed ice at y=100 must be 0");

        // Tinted glass: dampening = 15 -> entering tinted glass at y=100 gets 15-15=0
        assert_eq!(light.get_sky_light(15, 100, 15), 0, "inside tinted glass at y=100 must be 0");

        // Regular ice in isolated shaft: dampening = 1 -> direct sky stopped at y=101;
        // entering ice at y=100 gets 15-1=14; y=99 gets 14-1=13; y=98 gets 13-1=12
        assert_eq!(light.get_sky_light(1, 100, 1), 14, "ice at y=100 receives 14");
        assert_eq!(light.get_sky_light(1, 99, 1), 13, "ice at y=99 receives 13");
        assert_eq!(light.get_sky_light(1, 98, 1), 12, "ice at y=98 receives 12");
    }

    #[test]
    fn test_cross_chunk_boundary_light_seeding() {
        let reg = load_oracle_registry();
        let air_id = state_id_of(&reg, "air");
        let torch_id = state_id_of(&reg, "torch"); // emission = 14

        // Chunk A (West neighbor) has a torch placed at x = 15, y = 0, z = 8
        let mut voxels_a = ChunkVoxelColumn::new(air_id);
        voxels_a.set_state(15, 0, 8, torch_id);
        let res_a = recompute_light_column(&voxels_a, &reg, None, None);

        // Torch at x=15 has emission 14
        assert_eq!(res_a.light.get_block_light(15, 0, 8), 14);

        // Chunk B (Current chunk, located East of A):
        // Has air everywhere. West neighbor is Chunk A.
        let voxels_b = ChunkVoxelColumn::new(air_id);
        let nbrs = NeighborChunkLights {
            west: Some(&res_a.light),
            ..Default::default()
        };

        let res_b = recompute_light_column(&voxels_b, &reg, Some(&nbrs), None);
        let light_b = res_b.light;

        // At x = 0 in Chunk B, light crosses the boundary:
        // Neighbor at x=15 was 14; destination air dampens by 1 -> 14 - 1 = 13
        assert_eq!(light_b.get_block_light(0, 0, 8), 13, "light crosses boundary from west chunk to x=0");
        assert_eq!(light_b.get_block_light(1, 0, 8), 12, "light continues into x=1");
        assert_eq!(light_b.get_block_light(2, 0, 8), 11, "light continues into x=2");
    }

    #[test]
    fn test_above_world_sky_clamp() {
        let reg = load_oracle_registry();
        let air_id = state_id_of(&reg, "air");
        let stone_id = state_id_of(&reg, "stone");

        let mut voxels = ChunkVoxelColumn::new(air_id);
        // Place stone block at y = 319 (topmost world block)
        voxels.set_state(8, 319, 8, stone_id);

        let res = recompute_light_column(&voxels, &reg, None, None);
        let light = res.light;

        // Top face neighbor at y = 320 (y >= MAX_Y) must read sky 15 / block 0
        assert_eq!(light.get_sky_light(8, 320, 8), 15, "y >= MAX_Y must yield sky 15");
        assert_eq!(light.get_block_light(8, 320, 8), 0, "y >= MAX_Y must yield block 0");
        assert_eq!(light.get_raw_brightness(8, 320, 8, 0), 15);
        assert_eq!(light.get_raw_brightness(8, 320, 8, 4), 11);

        // Way above world at y = 500
        assert_eq!(light.get_sky_light(8, 500, 8), 15);
        assert_eq!(light.get_block_light(8, 500, 8), 0);

        // Below world (y < MIN_Y, y = -65) stays 0/0
        assert_eq!(light.get_sky_light(8, -65, 8), 0, "y < MIN_Y must yield sky 0");
        assert_eq!(light.get_block_light(8, -65, 8), 0, "y < MIN_Y must yield block 0");
        assert_eq!(light.get_raw_brightness(8, -65, 8, 0), 0);
    }

    #[test]
    fn test_dry_foliage_word1_spare_bit_packing() {
        // TINT_DRY_FOLIAGE (4) is carried by Word 1 spare bit 0 (bit 21).
        // Word 0 tint_mode retains 0 (or lower bits), while Word 1 bit 21 flags dry_foliage.
        let mut v = PackedVoxelVertex::new(8, 8, 8, 0, 0, 0, 14, 15, 3, 100);
        assert_eq!(v.word1 & (1 << 21), 0, "spare bit 0 must initially be 0");

        // Flag dry foliage:
        v.word1 |= 1 << 21;
        assert_ne!(v.word1 & (1 << 21), 0, "dry foliage bit 21 set");
        // Verify other fields in Word 1 remain undisturbed
        assert_eq!(v.block_light(), 14);
        assert_eq!(v.sky_light(), 15);
        assert_eq!(v.ao(), 3);
        assert_eq!(v.base_layer(), 100);
    }

    #[test]
    fn test_set_registry_tint_mode_helper() {
        let mut v = PackedVoxelVertex::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

        for mode in 0..=3 {
            v.set_registry_tint_mode(mode);
            assert_eq!(v.tint_mode(), mode as u32);
            assert!(!v.dry_foliage(), "tint_mode 0..=3 must clear dry_foliage");
            assert_eq!(v.word1 & (1 << 21), 0);
        }

        v.set_registry_tint_mode(4);
        assert_eq!(v.tint_mode(), 0, "tint_mode == 4 must route Word 0 tint to 0");
        assert!(v.dry_foliage(), "tint_mode == 4 must set dry_foliage bit");
        assert_ne!(v.word1 & (1 << 21), 0);

        // Switching back clears dry_foliage and sets Word 0 tint
        v.set_registry_tint_mode(1);
        assert_eq!(v.tint_mode(), 1);
        assert!(!v.dry_foliage());
        assert_eq!(v.word1 & (1 << 21), 0);

        // Direct set_dry_foliage toggle
        v.set_dry_foliage(true);
        assert!(v.dry_foliage());
        v.set_dry_foliage(false);
        assert!(!v.dry_foliage());
    }

    #[test]
    fn test_halo_indexing_and_bounds() {
        let mut halo = SectionHalo::new();
        // Check corners and center
        halo.set_voxel(-1, -1, -1, 101);
        halo.set_packed_light(-1, -1, -1, 0x12);
        assert_eq!(halo.get_voxel(-1, -1, -1), 101);
        assert_eq!(halo.get_packed_light(-1, -1, -1), 0x12);
        assert_eq!(halo.get_sky_light(-1, -1, -1), 1);
        assert_eq!(halo.get_block_light(-1, -1, -1), 2);

        halo.set_voxel(16, 16, 16, 202);
        halo.set_packed_light(16, 16, 16, 0xEF);
        assert_eq!(halo.get_voxel(16, 16, 16), 202);
        assert_eq!(halo.get_packed_light(16, 16, 16), 0xEF);
        assert_eq!(halo.get_sky_light(16, 16, 16), 14);
        assert_eq!(halo.get_block_light(16, 16, 16), 15);

        halo.set_voxel(0, 0, 0, 303);
        assert_eq!(halo.get_voxel(0, 0, 0), 303);
    }

    #[test]
    fn test_halo_wait_for_8_contract() {
        let cv = ChunkVoxelColumn::new(0);
        let cl = ChunkLightColumn::default();

        let mut opt = OptionalHorizontalNeighbors::default();
        assert!(opt.as_complete().is_none(), "empty neighbors must return None");

        opt.west = Some((&cv, &cl));
        opt.east = Some((&cv, &cl));
        opt.north = Some((&cv, &cl));
        opt.south = Some((&cv, &cl));
        opt.nw = Some((&cv, &cl));
        opt.ne = Some((&cv, &cl));
        opt.sw = Some((&cv, &cl));
        assert!(opt.as_complete().is_none(), "7/8 neighbors present must return None");

        opt.se = Some((&cv, &cl));
        assert!(opt.as_complete().is_some(), "all 8 neighbors present must return Some");
    }

    #[test]
    fn test_halo_8_neighbors_sampling_and_clamps() {
        let air = 23u16;
        let mut center_v = ChunkVoxelColumn::new(air);
        let mut center_l = ChunkLightColumn::default();

        let mut west_v = ChunkVoxelColumn::new(air);
        let mut west_l = ChunkLightColumn::default();

        let mut east_v = ChunkVoxelColumn::new(air);
        let mut east_l = ChunkLightColumn::default();

        let mut north_v = ChunkVoxelColumn::new(air);
        let mut north_l = ChunkLightColumn::default();

        let mut south_v = ChunkVoxelColumn::new(air);
        let mut south_l = ChunkLightColumn::default();

        let mut nw_v = ChunkVoxelColumn::new(air);
        let mut nw_l = ChunkLightColumn::default();

        let mut ne_v = ChunkVoxelColumn::new(air);
        let mut ne_l = ChunkLightColumn::default();

        let mut sw_v = ChunkVoxelColumn::new(air);
        let mut sw_l = ChunkLightColumn::default();

        let mut se_v = ChunkVoxelColumn::new(air);
        let mut se_l = ChunkLightColumn::default();

        // Target test section: section 1 (Y in [-48, -32), world_y for local 0 is -48)
        let sec_idx = 1;
        let y_test = -48; // rel_y = 0

        // Interior
        center_v.set_state(5, y_test, 5, 500);
        center_l.set_packed_light(5, y_test, 5, 0x55);

        // Cardinal West (-X: rel_x = -1, z = 7) -> sampled from West chunk (15, y, 7)
        west_v.set_state(15, y_test, 7, 601);
        west_l.set_packed_light(15, y_test, 7, 0x61);

        // Cardinal East (+X: rel_x = 16, z = 7) -> sampled from East chunk (0, y, 7)
        east_v.set_state(0, y_test, 7, 602);
        east_l.set_packed_light(0, y_test, 7, 0x62);

        // Cardinal North (-Z: x = 7, rel_z = -1) -> sampled from North chunk (7, y, 15)
        north_v.set_state(7, y_test, 15, 603);
        north_l.set_packed_light(7, y_test, 15, 0x63);

        // Cardinal South (+Z: x = 7, rel_z = 16) -> sampled from South chunk (7, y, 0)
        south_v.set_state(7, y_test, 0, 604);
        south_l.set_packed_light(7, y_test, 0, 0x64);

        // Diagonal NW (rel_x = -1, rel_z = -1) -> NW chunk (15, y, 15)
        nw_v.set_state(15, y_test, 15, 701);
        nw_l.set_packed_light(15, y_test, 15, 0x71);

        // Diagonal NE (rel_x = 16, rel_z = -1) -> NE chunk (0, y, 15)
        ne_v.set_state(0, y_test, 15, 702);
        ne_l.set_packed_light(0, y_test, 15, 0x72);

        // Diagonal SW (rel_x = -1, rel_z = 16) -> SW chunk (15, y, 0)
        sw_v.set_state(15, y_test, 0, 703);
        sw_l.set_packed_light(15, y_test, 0, 0x73);

        // Diagonal SE (rel_x = 16, rel_z = 16) -> SE chunk (0, y, 0)
        se_v.set_state(0, y_test, 0, 704);
        se_l.set_packed_light(0, y_test, 0, 0x74);

        let neighbors = HorizontalNeighbors {
            west: (&west_v, &west_l),
            east: (&east_v, &east_l),
            north: (&north_v, &north_l),
            south: (&south_v, &south_l),
            nw: (&nw_v, &nw_l),
            ne: (&ne_v, &ne_l),
            sw: (&sw_v, &sw_l),
            se: (&se_v, &se_l),
        };

        let mut halo = SectionHalo::new();
        halo.populate(sec_idx, &center_v, &center_l, &neighbors, air);

        // Verify interior
        assert_eq!(halo.get_voxel(5, 0, 5), 500);
        assert_eq!(halo.get_packed_light(5, 0, 5), 0x55);

        // Verify cardinals
        assert_eq!(halo.get_voxel(-1, 0, 7), 601);
        assert_eq!(halo.get_packed_light(-1, 0, 7), 0x61);

        assert_eq!(halo.get_voxel(16, 0, 7), 602);
        assert_eq!(halo.get_packed_light(16, 0, 7), 0x62);

        assert_eq!(halo.get_voxel(7, 0, -1), 603);
        assert_eq!(halo.get_packed_light(7, 0, -1), 0x63);

        assert_eq!(halo.get_voxel(7, 0, 16), 604);
        assert_eq!(halo.get_packed_light(7, 0, 16), 0x64);

        // Verify 4 diagonals
        assert_eq!(halo.get_voxel(-1, 0, -1), 701);
        assert_eq!(halo.get_packed_light(-1, 0, -1), 0x71);

        assert_eq!(halo.get_voxel(16, 0, -1), 702);
        assert_eq!(halo.get_packed_light(16, 0, -1), 0x72);

        assert_eq!(halo.get_voxel(-1, 0, 16), 703);
        assert_eq!(halo.get_packed_light(-1, 0, 16), 0x73);

        assert_eq!(halo.get_voxel(16, 0, 16), 704);
        assert_eq!(halo.get_packed_light(16, 0, 16), 0x74);

        // Verify Section 0 bottom floor clamp (y < -64)
        let mut halo_bottom = SectionHalo::new();
        halo_bottom.populate(0, &center_v, &center_l, &neighbors, air);
        for z in -1..=16 {
            for x in -1..=16 {
                assert_eq!(halo_bottom.get_voxel(x, -1, z), air, "floor voxel clamp");
                assert_eq!(halo_bottom.get_packed_light(x, -1, z), 0x00, "floor light clamp");
            }
        }

        // Verify Section 23 top ceiling clamp (y >= 320)
        let mut halo_top = SectionHalo::new();
        halo_top.populate(23, &center_v, &center_l, &neighbors, air);
        for z in -1..=16 {
            for x in -1..=16 {
                assert_eq!(halo_top.get_voxel(x, 16, z), air, "ceiling voxel clamp");
                assert_eq!(halo_top.get_packed_light(x, 16, z), 0xF0, "ceiling light clamp (sky 15)");
            }
        }
    }

    #[test]
    fn test_cull_classes_pale_oak_fluids_glass_ice() {
        let reg = load_oracle_registry();
        let pale_oak_id = state_id_of(&reg, "pale_oak_leaves");
        let oak_id = state_id_of(&reg, "oak_leaves");
        let water_id = state_id_of(&reg, "water");
        let lava_id = state_id_of(&reg, "lava");
        let glass_id = state_id_of(&reg, "glass");
        let ice_id = state_id_of(&reg, "ice");
        let stone_id = state_id_of(&reg, "stone");
        let air_id = state_id_of(&reg, "air");

        let pale_oak = reg.get_state_unchecked(pale_oak_id as u32);
        let oak = reg.get_state_unchecked(oak_id as u32);
        let water = reg.get_state_unchecked(water_id as u32);
        let lava = reg.get_state_unchecked(lava_id as u32);
        let glass = reg.get_state_unchecked(glass_id as u32);
        let ice = reg.get_state_unchecked(ice_id as u32);
        let stone = reg.get_state_unchecked(stone_id as u32);
        let air = reg.get_state_unchecked(air_id as u32);

        // 1. Pale oak leaves are LeavesBlock (cull_class = CULL_LEAVES) and do NOT cull internal canopy
        assert_eq!(pale_oak.cull_class(), CULL_LEAVES);
        assert_eq!(oak.cull_class(), CULL_LEAVES);
        assert!(!should_cull_face(pale_oak, pale_oak), "pale_oak | pale_oak must not cull");
        assert!(!should_cull_face(pale_oak, oak), "pale_oak | oak must not cull");

        // 2. Water & lava are distinct cull classes and render BOTH boundaries
        assert_eq!(water.cull_class(), CULL_WATER);
        assert_eq!(lava.cull_class(), CULL_LAVA);
        assert!(!should_cull_face(water, lava), "water | lava renders both");
        assert!(!should_cull_face(lava, water), "lava | water renders both");

        // 3. Same fluid culls internal faces
        assert!(should_cull_face(water, water), "water | water must cull");
        assert!(should_cull_face(lava, lava), "lava | lava must cull");

        // 4. Glass & Ice are distinct transparent cull classes and render both boundaries
        assert!(glass.cull_class() >= CULL_TRANSPARENT_START);
        assert!(ice.cull_class() >= CULL_TRANSPARENT_START);
        assert_ne!(glass.cull_class(), ice.cull_class());
        assert!(!should_cull_face(glass, ice), "glass | ice renders both");
        assert!(!should_cull_face(ice, glass), "ice | glass renders both");

        // 5. Identical transparent blocks cull
        assert!(should_cull_face(glass, glass), "glass | glass must cull");
        assert!(should_cull_face(ice, ice), "ice | ice must cull");

        // 6. Opaque stone interactions
        assert!(should_cull_face(stone, stone), "stone | stone culls");
        assert!(!should_cull_face(stone, air), "stone | air does not cull");
        assert!(!should_cull_face(stone, glass), "stone | glass does not cull (stone visible through glass)");
        assert!(should_cull_face(glass, stone), "glass | stone culls (stone occludes glass face)");
    }

    #[test]
    fn test_corner_lighting_sealed_and_glowstone() {
        let reg = load_oracle_registry();
        let glowstone_id = state_id_of(&reg, "glowstone");
        let stone_id = state_id_of(&reg, "stone");
        let air_id = state_id_of(&reg, "air");

        // 1. Glowstone-wall corner keeps 15
        let mut halo = SectionHalo::new();
        halo.set_voxel(5, 1, 5, air_id);
        halo.set_packed_light(5, 1, 5, 0x0F);
        halo.set_voxel(4, 1, 5, glowstone_id);
        halo.set_packed_light(4, 1, 5, 0x0F);
        halo.set_voxel(5, 1, 4, air_id);
        halo.set_packed_light(5, 1, 4, 0x0F);
        halo.set_voxel(4, 1, 4, air_id);
        halo.set_packed_light(4, 1, 4, 0x0F);

        let (_ao, sky, blk) = compute_corner(&halo, &reg, (5, 1, 5), (4, 1, 5), (5, 1, 4), (4, 1, 4));
        assert_eq!(sky, 0);
        assert_eq!(blk, 15, "glowstone-wall corner keeps 15");

        // 2. Torch-near-wall not darkened:
        halo.set_voxel(5, 1, 5, air_id);
        halo.set_packed_light(5, 1, 5, 0x0E);
        halo.set_voxel(4, 1, 5, stone_id);
        halo.set_packed_light(4, 1, 5, 0x00);
        halo.set_voxel(5, 1, 4, air_id);
        halo.set_packed_light(5, 1, 4, 0x0E);
        halo.set_voxel(4, 1, 4, air_id);
        halo.set_packed_light(4, 1, 4, 0x0E);

        let (_ao, _sky, blk) = compute_corner(&halo, &reg, (5, 1, 5), (4, 1, 5), (5, 1, 4), (4, 1, 4));
        assert_eq!(blk, 14, "torch-near-wall not darkened by solid stone wall");
    }

    #[test]
    fn test_greedy_mesh_solid_16_cube_merges_to_6_quads() {
        let reg = load_oracle_registry();
        let stone_id = state_id_of(&reg, "stone");
        let air_id = state_id_of(&reg, "air");

        let mut halo = SectionHalo::new();
        // Fill 16³ interior with stone
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    halo.set_voxel(x, y, z, stone_id);
                    halo.set_packed_light(x, y, z, 0x00);
                }
            }
        }
        // Fill halo border with air and uniform sky 15
        for y in -1..=16 {
            for z in -1..=16 {
                for x in -1..=16 {
                    if x == -1 || x == 16 || y == -1 || y == 16 || z == -1 || z == 16 {
                        halo.set_voxel(x, y, z, air_id);
                        halo.set_packed_light(x, y, z, 0xF0);
                    }
                }
            }
        }

        let mesh = mesh_section(&halo, &reg);
        // Exactly 6 faces * 1 quad each = 6 quads = 36 vertices
        assert_eq!(mesh.quad_count(), 6, "16³ solid cube must merge to exactly 6 quads");
        assert_eq!(mesh.vertex_count(), 36, "16³ solid cube must produce exactly 36 vertices");

        // Verify all 6 directions are present
        let mut normal_counts = [0usize; 6];
        for v in &mesh.opaque {
            normal_counts[v.normal_id() as usize] += 1;
        }
        for (i, count) in normal_counts.iter().enumerate() {
            assert_eq!(*count, 6, "normal {} must have exactly 1 quad (6 vertices)", i);
        }
    }

    #[test]
    fn test_greedy_mesh_preserves_torch_falloff_steps() {
        let reg = load_oracle_registry();
        let stone_id = state_id_of(&reg, "stone");
        let air_id = state_id_of(&reg, "air");

        let mut halo = SectionHalo::new();
        // Floor of stone at y = 0
        for z in 0..16 {
            for x in 0..16 {
                halo.set_voxel(x, 0, z, stone_id);
            }
        }
        // Rest is air
        for y in 1..=16 {
            for z in -1..=16 {
                for x in -1..=16 {
                    halo.set_voxel(x, y, z, air_id);
                }
            }
        }
        for z in -1..=16 {
            for x in -1..=16 {
                halo.set_voxel(x, -1, z, stone_id);
            }
        }

        // Apply light step at y = 1: x in -1..8 gets block 14, x in 8..=16 gets block 10
        for z in -1..=16 {
            for x in -1..8 {
                halo.set_packed_light(x, 1, z, 0x0E);
            }
            for x in 8..=16 {
                halo.set_packed_light(x, 1, z, 0x0A);
            }
        }

        let mesh = mesh_section(&halo, &reg);
        // Find UP faces (normal_id == 0)
        let up_vertices: Vec<_> = mesh.opaque.iter().filter(|v| v.normal_id() == 0).collect();
        // UP faces must NOT merge into a single quad because the light step at x=8 splits them
        assert!(up_vertices.len() > 6, "UP faces must not merge across light step boundary");
    }

    #[test]
    fn test_quad_diagonal_flip_on_asymmetric_ao() {
        let key_asymmetric = FaceMergeKey {
            base_layer: 100,
            tint_mode: 0,
            uv_rot: 0,
            pass_bucket: 0,
            corner_ao: [1, 3, 3, 3], // corner 0 in shadow
            corner_sky: [15, 15, 15, 15],
            corner_block: [0, 0, 0, 0],
        };

        let mut verts_flipped = Vec::new();
        let v0 = (0, 1, 0);
        let v1 = (0, 1, 1);
        let v2 = (1, 1, 1);
        let v3 = (1, 1, 0);

        // When corner 0 has ao = 1, d02 = 1 + 3 = 4, d13 = 3 + 3 = 6.
        // d02 < d13 => flip is true! Triangles: (V1, V2, V3) and (V1, V3, V0).
        // First emitted vertex must be V1.
        emit_quad(&mut verts_flipped, v0, v1, v2, v3, 0, &key_asymmetric);
        assert_eq!(verts_flipped.len(), 6);
        assert_eq!(verts_flipped[0].x(), v1.0);
        assert_eq!(verts_flipped[0].y(), v1.1);
        assert_eq!(verts_flipped[0].z(), v1.2);

        // Symmetric case: all ao = 3. d02 == d13, l02 == l13 => flip is false!
        // Triangles: (V0, V1, V2) and (V0, V2, V3).
        // First emitted vertex must be V0.
        let key_symmetric = FaceMergeKey {
            corner_ao: [3, 3, 3, 3],
            ..key_asymmetric
        };
        let mut verts_normal = Vec::new();
        emit_quad(&mut verts_normal, v0, v1, v2, v3, 0, &key_symmetric);
        assert_eq!(verts_normal.len(), 6);
        assert_eq!(verts_normal[0].x(), v0.0);
        assert_eq!(verts_normal[0].y(), v0.1);
        assert_eq!(verts_normal[0].z(), v0.2);
    }

    #[test]
    fn test_stained_glass_distinct_color_culling() {
        let reg = load_oracle_registry();
        let red_id = state_id_of(&reg, "red_stained_glass");
        let blue_id = state_id_of(&reg, "blue_stained_glass");
        let green_id = state_id_of(&reg, "green_stained_glass");
        let clear_id = state_id_of(&reg, "glass");
        let grate_id = state_id_of(&reg, "copper_grate");
        let waxed_grate_id = state_id_of(&reg, "waxed_copper_grate");

        let red = reg.get_state_unchecked(red_id as u32);
        let blue = reg.get_state_unchecked(blue_id as u32);
        let green = reg.get_state_unchecked(green_id as u32);
        let clear = reg.get_state_unchecked(clear_id as u32);
        let grate = reg.get_state_unchecked(grate_id as u32);
        let waxed_grate = reg.get_state_unchecked(waxed_grate_id as u32);

        // 1. Transparent blocks must have cull_class >= CULL_TRANSPARENT_START
        assert!(red.cull_class() >= CULL_TRANSPARENT_START);
        assert!(blue.cull_class() >= CULL_TRANSPARENT_START);
        assert!(green.cull_class() >= CULL_TRANSPARENT_START);
        assert!(clear.cull_class() >= CULL_TRANSPARENT_START);
        assert!(grate.cull_class() >= CULL_TRANSPARENT_START);
        assert!(waxed_grate.cull_class() >= CULL_TRANSPARENT_START);

        // 2. Distinct colors and types have distinct cull identities
        assert_ne!(red.cull_class(), blue.cull_class());
        assert_ne!(red.cull_class(), green.cull_class());
        assert_ne!(red.cull_class(), clear.cull_class());
        assert_ne!(blue.cull_class(), clear.cull_class());

        // 3. Waxed copper grate has distinct identity from unwaxed copper grate (HalfTransparentBlock.is(this))
        assert_ne!(grate.cull_class(), waxed_grate.cull_class());
        assert_eq!(waxed_grate.is_opaque, 0);

        // 4. Same color / block identity culls internal boundary
        assert!(should_cull_face(red, red), "red | red must cull");
        assert!(should_cull_face(blue, blue), "blue | blue must cull");
        assert!(should_cull_face(clear, clear), "clear | clear must cull");
        assert!(should_cull_face(grate, grate), "copper_grate | copper_grate must cull");
        assert!(should_cull_face(waxed_grate, waxed_grate), "waxed_copper_grate | waxed_copper_grate must cull");

        // 5. Different colors / block identities render BOTH boundaries
        assert!(!should_cull_face(red, blue), "red | blue renders both");
        assert!(!should_cull_face(blue, red), "blue | red renders both");
        assert!(!should_cull_face(red, clear), "red | clear renders both");
        assert!(!should_cull_face(clear, red), "clear | red renders both");
        assert!(!should_cull_face(grate, waxed_grate), "copper_grate | waxed_copper_grate renders both");
        assert!(!should_cull_face(waxed_grate, grate), "waxed_copper_grate | copper_grate renders both");
        assert!(!should_cull_face(grate, red), "copper_grate | red renders both");
    }

    #[test]
    fn test_sealed_corner_explicit_side1_substitution() {
        let reg = load_oracle_registry();
        let air_id = state_id_of(&reg, "air");
        let stone_id = state_id_of(&reg, "stone");

        let center = (0, 0, 0);
        let side1 = (-1, 0, 0);
        let side2 = (0, 0, -1);
        let corner = (-1, 0, -1);

        // Pattern: center = 0, side1 = 4, side2 = 8, corner = 15
        let mut halo = SectionHalo::new();
        halo.set_voxel(center.0, center.1, center.2, air_id);
        halo.set_packed_light(center.0, center.1, center.2, 0x00); // 0 sky, 0 block

        halo.set_voxel(side1.0, side1.1, side1.2, stone_id); // opaque side1
        halo.set_packed_light(side1.0, side1.1, side1.2, 0x44); // 4 sky, 4 block

        halo.set_voxel(side2.0, side2.1, side2.2, stone_id); // opaque side2
        halo.set_packed_light(side2.0, side2.1, side2.2, 0x88); // 8 sky, 8 block

        halo.set_voxel(corner.0, corner.1, corner.2, stone_id); // opaque corner
        halo.set_packed_light(corner.0, corner.1, corner.2, 0xFF); // 15 sky, 15 block

        // Case A: Sealed corner (side1 & side2 both opaque):
        // corner must be explicitly substituted by side1 (value 4).
        // sum = center(0) + side1(4) + side2(8) + corner_substituted_by_side1(4) = 16.
        // light = 16 >> 2 = 4.
        // (Note: if corner were unsubstituted (15): sum = 27 >> 2 = 6.
        //        if corner were side2 (8):         sum = 20 >> 2 = 5.
        //        if corner were center (0):        sum = 12 >> 2 = 3.
        // Only side1 substitution mathematically yields exactly 4.)
        let (ao_sealed, sky_sealed, blk_sealed) = compute_corner(&halo, &reg, center, side1, side2, corner);
        assert_eq!(ao_sealed, 0, "sealed corner has ao = 0");
        assert_eq!(sky_sealed, 4, "sealed corner must substitute side1 for sky light");
        assert_eq!(blk_sealed, 4, "sealed corner must substitute side1 for block light");

        // Case B: Unsealed corner (side2 changed to non-opaque air):
        // corner is NOT sealed, so its actual light (15) is retained without substitution.
        // sum = center(0) + side1(4) + side2(8) + corner(15) = 27.
        // light = 27 >> 2 = 6.
        halo.set_voxel(side2.0, side2.1, side2.2, air_id);
        let (ao_unsealed, sky_unsealed, blk_unsealed) = compute_corner(&halo, &reg, center, side1, side2, corner);
        assert_eq!(ao_unsealed, 1, "unsealed corner with 2 opaque blocks has ao = 1");
        assert_eq!(sky_unsealed, 6, "unsealed corner preserves corner light (sum 27 >> 2 = 6)");
        assert_eq!(blk_unsealed, 6, "unsealed corner preserves corner light (sum 27 >> 2 = 6)");
    }

    #[test]
    fn test_face_winding_normals_golden() {
        // Face normal definitions and unit normal vectors:
        // 0: +Y (0, 1, 0)
        // 1: -Y (0, -1, 0)
        // 2: +Z (0, 0, 1)
        // 3: -Z (0, 0, -1)
        // 4: +X (1, 0, 0)
        // 5: -X (-1, 0, 0)
        let expected_normals: [(i32, i32, i32); 6] = [
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (1, 0, 0),
            (-1, 0, 0),
        ];

        // The quad corners (v0, v1, v2, v3) in canonical order for a unit voxel at (0, 0, 0)
        // matching the mesher's coordinate emitter:
        let face_quads: [((u32, u32, u32), (u32, u32, u32), (u32, u32, u32), (u32, u32, u32)); 6] = [
            // 0: FACE_UP (+Y, y = 1)
            ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
            // 1: FACE_DOWN (-Y, y = 0)
            ((0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1)),
            // 2: FACE_SOUTH (+Z, z = 1)
            ((0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
            // 3: FACE_NORTH (-Z, z = 0)
            ((1, 1, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0)),
            // 4: FACE_EAST (+X, x = 1)
            ((1, 1, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0)),
            // 5: FACE_WEST (-X, x = 0)
            ((0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)),
        ];

        let key_normal = FaceMergeKey {
            base_layer: 1,
            tint_mode: 0,
            uv_rot: 0,
            pass_bucket: 0,
            corner_ao: [3, 3, 3, 3], // symmetric -> flip = false
            corner_sky: [15, 15, 15, 15],
            corner_block: [0, 0, 0, 0],
        };

        let key_flipped = FaceMergeKey {
            base_layer: 1,
            tint_mode: 0,
            uv_rot: 0,
            pass_bucket: 0,
            corner_ao: [1, 3, 3, 3], // d02 (4) < d13 (6) -> flip = true
            corner_sky: [15, 15, 15, 15],
            corner_block: [0, 0, 0, 0],
        };

        for (face_idx, &(v0, v1, v2, v3)) in face_quads.iter().enumerate() {
            let exp_n = expected_normals[face_idx];

            for (branch_name, key) in [("normal", &key_normal), ("flipped", &key_flipped)] {
                let mut verts = Vec::with_capacity(6);
                emit_quad(&mut verts, v0, v1, v2, v3, face_idx as u32, key);
                assert_eq!(verts.len(), 6, "face {} ({}) must emit 6 vertices", face_idx, branch_name);

                // Two triangles per quad: (0, 1, 2) and (3, 4, 5)
                for tri_idx in 0..2 {
                    let base = tri_idx * 3;
                    let p0 = (verts[base].x() as i32, verts[base].y() as i32, verts[base].z() as i32);
                    let p1 = (verts[base + 1].x() as i32, verts[base + 1].y() as i32, verts[base + 1].z() as i32);
                    let p2 = (verts[base + 2].x() as i32, verts[base + 2].y() as i32, verts[base + 2].z() as i32);

                    // Cross product (p1 - p0) x (p2 - p0) for CCW front-face normal
                    let e1 = (p1.0 - p0.0, p1.1 - p0.1, p1.2 - p0.2);
                    let e2 = (p2.0 - p0.0, p2.1 - p0.1, p2.2 - p0.2);
                    let nx = e1.1 * e2.2 - e1.2 * e2.1;
                    let ny = e1.2 * e2.0 - e1.0 * e2.2;
                    let nz = e1.0 * e2.1 - e1.1 * e2.0;

                    let dot = nx * exp_n.0 + ny * exp_n.1 + nz * exp_n.2;
                    assert!(
                        dot > 0,
                        "face {} ({}) tri {} winding cross-product ({}, {}, {}) dot expected ({}, {}, {}) = {} (must be > 0)",
                        face_idx, branch_name, tri_idx, nx, ny, nz, exp_n.0, exp_n.1, exp_n.2, dot
                    );
                    assert_eq!((nx, ny, nz), exp_n, "computed normal must exactly match unit normal");
                }
            }
        }
    }

    #[test]
    fn test_render_layers_golden() {
        let reg = load_oracle_registry();

        // 1. stone == LAYER_SOLID (0)
        let stone = reg.get_block("stone").expect("stone in registry");
        let stone_state = reg.get_state(stone.state_start).unwrap();
        assert_eq!(stone_state.render_layer(), LAYER_SOLID, "stone must be LAYER_SOLID");

        // 2. copper_grate == LAYER_CUTOUT (2)
        let copper_grate = reg.get_block("copper_grate").expect("copper_grate in registry");
        let copper_grate_state = reg.get_state(copper_grate.state_start).unwrap();
        assert_eq!(copper_grate_state.render_layer(), LAYER_CUTOUT, "copper_grate must be LAYER_CUTOUT");

        // Weathered / waxed copper grates parity
        for prefix in ["exposed_", "weathered_", "oxidized_", "waxed_", "waxed_exposed_", "waxed_weathered_", "waxed_oxidized_"] {
            let name = format!("{prefix}copper_grate");
            let b = reg.get_block(&name).unwrap_or_else(|| panic!("{name} in registry"));
            let s = reg.get_state(b.state_start).unwrap();
            assert_eq!(s.render_layer(), LAYER_CUTOUT, "{name} must be LAYER_CUTOUT");
        }

        // 3. white_stained_glass == LAYER_TRANSLUCENT (3)
        let white_stained_glass = reg.get_block("white_stained_glass").expect("white_stained_glass in registry");
        let white_stained_glass_state = reg.get_state(white_stained_glass.state_start).unwrap();
        assert_eq!(white_stained_glass_state.render_layer(), LAYER_TRANSLUCENT, "white_stained_glass must be LAYER_TRANSLUCENT");

        // 4. oak_leaves == LAYER_CUTOUT_MIPPED (1)
        let oak_leaves = reg.get_block("oak_leaves").expect("oak_leaves in registry");
        let oak_leaves_state = reg.get_state(oak_leaves.state_start).unwrap();
        assert_eq!(oak_leaves_state.render_layer(), LAYER_CUTOUT_MIPPED, "oak_leaves must be LAYER_CUTOUT_MIPPED");

        // 5. pale_oak_leaves == LAYER_CUTOUT_MIPPED (1)
        let pale_oak_leaves = reg.get_block("pale_oak_leaves").expect("pale_oak_leaves in registry");
        let pale_oak_leaves_state = reg.get_state(pale_oak_leaves.state_start).unwrap();
        assert_eq!(pale_oak_leaves_state.render_layer(), LAYER_CUTOUT_MIPPED, "pale_oak_leaves must be LAYER_CUTOUT_MIPPED");
    }

    #[test]
    fn test_grass_block_side_composite_golden() {
        let reg = load_oracle_registry();
        let grass = reg.get_block("grass_block").expect("grass_block in registry");
        let state = reg.get_state(grass.state_start).unwrap();

        // 1. Top face is grass-tinted (TINT_GRASS = 1)
        assert_eq!(state.faces[FACE_UP].tint_mode, 1, "top face must be TINT_GRASS");

        // 2. Bottom face is dirt (TINT_NONE = 0)
        assert_eq!(state.faces[FACE_DOWN].tint_mode, 0, "bottom face must be TINT_NONE");

        // 3. Side faces must use the bake-added composite layer with tint_mode = 0 [APPROX-RENDERING-09]
        let side_layer = state.faces[FACE_NORTH].base_layer;
        assert_eq!(side_layer, 1843, "grass_block side face must point to composite layer 1843");
        for f in [FACE_NORTH, FACE_SOUTH, FACE_EAST, FACE_WEST] {
            assert_eq!(state.faces[f].base_layer, side_layer, "all 4 side faces must match composite layer");
            assert_eq!(state.faces[f].tint_mode, 0, "side faces must have tint_mode 0 to prevent recoloring dirt [APPROX-RENDERING-09]");
        }
    }

    #[test]
    fn test_adapter_limits_validation_seam() {
        let mut limits = wgpu::Limits::downlevel_defaults();

        // 1. Success when >= 2048
        limits.max_texture_array_layers = 2048;
        assert!(render::validate_adapter_limits(&limits).is_ok());

        limits.max_texture_array_layers = 4096;
        assert!(render::validate_adapter_limits(&limits).is_ok());

        // 2. Descriptive error when < 2048
        limits.max_texture_array_layers = 256;
        let err = render::validate_adapter_limits(&limits).unwrap_err();
        assert!(err.contains("max_texture_array_layers is 256"));
        assert!(err.contains("required >= 2048"));
    }

    #[test]
    fn test_gpu_context_probe() {
        let ctx_res = render::GpuContext::new_headless();
        match ctx_res {
            Ok(ctx) => {
                let info = ctx.adapter.get_info();
                println!("Headless Adapter Info: {:?}", info);
                println!("max_texture_array_layers: {}", ctx.adapter.limits().max_texture_array_layers);
                assert!(ctx.adapter.limits().max_texture_array_layers >= 2048);
            }
            Err(e) => {
                println!("No headless GPU adapter found in environment: {}", e);
            }
        }
    }

    #[test]
    fn test_render_layer_auto_split_stone_and_water() {
        let reg = load_oracle_registry();
        let stone_id = state_id_of(&reg, "stone");
        let water_id = state_id_of(&reg, "water");
        let air_id = state_id_of(&reg, "air");

        let stone_state = reg.get_state(stone_id as u32).unwrap();
        let water_state = reg.get_state(water_id as u32).unwrap();
        assert_eq!(stone_state.render_layer(), LAYER_SOLID, "stone must be LAYER_SOLID (0)");
        assert_eq!(water_state.render_layer(), LAYER_TRANSLUCENT, "water must be LAYER_TRANSLUCENT (3)");

        let mut halo = SectionHalo::new();
        // Section: bottom half (y=0..8) stone, top half (y=8..16) water, surrounded by air
        for z in -1..=16 {
            for x in -1..=16 {
                for y in -1..=16 {
                    if (0..8).contains(&y) && (0..16).contains(&x) && (0..16).contains(&z) {
                        halo.set_voxel(x, y, z, stone_id);
                    } else if (8..16).contains(&y) && (0..16).contains(&x) && (0..16).contains(&z) {
                        halo.set_voxel(x, y, z, water_id);
                    } else {
                        halo.set_voxel(x, y, z, air_id);
                    }
                }
            }
        }

        let mesh = mesh_section(&halo, &reg);

        // Assert both pass streams are populated
        assert!(!mesh.opaque.is_empty(), "opaque stream must contain stone quads");
        assert!(!mesh.translucent.is_empty(), "translucent stream must contain water quads");

        let stone_layer = stone_state.faces[FACE_UP].base_layer;
        let water_layer = water_state.faces[FACE_UP].base_layer;

        // Verify clean segregation: NO water in opaque, NO stone in translucent
        for v in &mesh.opaque {
            assert_eq!(v.base_layer(), stone_layer as u32, "mesh.opaque must only contain stone quads");
        }
        for v in &mesh.translucent {
            assert_eq!(v.base_layer(), water_layer as u32, "mesh.translucent must only contain water quads");
        }

        // Verify total vertices and combined view
        assert_eq!(mesh.vertex_count(), mesh.opaque.len() + mesh.translucent.len());
        assert_eq!(mesh.combined().len(), mesh.vertex_count());
    }
}
