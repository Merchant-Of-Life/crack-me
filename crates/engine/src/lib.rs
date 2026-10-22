use bytemuck::{Pod, Zeroable};

pub mod bfs;
pub mod light;
pub mod registry;

pub use bfs::{recompute_light_column, LightComputeResult, NeighborChunkLights, CARDINAL_DIRECTIONS};
pub use light::{
    section_index_to_coords, section_voxel_index, ChunkLightColumn, ChunkVoxelColumn, LightSection,
    MAX_Y, MIN_Y, SECTION_COUNT, SECTION_SIZE, SECTION_VOXELS,
};
pub use registry::{
    BlockIndexEntry, BlockRegistry, BlockStateEntry, FaceInfo, RegistryError, FACE_DOWN, FACE_EAST,
    FACE_NORTH, FACE_SOUTH, FACE_UP, FACE_WEST, MESH_COMPLEX, MESH_CROSS, MESH_CUBE, MESH_EMPTY,
    MESH_FLUID,
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
}
