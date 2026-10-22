//! Pure CPU Greedy Mesher for 16³ voxel sections.
//!
//! Converts chunk sections and light fields into compact 64-bit PackedVoxelVertex
//! triangle buffers with 4-corner AO, sealed-corner light blending, quad diagonal flip,
//! and 8-neighbor halo boundary access according to SPECS.md Step 3.

use crate::light::{ChunkLightColumn, ChunkVoxelColumn, MAX_Y, MIN_Y, SECTION_COUNT, SECTION_SIZE};
use crate::registry::{
    BlockRegistry, BlockStateEntry, CULL_LEAVES, CULL_OPAQUE, FACE_DOWN, FACE_EAST, FACE_NORTH,
    FACE_SOUTH, FACE_UP, FACE_WEST, MESH_CUBE, MESH_EMPTY, MESH_FLUID,
};
use crate::PackedVoxelVertex;

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
        assert!(
            section_idx < SECTION_COUNT,
            "section_idx must be < SECTION_COUNT"
        );
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
                let l = neighbors
                    .west
                    .1
                    .get_packed_light(15, world_y, rel_z as usize);
                self.set_voxel(-1, rel_y, rel_z, v);
                self.set_packed_light(-1, rel_y, rel_z, l);
            }

            // Cardinal East (rel_x = 16, rel_z ∈ 0..16) -> East neighbor at x = 0
            for rel_z in 0..16 {
                let v = neighbors.east.0.get_state(0, world_y, rel_z as usize);
                let l = neighbors
                    .east
                    .1
                    .get_packed_light(0, world_y, rel_z as usize);
                self.set_voxel(16, rel_y, rel_z, v);
                self.set_packed_light(16, rel_y, rel_z, l);
            }

            // Cardinal North (rel_z = -1, rel_x ∈ 0..16) -> North neighbor at z = 15
            for rel_x in 0..16 {
                let v = neighbors.north.0.get_state(rel_x as usize, world_y, 15);
                let l = neighbors
                    .north
                    .1
                    .get_packed_light(rel_x as usize, world_y, 15);
                self.set_voxel(rel_x, rel_y, -1, v);
                self.set_packed_light(rel_x, rel_y, -1, l);
            }

            // Cardinal South (rel_z = 16, rel_x ∈ 0..16) -> South neighbor at z = 0
            for rel_x in 0..16 {
                let v = neighbors.south.0.get_state(rel_x as usize, world_y, 0);
                let l = neighbors
                    .south
                    .1
                    .get_packed_light(rel_x as usize, world_y, 0);
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
    pub west: (&'a ChunkVoxelColumn, &'a ChunkLightColumn), // -X
    pub east: (&'a ChunkVoxelColumn, &'a ChunkLightColumn), // +X
    pub north: (&'a ChunkVoxelColumn, &'a ChunkLightColumn), // -Z
    pub south: (&'a ChunkVoxelColumn, &'a ChunkLightColumn), // +Z
    pub nw: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),   // -X, -Z
    pub ne: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),   // +X, -Z
    pub sw: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),   // -X, +Z
    pub se: (&'a ChunkVoxelColumn, &'a ChunkLightColumn),   // +X, +Z
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

/// Key identifying mergeable face attributes for 2D greedy meshing (SPECS §5, R7).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FaceMergeKey {
    pub base_layer: u16,
    pub tint_mode: u8,
    pub uv_rot: u8,
    pub pass_bucket: u8, // 0 = opaque/cutout (render_layer < 3), 1 = translucent (render_layer == 3)
    pub corner_ao: [u8; 4],
    pub corner_sky: [u8; 4],
    pub corner_block: [u8; 4],
}

/// Result of meshing a single 16³ chunk section into pass streams (Requirement R7).
#[derive(Clone, Debug, Default)]
pub struct MeshResult {
    pub opaque: Vec<PackedVoxelVertex>, // Pass 1: LAYER_SOLID, LAYER_CUTOUT_MIPPED, LAYER_CUTOUT
    pub translucent: Vec<PackedVoxelVertex>, // Pass 2: LAYER_TRANSLUCENT
}

impl MeshResult {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn quad_count(&self) -> usize {
        (self.opaque.len() + self.translucent.len()) / 6
    }

    #[inline(always)]
    pub fn vertex_count(&self) -> usize {
        self.opaque.len() + self.translucent.len()
    }

    /// Combined view of all vertices for testing or single-buffer dispatch.
    pub fn combined(&self) -> Vec<PackedVoxelVertex> {
        let mut v = Vec::with_capacity(self.opaque.len() + self.translucent.len());
        v.extend_from_slice(&self.opaque);
        v.extend_from_slice(&self.translucent);
        v
    }
}

/// Evaluates face culling between current block and neighbor along normal (SPECS §2).
///
/// Rules:
/// 1. Opaque neighbor (`neighbor.is_opaque == 1`) always culls.
/// 2. Empty neighbor (`neighbor.mesh_type == MESH_EMPTY`) never culls.
/// 3. If current block is opaque, it renders against non-opaque neighbors.
/// 4. Leaves family: never cull against any LeavesBlock (render internal canopy).
/// 5. Same non-opaque block (water|water, lava|lava, glass|glass, ice|ice): cull internal faces.
/// 6. Different non-opaque blocks (water vs lava, glass vs ice): render both faces.
#[inline(always)]
pub fn should_cull_face(curr: &BlockStateEntry, neighbor: &BlockStateEntry) -> bool {
    if neighbor.is_opaque != 0 {
        return true;
    }

    if neighbor.mesh_type == MESH_EMPTY {
        return false;
    }

    if curr.is_opaque != 0 {
        return false;
    }

    let curr_cc = curr.cull_class();
    let neighbor_cc = neighbor.cull_class();

    if curr_cc == CULL_LEAVES && neighbor_cc == CULL_LEAVES {
        return false;
    }

    if curr_cc != CULL_OPAQUE && curr_cc == neighbor_cc {
        return true;
    }

    false
}

/// Computes (ao, sky, block) for a vertex corner using the sealed-corner lighting rule (SPECS §4).
#[inline(always)]
pub fn compute_corner(
    halo: &SectionHalo,
    reg: &BlockRegistry,
    center: (i32, i32, i32),
    side1: (i32, i32, i32),
    side2: (i32, i32, i32),
    corner: (i32, i32, i32),
) -> (u8, u8, u8) {
    let s1_state = reg
        .get_state(halo.get_voxel(side1.0, side1.1, side1.2) as u32)
        .unwrap_or_else(|| reg.get_state_unchecked(0));
    let s2_state = reg
        .get_state(halo.get_voxel(side2.0, side2.1, side2.2) as u32)
        .unwrap_or_else(|| reg.get_state_unchecked(0));
    let c_state = reg
        .get_state(halo.get_voxel(corner.0, corner.1, corner.2) as u32)
        .unwrap_or_else(|| reg.get_state_unchecked(0));

    let s1_opaque = s1_state.is_opaque != 0;
    let s2_opaque = s2_state.is_opaque != 0;
    let c_opaque = c_state.is_opaque != 0;

    let ao = if s1_opaque && s2_opaque {
        0
    } else {
        3 - (s1_opaque as u8 + s2_opaque as u8 + c_opaque as u8)
    };

    let center_light = halo.get_packed_light(center.0, center.1, center.2);
    let side1_light = halo.get_packed_light(side1.0, side1.1, side1.2);
    let side2_light = halo.get_packed_light(side2.0, side2.1, side2.2);

    let corner_light = if s1_opaque && s2_opaque {
        // Sealed corner: substitute by first side (side1)
        side1_light
    } else {
        halo.get_packed_light(corner.0, corner.1, corner.2)
    };

    let center_sky = (center_light >> 4) & 0x0F;
    let center_block = center_light & 0x0F;

    let (s1_sky, s1_blk, s2_sky, s2_blk, c_sky, c_blk) = if center_sky > 2 || center_block > 2 {
        let fix_sample = |sample: u8| -> (u8, u8) {
            if sample == 0 {
                (center_sky, center_block)
            } else {
                let sky = (sample >> 4) & 0x0F;
                let block = sample & 0x0F;
                let final_sky = if sky == 0 { center_sky } else { sky };
                (final_sky, block)
            }
        };
        let (s1_s, s1_b) = fix_sample(side1_light);
        let (s2_s, s2_b) = fix_sample(side2_light);
        let (c_s, c_b) = fix_sample(corner_light);
        (s1_s, s1_b, s2_s, s2_b, c_s, c_b)
    } else {
        (
            (side1_light >> 4) & 0x0F,
            side1_light & 0x0F,
            (side2_light >> 4) & 0x0F,
            side2_light & 0x0F,
            (corner_light >> 4) & 0x0F,
            corner_light & 0x0F,
        )
    };

    let sum_sky = (center_sky as u16) + (s1_sky as u16) + (s2_sky as u16) + (c_sky as u16);
    let sum_block = (center_block as u16) + (s1_blk as u16) + (s2_blk as u16) + (c_blk as u16);

    let sky = (sum_sky >> 2) as u8;
    let block = (sum_block >> 2) as u8;

    (ao, sky, block)
}

#[inline(always)]
pub fn emit_quad(
    out: &mut Vec<PackedVoxelVertex>,
    v0: (u32, u32, u32),
    v1: (u32, u32, u32),
    v2: (u32, u32, u32),
    v3: (u32, u32, u32),
    normal_id: u32,
    key: &FaceMergeKey,
) {
    let brightness = |idx: usize| -> u32 {
        let ao = key.corner_ao[idx] as u32;
        let sky = key.corner_sky[idx] as u32;
        let blk = key.corner_block[idx] as u32;
        (ao << 5) + sky.max(blk)
    };

    let b0 = brightness(0);
    let b1 = brightness(1);
    let b2 = brightness(2);
    let b3 = brightness(3);

    let d02 = b0 + b2;
    let d13 = b1 + b3;

    let flip = if d02 > d13 {
        false
    } else if d02 < d13 {
        true
    } else {
        let l02 = (key.corner_sky[0].max(key.corner_block[0]) as u32)
            + (key.corner_sky[2].max(key.corner_block[2]) as u32);
        let l13 = (key.corner_sky[1].max(key.corner_block[1]) as u32)
            + (key.corner_sky[3].max(key.corner_block[3]) as u32);
        l02 > l13
    };

    let make_vert = |pos: (u32, u32, u32), corner_idx: usize| -> PackedVoxelVertex {
        let mut vert = PackedVoxelVertex::new(
            pos.0,
            pos.1,
            pos.2,
            normal_id,
            0,
            key.uv_rot as u32,
            key.corner_block[corner_idx] as u32,
            key.corner_sky[corner_idx] as u32,
            key.corner_ao[corner_idx] as u32,
            key.base_layer as u32,
        );
        vert.set_registry_tint_mode(key.tint_mode);
        vert
    };

    if !flip {
        // Tri 1: V0, V1, V2
        // Tri 2: V0, V2, V3
        out.push(make_vert(v0, 0));
        out.push(make_vert(v1, 1));
        out.push(make_vert(v2, 2));

        out.push(make_vert(v0, 0));
        out.push(make_vert(v2, 2));
        out.push(make_vert(v3, 3));
    } else {
        // Tri 1: V1, V2, V3
        // Tri 2: V1, V3, V0
        out.push(make_vert(v1, 1));
        out.push(make_vert(v2, 2));
        out.push(make_vert(v3, 3));

        out.push(make_vert(v1, 1));
        out.push(make_vert(v3, 3));
        out.push(make_vert(v0, 0));
    }
}

#[inline(always)]
fn greedy_sweep<F>(mask: &[Option<FaceMergeKey>; 256], out: &mut MeshResult, mut emit: F)
where
    F: FnMut(&mut MeshResult, usize, usize, usize, usize, &FaceMergeKey),
{
    let mut visited = [false; 256];
    for v in 0..16 {
        for u in 0..16 {
            let idx = v * 16 + u;
            if visited[idx] {
                continue;
            }
            let Some(key) = mask[idx] else {
                continue;
            };

            // 1. Expand along U (width)
            let mut w = 1;
            while u + w < 16 {
                let next_idx = v * 16 + (u + w);
                if !visited[next_idx] && mask[next_idx] == Some(key) {
                    w += 1;
                } else {
                    break;
                }
            }

            // 2. Expand along V (height)
            let mut h = 1;
            'expand_v: while v + h < 16 {
                for du in 0..w {
                    let next_idx = (v + h) * 16 + (u + du);
                    if visited[next_idx] || mask[next_idx] != Some(key) {
                        break 'expand_v;
                    }
                }
                h += 1;
            }

            // 3. Mark merged cells visited
            for dv in 0..h {
                for du in 0..w {
                    visited[(v + dv) * 16 + (u + du)] = true;
                }
            }

            // 4. Emit the quad for [u, u+w], [v, v+h]
            emit(out, u, u + w, v, v + h, &key);
        }
    }
}

/// Meshes a 16³ chunk section using its extracted 18³ halo buffer.
pub fn mesh_section(halo: &SectionHalo, reg: &BlockRegistry) -> MeshResult {
    let mut result = MeshResult::default();
    mesh_section_into(halo, reg, &mut result);
    result
}

/// Meshes a 16³ chunk section into an existing reusable `MeshResult`.
pub fn mesh_section_into(halo: &SectionHalo, reg: &BlockRegistry, out: &mut MeshResult) {
    out.opaque.clear();
    out.translucent.clear();

    // 0: FACE_UP (+Y)
    for y in 0..16 {
        let mut mask = [None; 256];
        for z in 0..16 {
            for x in 0..16 {
                let v_id = halo.get_voxel(x, y, z);
                let state = reg
                    .get_state(v_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if state.mesh_type != MESH_CUBE && state.mesh_type != MESH_FLUID {
                    continue;
                }
                let face = state.faces[FACE_UP];
                if face.base_layer >= 2048 {
                    continue;
                }
                let n_id = halo.get_voxel(x, y + 1, z);
                let n_state = reg
                    .get_state(n_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if should_cull_face(state, n_state) {
                    continue;
                }

                let center = (x, y + 1, z);
                let c0 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y + 1, z),
                    (x, y + 1, z - 1),
                    (x - 1, y + 1, z - 1),
                );
                let c1 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y + 1, z),
                    (x, y + 1, z + 1),
                    (x - 1, y + 1, z + 1),
                );
                let c2 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y + 1, z),
                    (x, y + 1, z + 1),
                    (x + 1, y + 1, z + 1),
                );
                let c3 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y + 1, z),
                    (x, y + 1, z - 1),
                    (x + 1, y + 1, z - 1),
                );

                let pass_bucket = if state.render_layer() >= 3 { 1 } else { 0 };

                mask[(z as usize) * 16 + (x as usize)] = Some(FaceMergeKey {
                    base_layer: face.base_layer,
                    tint_mode: face.tint_mode,
                    uv_rot: face.uv_rot,
                    pass_bucket,
                    corner_ao: [c0.0, c1.0, c2.0, c3.0],
                    corner_sky: [c0.1, c1.1, c2.1, c3.1],
                    corner_block: [c0.2, c1.2, c2.2, c3.2],
                });
            }
        }
        greedy_sweep(&mask, out, |out, u0, u1, v0, v1, key| {
            let target = if key.pass_bucket == 1 {
                &mut out.translucent
            } else {
                &mut out.opaque
            };
            let v0_p = (u0 as u32, (y + 1) as u32, v0 as u32);
            let v1_p = (u0 as u32, (y + 1) as u32, v1 as u32);
            let v2_p = (u1 as u32, (y + 1) as u32, v1 as u32);
            let v3_p = (u1 as u32, (y + 1) as u32, v0 as u32);
            emit_quad(target, v0_p, v1_p, v2_p, v3_p, 0, key);
        });
    }

    // 1: FACE_DOWN (-Y)
    for y in 0..16 {
        let mut mask = [None; 256];
        for z in 0..16 {
            for x in 0..16 {
                let v_id = halo.get_voxel(x, y, z);
                let state = reg
                    .get_state(v_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if state.mesh_type != MESH_CUBE && state.mesh_type != MESH_FLUID {
                    continue;
                }
                let face = state.faces[FACE_DOWN];
                if face.base_layer >= 2048 {
                    continue;
                }
                let n_id = halo.get_voxel(x, y - 1, z);
                let n_state = reg
                    .get_state(n_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if should_cull_face(state, n_state) {
                    continue;
                }

                let center = (x, y - 1, z);
                let c0 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y - 1, z),
                    (x, y - 1, z + 1),
                    (x - 1, y - 1, z + 1),
                );
                let c1 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y - 1, z),
                    (x, y - 1, z - 1),
                    (x - 1, y - 1, z - 1),
                );
                let c2 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y - 1, z),
                    (x, y - 1, z - 1),
                    (x + 1, y - 1, z - 1),
                );
                let c3 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y - 1, z),
                    (x, y - 1, z + 1),
                    (x + 1, y - 1, z + 1),
                );

                let pass_bucket = if state.render_layer() >= 3 { 1 } else { 0 };

                mask[(z as usize) * 16 + (x as usize)] = Some(FaceMergeKey {
                    base_layer: face.base_layer,
                    tint_mode: face.tint_mode,
                    uv_rot: face.uv_rot,
                    pass_bucket,
                    corner_ao: [c0.0, c1.0, c2.0, c3.0],
                    corner_sky: [c0.1, c1.1, c2.1, c3.1],
                    corner_block: [c0.2, c1.2, c2.2, c3.2],
                });
            }
        }
        greedy_sweep(&mask, out, |out, u0, u1, v0, v1, key| {
            let target = if key.pass_bucket == 1 {
                &mut out.translucent
            } else {
                &mut out.opaque
            };
            let v0_p = (u0 as u32, y as u32, v1 as u32);
            let v1_p = (u0 as u32, y as u32, v0 as u32);
            let v2_p = (u1 as u32, y as u32, v0 as u32);
            let v3_p = (u1 as u32, y as u32, v1 as u32);
            emit_quad(target, v0_p, v1_p, v2_p, v3_p, 1, key);
        });
    }

    // 2: FACE_SOUTH (+Z)
    for z in 0..16 {
        let mut mask = [None; 256];
        for y in 0..16 {
            for x in 0..16 {
                let v_id = halo.get_voxel(x, y, z);
                let state = reg
                    .get_state(v_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if state.mesh_type != MESH_CUBE && state.mesh_type != MESH_FLUID {
                    continue;
                }
                let face = state.faces[FACE_SOUTH];
                if face.base_layer >= 2048 {
                    continue;
                }
                let n_id = halo.get_voxel(x, y, z + 1);
                let n_state = reg
                    .get_state(n_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if should_cull_face(state, n_state) {
                    continue;
                }

                let center = (x, y, z + 1);
                let c0 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z + 1),
                    (x, y + 1, z + 1),
                    (x - 1, y + 1, z + 1),
                );
                let c1 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z + 1),
                    (x, y - 1, z + 1),
                    (x - 1, y - 1, z + 1),
                );
                let c2 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z + 1),
                    (x, y - 1, z + 1),
                    (x + 1, y - 1, z + 1),
                );
                let c3 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z + 1),
                    (x, y + 1, z + 1),
                    (x + 1, y + 1, z + 1),
                );

                let pass_bucket = if state.render_layer() >= 3 { 1 } else { 0 };

                mask[(y as usize) * 16 + (x as usize)] = Some(FaceMergeKey {
                    base_layer: face.base_layer,
                    tint_mode: face.tint_mode,
                    uv_rot: face.uv_rot,
                    pass_bucket,
                    corner_ao: [c0.0, c1.0, c2.0, c3.0],
                    corner_sky: [c0.1, c1.1, c2.1, c3.1],
                    corner_block: [c0.2, c1.2, c2.2, c3.2],
                });
            }
        }
        greedy_sweep(&mask, out, |out, u0, u1, v0, v1, key| {
            let target = if key.pass_bucket == 1 {
                &mut out.translucent
            } else {
                &mut out.opaque
            };
            let v0_p = (u0 as u32, v1 as u32, (z + 1) as u32);
            let v1_p = (u0 as u32, v0 as u32, (z + 1) as u32);
            let v2_p = (u1 as u32, v0 as u32, (z + 1) as u32);
            let v3_p = (u1 as u32, v1 as u32, (z + 1) as u32);
            emit_quad(target, v0_p, v1_p, v2_p, v3_p, 2, key);
        });
    }

    // 3: FACE_NORTH (-Z)
    for z in 0..16 {
        let mut mask = [None; 256];
        for y in 0..16 {
            for x in 0..16 {
                let v_id = halo.get_voxel(x, y, z);
                let state = reg
                    .get_state(v_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if state.mesh_type != MESH_CUBE && state.mesh_type != MESH_FLUID {
                    continue;
                }
                let face = state.faces[FACE_NORTH];
                if face.base_layer >= 2048 {
                    continue;
                }
                let n_id = halo.get_voxel(x, y, z - 1);
                let n_state = reg
                    .get_state(n_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if should_cull_face(state, n_state) {
                    continue;
                }

                let center = (x, y, z - 1);
                let c0 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z - 1),
                    (x, y + 1, z - 1),
                    (x + 1, y + 1, z - 1),
                );
                let c1 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z - 1),
                    (x, y - 1, z - 1),
                    (x + 1, y - 1, z - 1),
                );
                let c2 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z - 1),
                    (x, y - 1, z - 1),
                    (x - 1, y - 1, z - 1),
                );
                let c3 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z - 1),
                    (x, y + 1, z - 1),
                    (x - 1, y + 1, z - 1),
                );

                let pass_bucket = if state.render_layer() >= 3 { 1 } else { 0 };

                mask[(y as usize) * 16 + (x as usize)] = Some(FaceMergeKey {
                    base_layer: face.base_layer,
                    tint_mode: face.tint_mode,
                    uv_rot: face.uv_rot,
                    pass_bucket,
                    corner_ao: [c0.0, c1.0, c2.0, c3.0],
                    corner_sky: [c0.1, c1.1, c2.1, c3.1],
                    corner_block: [c0.2, c1.2, c2.2, c3.2],
                });
            }
        }
        greedy_sweep(&mask, out, |out, u0, u1, v0, v1, key| {
            let target = if key.pass_bucket == 1 {
                &mut out.translucent
            } else {
                &mut out.opaque
            };
            let v0_p = (u1 as u32, v1 as u32, z as u32);
            let v1_p = (u1 as u32, v0 as u32, z as u32);
            let v2_p = (u0 as u32, v0 as u32, z as u32);
            let v3_p = (u0 as u32, v1 as u32, z as u32);
            emit_quad(target, v0_p, v1_p, v2_p, v3_p, 3, key);
        });
    }

    // 4: FACE_EAST (+X)
    for x in 0..16 {
        let mut mask = [None; 256];
        for y in 0..16 {
            for z in 0..16 {
                let v_id = halo.get_voxel(x, y, z);
                let state = reg
                    .get_state(v_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if state.mesh_type != MESH_CUBE && state.mesh_type != MESH_FLUID {
                    continue;
                }
                let face = state.faces[FACE_EAST];
                if face.base_layer >= 2048 {
                    continue;
                }
                let n_id = halo.get_voxel(x + 1, y, z);
                let n_state = reg
                    .get_state(n_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if should_cull_face(state, n_state) {
                    continue;
                }

                let center = (x + 1, y, z);
                let c0 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z + 1),
                    (x + 1, y + 1, z),
                    (x + 1, y + 1, z + 1),
                );
                let c1 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z + 1),
                    (x + 1, y - 1, z),
                    (x + 1, y - 1, z + 1),
                );
                let c2 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z - 1),
                    (x + 1, y - 1, z),
                    (x + 1, y - 1, z - 1),
                );
                let c3 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x + 1, y, z - 1),
                    (x + 1, y + 1, z),
                    (x + 1, y + 1, z - 1),
                );

                let pass_bucket = if state.render_layer() >= 3 { 1 } else { 0 };

                mask[(y as usize) * 16 + (z as usize)] = Some(FaceMergeKey {
                    base_layer: face.base_layer,
                    tint_mode: face.tint_mode,
                    uv_rot: face.uv_rot,
                    pass_bucket,
                    corner_ao: [c0.0, c1.0, c2.0, c3.0],
                    corner_sky: [c0.1, c1.1, c2.1, c3.1],
                    corner_block: [c0.2, c1.2, c2.2, c3.2],
                });
            }
        }
        greedy_sweep(&mask, out, |out, u0, u1, v0, v1, key| {
            let target = if key.pass_bucket == 1 {
                &mut out.translucent
            } else {
                &mut out.opaque
            };
            let v0_p = ((x + 1) as u32, v1 as u32, u1 as u32);
            let v1_p = ((x + 1) as u32, v0 as u32, u1 as u32);
            let v2_p = ((x + 1) as u32, v0 as u32, u0 as u32);
            let v3_p = ((x + 1) as u32, v1 as u32, u0 as u32);
            emit_quad(target, v0_p, v1_p, v2_p, v3_p, 4, key);
        });
    }

    // 5: FACE_WEST (-X)
    for x in 0..16 {
        let mut mask = [None; 256];
        for y in 0..16 {
            for z in 0..16 {
                let v_id = halo.get_voxel(x, y, z);
                let state = reg
                    .get_state(v_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if state.mesh_type != MESH_CUBE && state.mesh_type != MESH_FLUID {
                    continue;
                }
                let face = state.faces[FACE_WEST];
                if face.base_layer >= 2048 {
                    continue;
                }
                let n_id = halo.get_voxel(x - 1, y, z);
                let n_state = reg
                    .get_state(n_id as u32)
                    .unwrap_or_else(|| reg.get_state_unchecked(0));
                if should_cull_face(state, n_state) {
                    continue;
                }

                let center = (x - 1, y, z);
                let c0 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z - 1),
                    (x - 1, y + 1, z),
                    (x - 1, y + 1, z - 1),
                );
                let c1 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z - 1),
                    (x - 1, y - 1, z),
                    (x - 1, y - 1, z - 1),
                );
                let c2 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z + 1),
                    (x - 1, y - 1, z),
                    (x - 1, y - 1, z + 1),
                );
                let c3 = compute_corner(
                    halo,
                    reg,
                    center,
                    (x - 1, y, z + 1),
                    (x - 1, y + 1, z),
                    (x - 1, y + 1, z + 1),
                );

                let pass_bucket = if state.render_layer() >= 3 { 1 } else { 0 };

                mask[(y as usize) * 16 + (z as usize)] = Some(FaceMergeKey {
                    base_layer: face.base_layer,
                    tint_mode: face.tint_mode,
                    uv_rot: face.uv_rot,
                    pass_bucket,
                    corner_ao: [c0.0, c1.0, c2.0, c3.0],
                    corner_sky: [c0.1, c1.1, c2.1, c3.1],
                    corner_block: [c0.2, c1.2, c2.2, c3.2],
                });
            }
        }
        greedy_sweep(&mask, out, |out, u0, u1, v0, v1, key| {
            let target = if key.pass_bucket == 1 {
                &mut out.translucent
            } else {
                &mut out.opaque
            };
            let v0_p = (x as u32, v1 as u32, u0 as u32);
            let v1_p = (x as u32, v0 as u32, u0 as u32);
            let v2_p = (x as u32, v0 as u32, u1 as u32);
            let v3_p = (x as u32, v1 as u32, u1 as u32);
            emit_quad(target, v0_p, v1_p, v2_p, v3_p, 5, key);
        });
    }
}

/// Convenience helper to populate halo and mesh section in one call.
pub fn mesh_section_with_neighbors(
    section_idx: usize,
    center_voxels: &ChunkVoxelColumn,
    center_light: &ChunkLightColumn,
    neighbors: &HorizontalNeighbors,
    reg: &BlockRegistry,
    air_state_id: u16,
) -> MeshResult {
    let mut halo = SectionHalo::new();
    halo.populate(
        section_idx,
        center_voxels,
        center_light,
        neighbors,
        air_state_id,
    );
    mesh_section(&halo, reg)
}
