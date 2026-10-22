//! Chunk Light Field BFS flood-fill subsystem.
//!
//! Implements direct sky column calculation, block and sky light BFS flood fill,
//! neighbor boundary seeding, and section dirty tracking according to Minecraft 26.3
//! release-jar mechanics (SPECS.md Step 2.5).
//!
//! ## Cross-Chunk Boundary Equivalence
//! In vanilla Minecraft, sunlight enters the world strictly from above: the sky
//! column is 15 down to the first non-propagating block (`SkyLightEngine`). The
//! above-world clamp (`y >= MAX_Y` -> `sky = 15`) guarantees that the top horizontal
//! surface (y = 319) reads full sunlight without needing a vertical neighbor column.
//! At the bottom of the world (`y < MIN_Y`), void/bedrock is completely unlit (sky = 0,
//! block = 0), so no vertical upward light enters from below.
//!
//! As a result, cross-chunk boundary transfer occurs exclusively across the four
//! cardinal horizontal boundaries (West, East, North, South). Combining downward-sky-15
//! with W/E/N/S boundary seeding is mathematically identical to full 3D boundary
//! propagation in vanilla.
//!
//! Vanilla's no-decrement downward sky flow exists ONLY on per-(x, z) direct vertical
//! columns, which never cross horizontal chunk borders; every horizontal chunk border
//! crossing is lateral and therefore always decremented.

use std::collections::VecDeque;

use crate::light::{
    ChunkLightColumn, ChunkVoxelColumn, MAX_Y, MIN_Y, SECTION_COUNT, SECTION_SIZE, SECTION_VOXELS,
};
use crate::registry::BlockRegistry;

/// 6 Manhattan cardinal directions for 3D grid propagation.
pub const CARDINAL_DIRECTIONS: [(i32, i32, i32); 6] = [
    (0, -1, 0), // -Y (down)
    (0, 1, 0),  // +Y (up)
    (-1, 0, 0), // -X (west)
    (1, 0, 0),  // +X (east)
    (0, 0, -1), // -Z (north)
    (0, 0, 1),  // +Z (south)
];

/// Neighbor chunk columns for cross-chunk boundary light seeding.
#[derive(Default)]
pub struct NeighborChunkLights<'a> {
    pub west: Option<&'a ChunkLightColumn>,  // -X (x = 15 in west neighbor)
    pub east: Option<&'a ChunkLightColumn>,  // +X (x = 0 in east neighbor)
    pub north: Option<&'a ChunkLightColumn>, // -Z (z = 15 in north neighbor)
    pub south: Option<&'a ChunkLightColumn>, // +Z (z = 0 in south neighbor)
}

/// Result of a chunk column light recomputation.
pub struct LightComputeResult {
    pub light: ChunkLightColumn,
    /// Bitmask where bit `s` (0..24) is 1 if section `s` changed relative to `prev_light`.
    pub dirty_mask: u32,
}

#[inline(always)]
fn pack_pos(x: usize, y: i32, z: usize) -> u32 {
    let rel_y = (y - MIN_Y) as u32;
    (x as u32 & 0x0F) | ((z as u32 & 0x0F) << 4) | (rel_y << 8)
}

#[inline(always)]
fn unpack_pos(packed: u32) -> (usize, i32, usize) {
    let x = (packed & 0x0F) as usize;
    let z = ((packed >> 4) & 0x0F) as usize;
    let rel_y = (packed >> 8) as i32;
    let y = rel_y + MIN_Y;
    (x, y, z)
}

/// Recomputes the entire light field for a 24-section chunk column (SPECS approximation 8).
///
/// # Arguments
/// * `voxels` - Chunk voxel states (raw registry state IDs).
/// * `registry` - Frozen block registry oracle (`assets/registry/block_registry.bin`).
/// * `neighbors` - Optional adjacent chunk light columns for cross-boundary light seeding.
/// * `prev_light` - Previous light column, used to compute `dirty_mask` (only sections
///   whose bytes actually changed are flagged).
pub fn recompute_light_column(
    voxels: &ChunkVoxelColumn,
    registry: &BlockRegistry,
    neighbors: Option<&NeighborChunkLights>,
    prev_light: Option<&ChunkLightColumn>,
) -> LightComputeResult {
    let mut light = ChunkLightColumn::new();
    let mut sky_queue: VecDeque<u32> = VecDeque::with_capacity(4096);
    let mut block_queue: VecDeque<u32> = VecDeque::with_capacity(4096);

    // -----------------------------------------------------------------------
    // Step 1: Direct Skylight Column Calculation (Top-Down)
    // -----------------------------------------------------------------------
    // Direct sky column rule (SkyLightEngine lowestSourceY):
    // For each column (x, z), trace top-down. All voxels down to the first
    // non-propagating block (dampening > 0) receive sky light = 15.
    for x in 0..SECTION_SIZE {
        for z in 0..SECTION_SIZE {
            let mut y = MAX_Y - 1;
            while y >= MIN_Y {
                let state_id = voxels.get_state(x, y, z) as u32;
                let filter = registry.light_filter(state_id);

                if filter == 0 {
                    light.set_sky_light(x, y, z, 15);
                    sky_queue.push_back(pack_pos(x, y, z));
                    y -= 1;
                } else {
                    // First non-propagating / attenuating block reached.
                    // Stop the direct 15-column. Further downward and lateral
                    // penetration is resolved via BFS.
                    break;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Sky Light Boundary Seeding from Neighbors
    // -----------------------------------------------------------------------
    if let Some(nbrs) = neighbors {
        seed_sky_boundaries(&mut light, voxels, registry, nbrs, &mut sky_queue);
    }

    // -----------------------------------------------------------------------
    // Step 3: Sky Light BFS Flood-Fill Propagation
    // -----------------------------------------------------------------------
    propagate_bfs_sky(&mut light, voxels, registry, &mut sky_queue);

    // -----------------------------------------------------------------------
    // Step 4: Block Light Emission Seeding
    // -----------------------------------------------------------------------
    for s in 0..SECTION_COUNT {
        let sec_base_y = MIN_Y + (s as i32) * (SECTION_SIZE as i32);
        for idx in 0..SECTION_VOXELS {
            let state_id = voxels.sections[s][idx] as u32;
            let emission = registry.light_emission(state_id);
            if emission > 0 {
                light.sections[s].set_block_light_idx(idx, emission);
                let (x, local_y, z) = crate::light::section_index_to_coords(idx);
                let y = sec_base_y + local_y as i32;
                block_queue.push_back(pack_pos(x, y, z));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Block Light Boundary Seeding from Neighbors
    // -----------------------------------------------------------------------
    if let Some(nbrs) = neighbors {
        seed_block_boundaries(&mut light, voxels, registry, nbrs, &mut block_queue);
    }

    // -----------------------------------------------------------------------
    // Step 6: Block Light BFS Flood-Fill Propagation
    // -----------------------------------------------------------------------
    propagate_bfs_block(&mut light, voxels, registry, &mut block_queue);

    // -----------------------------------------------------------------------
    // Step 7: Dirty Mask Computation
    // -----------------------------------------------------------------------
    let mut dirty_mask = 0u32;
    for s in 0..SECTION_COUNT {
        let changed = match prev_light {
            Some(prev) => light.sections[s].raw != prev.sections[s].raw,
            None => light.sections[s].raw.iter().any(|&b| b != 0),
        };
        if changed {
            dirty_mask |= 1 << s;
        }
    }

    LightComputeResult { light, dirty_mask }
}

/// Propagates Sky Light using 6-connected BFS.
fn propagate_bfs_sky(
    light: &mut ChunkLightColumn,
    voxels: &ChunkVoxelColumn,
    registry: &BlockRegistry,
    queue: &mut VecDeque<u32>,
) {
    while let Some(packed) = queue.pop_front() {
        let (x, y, z) = unpack_pos(packed);
        let cur_light = light.get_sky_light(x, y, z);
        if cur_light <= 1 {
            continue;
        }

        for (dx, dy, dz) in CARDINAL_DIRECTIONS {
            let nx = x as i32 + dx;
            let ny = y + dy;
            let nz = z as i32 + dz;

            if ny < MIN_Y || ny >= MAX_Y {
                continue;
            }
            if nx < 0 || nx >= SECTION_SIZE as i32 || nz < 0 || nz >= SECTION_SIZE as i32 {
                // Crosses chunk boundary horizontally; in single-chunk recompute, stops.
                continue;
            }

            let nx = nx as usize;
            let nz = nz as usize;
            let dest_state = voxels.get_state(nx, ny, nz) as u32;
            let dest_filter = registry.light_filter(dest_state);

            // Attenuation decrement = max(1, dampening(destination))
            // Note: LightEngine.getLightDampeningInto 16-gate applies to useShapeForLightOcclusion
            // blocks (stairs/panes). In v1 all supported meshes have open faces or simpleOpacity.
            let attenuation = dest_filter.max(1);
            let incoming = cur_light.saturating_sub(attenuation);

            if incoming > light.get_sky_light(nx, ny, nz) {
                light.set_sky_light(nx, ny, nz, incoming);
                queue.push_back(pack_pos(nx, ny, nz));
            }
        }
    }
}

/// Propagates Block Light using 6-connected BFS.
fn propagate_bfs_block(
    light: &mut ChunkLightColumn,
    voxels: &ChunkVoxelColumn,
    registry: &BlockRegistry,
    queue: &mut VecDeque<u32>,
) {
    while let Some(packed) = queue.pop_front() {
        let (x, y, z) = unpack_pos(packed);
        let cur_light = light.get_block_light(x, y, z);
        if cur_light <= 1 {
            continue;
        }

        for (dx, dy, dz) in CARDINAL_DIRECTIONS {
            let nx = x as i32 + dx;
            let ny = y + dy;
            let nz = z as i32 + dz;

            if ny < MIN_Y || ny >= MAX_Y {
                continue;
            }
            if nx < 0 || nx >= SECTION_SIZE as i32 || nz < 0 || nz >= SECTION_SIZE as i32 {
                continue;
            }

            let nx = nx as usize;
            let nz = nz as usize;
            let dest_state = voxels.get_state(nx, ny, nz) as u32;
            let dest_filter = registry.light_filter(dest_state);

            let attenuation = dest_filter.max(1);
            let incoming = cur_light.saturating_sub(attenuation);

            if incoming > light.get_block_light(nx, ny, nz) {
                light.set_block_light(nx, ny, nz, incoming);
                queue.push_back(pack_pos(nx, ny, nz));
            }
        }
    }
}

/// Seeds sky light across adjacent chunk boundaries.
fn seed_sky_boundaries(
    light: &mut ChunkLightColumn,
    voxels: &ChunkVoxelColumn,
    registry: &BlockRegistry,
    neighbors: &NeighborChunkLights,
    queue: &mut VecDeque<u32>,
) {
    for y in MIN_Y..MAX_Y {
        // West (-X, x = 0 in current chunk, reads x = 15 from west neighbor)
        if let Some(west) = neighbors.west {
            for z in 0..SECTION_SIZE {
                let n_sky = west.get_sky_light(SECTION_SIZE - 1, y, z);
                if n_sky > 1 {
                    let dest_state = voxels.get_state(0, y, z) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_sky.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_sky_light(0, y, z) {
                        light.set_sky_light(0, y, z, incoming);
                        queue.push_back(pack_pos(0, y, z));
                    }
                }
            }
        }
        // East (+X, x = 15 in current chunk, reads x = 0 from east neighbor)
        if let Some(east) = neighbors.east {
            for z in 0..SECTION_SIZE {
                let n_sky = east.get_sky_light(0, y, z);
                if n_sky > 1 {
                    let dest_state = voxels.get_state(SECTION_SIZE - 1, y, z) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_sky.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_sky_light(SECTION_SIZE - 1, y, z) {
                        light.set_sky_light(SECTION_SIZE - 1, y, z, incoming);
                        queue.push_back(pack_pos(SECTION_SIZE - 1, y, z));
                    }
                }
            }
        }
        // North (-Z, z = 0 in current chunk, reads z = 15 from north neighbor)
        if let Some(north) = neighbors.north {
            for x in 0..SECTION_SIZE {
                let n_sky = north.get_sky_light(x, y, SECTION_SIZE - 1);
                if n_sky > 1 {
                    let dest_state = voxels.get_state(x, y, 0) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_sky.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_sky_light(x, y, 0) {
                        light.set_sky_light(x, y, 0, incoming);
                        queue.push_back(pack_pos(x, y, 0));
                    }
                }
            }
        }
        // South (+Z, z = 15 in current chunk, reads z = 0 from south neighbor)
        if let Some(south) = neighbors.south {
            for x in 0..SECTION_SIZE {
                let n_sky = south.get_sky_light(x, y, 0);
                if n_sky > 1 {
                    let dest_state = voxels.get_state(x, y, SECTION_SIZE - 1) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_sky.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_sky_light(x, y, SECTION_SIZE - 1) {
                        light.set_sky_light(x, y, SECTION_SIZE - 1, incoming);
                        queue.push_back(pack_pos(x, y, SECTION_SIZE - 1));
                    }
                }
            }
        }
    }
}

/// Seeds block light across adjacent chunk boundaries.
fn seed_block_boundaries(
    light: &mut ChunkLightColumn,
    voxels: &ChunkVoxelColumn,
    registry: &BlockRegistry,
    neighbors: &NeighborChunkLights,
    queue: &mut VecDeque<u32>,
) {
    for y in MIN_Y..MAX_Y {
        // West (-X, x = 0)
        if let Some(west) = neighbors.west {
            for z in 0..SECTION_SIZE {
                let n_block = west.get_block_light(SECTION_SIZE - 1, y, z);
                if n_block > 1 {
                    let dest_state = voxels.get_state(0, y, z) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_block.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_block_light(0, y, z) {
                        light.set_block_light(0, y, z, incoming);
                        queue.push_back(pack_pos(0, y, z));
                    }
                }
            }
        }
        // East (+X, x = 15)
        if let Some(east) = neighbors.east {
            for z in 0..SECTION_SIZE {
                let n_block = east.get_block_light(0, y, z);
                if n_block > 1 {
                    let dest_state = voxels.get_state(SECTION_SIZE - 1, y, z) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_block.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_block_light(SECTION_SIZE - 1, y, z) {
                        light.set_block_light(SECTION_SIZE - 1, y, z, incoming);
                        queue.push_back(pack_pos(SECTION_SIZE - 1, y, z));
                    }
                }
            }
        }
        // North (-Z, z = 0)
        if let Some(north) = neighbors.north {
            for x in 0..SECTION_SIZE {
                let n_block = north.get_block_light(x, y, SECTION_SIZE - 1);
                if n_block > 1 {
                    let dest_state = voxels.get_state(x, y, 0) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_block.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_block_light(x, y, 0) {
                        light.set_block_light(x, y, 0, incoming);
                        queue.push_back(pack_pos(x, y, 0));
                    }
                }
            }
        }
        // South (+Z, z = 15)
        if let Some(south) = neighbors.south {
            for x in 0..SECTION_SIZE {
                let n_block = south.get_block_light(x, y, 0);
                if n_block > 1 {
                    let dest_state = voxels.get_state(x, y, SECTION_SIZE - 1) as u32;
                    let dest_filter = registry.light_filter(dest_state);
                    let incoming = n_block.saturating_sub(dest_filter.max(1));
                    if incoming > light.get_block_light(x, y, SECTION_SIZE - 1) {
                        light.set_block_light(x, y, SECTION_SIZE - 1, incoming);
                        queue.push_back(pack_pos(x, y, SECTION_SIZE - 1));
                    }
                }
            }
        }
    }
}
