// Bit-exact unpack per SPECS §4 (Pure WGSL Struct & Function)

struct UnpackedVoxelVertex {
    local_x: f32,
    local_y: f32,
    local_z: f32,
    normal_id: u32,
    tint_mode: u32,
    uv_rot: u32,
    block_light: f32,
    sky_light: f32,
    ao: f32,
    base_layer: u32,
    dry_foliage: bool,
};

fn unpack_voxel_vertex(word0: u32, word1: u32) -> UnpackedVoxelVertex {
    var out: UnpackedVoxelVertex;
    out.local_x = f32(word0 & 0x1Fu);
    out.local_y = f32((word0 >> 5u) & 0x1Fu);
    out.local_z = f32((word0 >> 10u) & 0x1Fu);
    out.normal_id = (word0 >> 15u) & 0x07u;
    let tint_mode_raw = (word0 >> 18u) & 0x03u;
    out.uv_rot = (word0 >> 20u) & 0x03u;

    out.block_light = f32(word1 & 0x0Fu);
    out.sky_light = f32((word1 >> 4u) & 0x0Fu);
    out.ao = f32((word1 >> 8u) & 0x03u);
    out.base_layer = (word1 >> 10u) & 0x07FFu;
    out.dry_foliage = ((word1 >> 21u) & 0x01u) != 0u;

    out.tint_mode = tint_mode_raw;
    if (out.dry_foliage) {
        out.tint_mode = 4u;
    }

    return out;
}
