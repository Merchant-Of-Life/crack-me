// Minecraft 26.3 Terrain Shader (Dual-Pass Opaque/Cutout + Translucent)
// Bit-exact unpack, Lightmap brightness curve f(L) = L / (4 - 3L),
// CardinalLighting.DEFAULT, AO curve [APPROX-RENDERING-05],
// and dynamic tinting [APPROX-RENDERING-07/09].

struct CameraUniform {
    view_proj: mat4x4<f32>,
    sky_dampen: u32,
    tick_count: u32,
    _pad: vec2<u32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var t_array: texture_2d_array<f32>;

@group(1) @binding(1)
var s_array: sampler;

struct AnimationMeta {
    offset: u32,
    len: u32,
    frametime: u32,
};

@group(1) @binding(2)
var<storage, read> animation_meta: array<AnimationMeta>;

@group(1) @binding(3)
var<storage, read> animation_timeline: array<u32>;

struct VertexInput {
    @location(0) word0: u32,
    @location(1) word1: u32,
    @location(2) world_offset: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
    @location(2) @interpolate(flat) tint_mode: u32,
    @location(3) @interpolate(flat) uv_rot: u32,
    @location(4) @interpolate(flat) normal_id: u32,
    @location(5) ao: f32,
    @location(6) block_light: f32,
    @location(7) sky_light: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Bit-exact unpack using shared unpack.wgsl:
    let v = unpack_voxel_vertex(in.word0, in.word1);

    let world_pos = in.world_offset + vec3<f32>(v.local_x, v.local_y, v.local_z);
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // UV Axis Mapping from SPECS §8:
    // FACE_UP (+Y, 0): U = x, V = z
    // FACE_DOWN (-Y, 1): U = x, V = 1.0 - z  (vanilla defaultFaceUV, mirrored along z)
    // FACE_SOUTH (+Z, 2): U = x, V = 1.0 - y
    // FACE_NORTH (-Z, 3): U = 1.0 - x, V = 1.0 - y
    // FACE_EAST (+X, 4): U = 1.0 - z, V = 1.0 - y
    // FACE_WEST (-X, 5): U = z, V = 1.0 - y
    var raw_uv: vec2<f32>;
    switch (v.normal_id) {
        case 0u: { // FACE_UP (+Y)
            raw_uv = vec2<f32>(v.local_x, v.local_z);
        }
        case 1u: { // FACE_DOWN (-Y)
            raw_uv = vec2<f32>(v.local_x, 1.0 - v.local_z);
        }
        case 2u: { // FACE_SOUTH (+Z)
            raw_uv = vec2<f32>(v.local_x, 1.0 - v.local_y);
        }
        case 3u: { // FACE_NORTH (-Z)
            raw_uv = vec2<f32>(1.0 - v.local_x, 1.0 - v.local_y);
        }
        case 4u: { // FACE_EAST (+X)
            raw_uv = vec2<f32>(1.0 - v.local_z, 1.0 - v.local_y);
        }
        default: { // 5u, FACE_WEST (-X)
            raw_uv = vec2<f32>(v.local_z, 1.0 - v.local_y);
        }
    }

    out.uv = raw_uv;

    let anim_meta = animation_meta[v.base_layer];
    var resolved_layer = v.base_layer;
    if (anim_meta.len > 0u) {
        if (anim_meta.frametime == 0u) {
            resolved_layer = v.base_layer + animation_timeline[anim_meta.offset];
        } else {
            let frame_idx = animation_timeline[anim_meta.offset + (camera.tick_count % anim_meta.len)];
            resolved_layer = v.base_layer + frame_idx;
        }
    }
    out.layer = resolved_layer;

    out.tint_mode = v.tint_mode;
    out.uv_rot = v.uv_rot;
    out.normal_id = v.normal_id;
    out.ao = v.ao;
    out.block_light = v.block_light;
    out.sky_light = v.sky_light;

    return out;
}

// Vanilla Brightness Curve: f(L) = L / (4.0 - 3.0 * L), L = lit / 15.0
// Net.minecraft.client.renderer.Lightmap.getBrightness bytecode
fn vanilla_brightness(lit: f32) -> f32 {
    let L = clamp(lit, 0.0, 15.0) / 15.0;
    let denom = 4.0 - 3.0 * L;
    if (denom <= 0.0) {
        return 1.0;
    }
    return L / denom;
}

// Directional face shading constants (CardinalLighting.DEFAULT):
// UP (0): 1.0, DOWN (1): 0.5, SOUTH (2): 0.8, NORTH (3): 0.8, EAST (4): 0.6, WEST (5): 0.6
fn cardinal_shading(normal_id: u32) -> f32 {
    switch (normal_id) {
        case 0u: { return 1.0; } // UP
        case 1u: { return 0.5; } // DOWN
        case 2u, 3u: { return 0.8; } // SOUTH, NORTH
        case 4u, 5u: { return 0.6; } // EAST, WEST
        default: { return 1.0; }
    }
}

// Linear tint constants from SPECS §7 (Requirement R8):
// 0: vec3(1.0, 1.0, 1.0)
// 1 (TINT_GRASS): #79C05A -> vec3(0.1912017, 0.5271151, 0.1022417)
// 2 (TINT_FOLIAGE): #59C93C -> vec3(0.0998987, 0.5840784, 0.0451862)
// 3 (TINT_WATER): #3F76E4 -> vec3(0.0497066, 0.1811642, 0.7758222)
// 4 (TINT_DRY_FOLIAGE): #929786 -> vec3(0.2874408, 0.3094689, 0.2383976)
fn evaluate_tint(tint_mode: u32) -> vec3<f32> {
    switch (tint_mode) {
        case 1u: { return vec3<f32>(0.1912017, 0.5271151, 0.1022417); }
        case 2u: { return vec3<f32>(0.0998987, 0.5840784, 0.0451862); }
        case 3u: { return vec3<f32>(0.0497066, 0.1811642, 0.7758222); }
        case 4u: { return vec3<f32>(0.2874408, 0.3094689, 0.2383976); }
        default: { return vec3<f32>(1.0, 1.0, 1.0); }
    }
}

fn apply_uv_rotation(tiled_uv: vec2<f32>, rot: u32) -> vec2<f32> {
    let u = tiled_uv.x;
    let v = tiled_uv.y;
    switch (rot) {
        case 1u: { return vec2<f32>(v, 1.0 - u); }
        case 2u: { return vec2<f32>(1.0 - u, 1.0 - v); }
        case 3u: { return vec2<f32>(1.0 - v, u); }
        default: { return vec2<f32>(u, v); }
    }
}

@fragment
fn fs_pass1_opaque_cutout(in: VertexOutput) -> @location(0) vec4<f32> {
    let tiled_uv = fract(in.uv);
    let sample_uv = apply_uv_rotation(tiled_uv, in.uv_rot);

    // Explicit LOD 0 sampling [APPROX-RENDERING-06]
    let tex_color = textureSampleLevel(t_array, s_array, sample_uv, in.layer, 0.0);

    // Alpha cutout threshold: 0.5 for terrain (ALPHA_CUTOUT_THRESHOLD_CUTOUT_TERRAIN [APPROX-RENDERING-02])
    if (tex_color.a < 0.5) {
        discard;
    }

    // Effective light level: lit = max(block_light, eff_sky * day_factor) [APPROX-RENDERING-03]
    let day_factor = 1.0;
    let eff_sky = max(0.0, in.sky_light - f32(camera.sky_dampen));
    let lit = max(in.block_light, eff_sky * day_factor);
    let f_light = vanilla_brightness(lit);

    // AO factor: 0.4 + 0.2 * ao [APPROX-RENDERING-05]
    let ao_factor = 0.4 + 0.2 * in.ao;

    // Directional cardinal lighting (CardinalLighting.DEFAULT)
    let f_cardinal = cardinal_shading(in.normal_id);

    // Combined light multiplier: FACE_SHADE * ao_factor * (0.05 + 0.95 * f(L)) [APPROX-RENDERING-04]
    let light_mult = f_cardinal * ao_factor * (0.05 + 0.95 * f_light);

    let tint = evaluate_tint(in.tint_mode);

    let final_rgb = tex_color.rgb * tint * light_mult;
    return vec4<f32>(final_rgb, 1.0);
}

@fragment
fn fs_pass2_translucent(in: VertexOutput) -> @location(0) vec4<f32> {
    let tiled_uv = fract(in.uv);
    let sample_uv = apply_uv_rotation(tiled_uv, in.uv_rot);

    // Explicit LOD 0 sampling [APPROX-RENDERING-06]
    let tex_color = textureSampleLevel(t_array, s_array, sample_uv, in.layer, 0.0);

    let day_factor = 1.0;
    let eff_sky = max(0.0, in.sky_light - f32(camera.sky_dampen));
    let lit = max(in.block_light, eff_sky * day_factor);
    let f_light = vanilla_brightness(lit);
    let ao_factor = 0.4 + 0.2 * in.ao;
    let f_cardinal = cardinal_shading(in.normal_id);
    let light_mult = f_cardinal * ao_factor * (0.05 + 0.95 * f_light);

    let tint = evaluate_tint(in.tint_mode);

    let final_rgb = tex_color.rgb * tint * light_mult;
    return vec4<f32>(final_rgb, tex_color.a);
}
