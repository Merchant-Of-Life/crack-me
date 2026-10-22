// Minecraft 26.3 Terrain Shader (Dual-Pass Opaque/Cutout + Translucent)
// Bit-exact unpack, Lightmap brightness curve f(L) = L / (4 - 3L),
// CardinalLighting.DEFAULT, AO curve [APPROX-RENDERING-05],
// and dynamic tinting [APPROX-RENDERING-07/09].

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var t_array: texture_2d_array<f32>;

@group(1) @binding(1)
var s_array: sampler;

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

    // Bit-exact unpack per SPECS §4:
    let local_x = f32(in.word0 & 0x1Fu);
    let local_y = f32((in.word0 >> 5u) & 0x1Fu);
    let local_z = f32((in.word0 >> 10u) & 0x1Fu);
    let normal_id = (in.word0 >> 15u) & 0x07u;
    let tint_mode_raw = (in.word0 >> 18u) & 0x03u;
    let uv_rot = (in.word0 >> 20u) & 0x03u;

    let block_light = f32(in.word1 & 0x0Fu);
    let sky_light = f32((in.word1 >> 4u) & 0x0Fu);
    let ao = f32((in.word1 >> 8u) & 0x03u);
    let base_layer = (in.word1 >> 10u) & 0x07FFu;
    let dry_foliage = ((in.word1 >> 21u) & 0x01u) != 0u;

    var tint_mode = tint_mode_raw;
    if (dry_foliage) {
        tint_mode = 4u;
    }

    let world_pos = in.world_offset + vec3<f32>(local_x, local_y, local_z);
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // UV Axis Mapping from SPECS §8:
    // FACE_UP (+Y, 0): U = x, V = z
    // FACE_DOWN (-Y, 1): U = x, V = 1.0 - z  (vanilla defaultFaceUV, mirrored along z)
    // FACE_SOUTH (+Z, 2): U = x, V = 1.0 - y
    // FACE_NORTH (-Z, 3): U = 1.0 - x, V = 1.0 - y
    // FACE_EAST (+X, 4): U = 1.0 - z, V = 1.0 - y
    // FACE_WEST (-X, 5): U = z, V = 1.0 - y
    var raw_uv: vec2<f32>;
    switch (normal_id) {
        case 0u: { // FACE_UP (+Y)
            raw_uv = vec2<f32>(local_x, local_z);
        }
        case 1u: { // FACE_DOWN (-Y)
            raw_uv = vec2<f32>(local_x, 1.0 - local_z);
        }
        case 2u: { // FACE_SOUTH (+Z)
            raw_uv = vec2<f32>(local_x, 1.0 - local_y);
        }
        case 3u: { // FACE_NORTH (-Z)
            raw_uv = vec2<f32>(1.0 - local_x, 1.0 - local_y);
        }
        case 4u: { // FACE_EAST (+X)
            raw_uv = vec2<f32>(1.0 - local_z, 1.0 - local_y);
        }
        default: { // 5u, FACE_WEST (-X)
            raw_uv = vec2<f32>(local_z, 1.0 - local_y);
        }
    }

    out.uv = raw_uv;
    out.layer = base_layer;
    out.tint_mode = tint_mode;
    out.uv_rot = uv_rot;
    out.normal_id = normal_id;
    out.ao = ao;
    out.block_light = block_light;
    out.sky_light = sky_light;

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

// Tint constants from SPECS §7:
// 0: vec3(1.0, 1.0, 1.0)
// 1 (TINT_GRASS): #79C05A -> vec3(0.4745, 0.7529, 0.3529)
// 2 (TINT_FOLIAGE): #59C93C -> vec3(0.3490, 0.7882, 0.2353)
// 3 (TINT_WATER): #3F76E4 -> vec3(0.2471, 0.4627, 0.8941)
// 4 (TINT_DRY_FOLIAGE): #929786 -> vec3(0.5725, 0.5922, 0.5255)
fn evaluate_tint(tint_mode: u32) -> vec3<f32> {
    switch (tint_mode) {
        case 1u: { return vec3<f32>(0.4745098, 0.7529412, 0.3529412); }
        case 2u: { return vec3<f32>(0.3490196, 0.7882353, 0.2352941); }
        case 3u: { return vec3<f32>(0.2470588, 0.4627451, 0.8941176); }
        case 4u: { return vec3<f32>(0.5725490, 0.5921569, 0.5254902); }
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

    // Effective light level: lit = max(block_light, sky_light * day_factor) [APPROX-RENDERING-03]
    let day_factor = 1.0;
    let lit = max(in.block_light, in.sky_light * day_factor);
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
    let lit = max(in.block_light, in.sky_light * day_factor);
    let f_light = vanilla_brightness(lit);
    let ao_factor = 0.4 + 0.2 * in.ao;
    let f_cardinal = cardinal_shading(in.normal_id);
    let light_mult = f_cardinal * ao_factor * (0.05 + 0.95 * f_light);

    let tint = evaluate_tint(in.tint_mode);

    let final_rgb = tex_color.rgb * tint * light_mult;
    return vec4<f32>(final_rgb, tex_color.a);
}
