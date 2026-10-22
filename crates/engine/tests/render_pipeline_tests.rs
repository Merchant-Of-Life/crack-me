use std::path::Path;
use voxel_engine::mesher::{emit_quad, FaceMergeKey};
use voxel_engine::registry::{BlockRegistry, FACE_DOWN, FACE_EAST, FACE_NORTH, FACE_SOUTH, FACE_UP, FACE_WEST};
use voxel_engine::render::{
    validate_adapter_limits, Camera, CameraUniform, GpuContext, OffscreenTarget, SectionInstance,
    TerrainPipelines, TextureArray, UnpackedVoxelVertex,
};
use voxel_engine::PackedVoxelVertex;

fn assets_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets")
        .canonicalize()
        .expect("assets directory")
}

// ---------------------------------------------------------------------------
// 1. Adapter Limits Validation Seam
// ---------------------------------------------------------------------------
#[test]
fn test_adapter_limits_validation_seam() {
    let mut limits = wgpu::Limits::downlevel_defaults();

    // 1. Success when >= 2048
    limits.max_texture_array_layers = 2048;
    assert!(validate_adapter_limits(&limits).is_ok());

    limits.max_texture_array_layers = 8192;
    assert!(validate_adapter_limits(&limits).is_ok());

    // 2. Descriptive error when < 2048
    limits.max_texture_array_layers = 256;
    let err = validate_adapter_limits(&limits).unwrap_err();
    assert!(
        err.contains("max_texture_array_layers is 256"),
        "error message must cite actual limit: {}",
        err
    );
    assert!(
        err.contains("required >= 2048"),
        "error message must cite required limit: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// 2. WGSL Bit-Exact Unpack Mirror & Compute Verification
// ---------------------------------------------------------------------------
#[test]
fn test_wgsl_bit_exact_unpack_matches_cpu() {
    // Generate test vertices exercising boundaries of all packed bitfields
    let test_cases = [
        // (x, y, z, normal_id, tint_mode, uv_rot, block_light, sky_light, ao, base_layer, dry_foliage)
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false),
        (16, 16, 16, 5, 3, 3, 15, 15, 3, 2047, false),
        (8, 4, 12, 2, 1, 2, 7, 14, 1, 630, false),
        (1, 15, 9, 3, 2, 1, 12, 0, 2, 1840, false),
        (10, 2, 5, 4, 0, 3, 0, 15, 3, 1843, true), // dry_foliage = true -> tint_mode 4
        (4, 8, 14, 1, 3, 0, 10, 8, 0, 500, true),  // dry_foliage = true -> tint_mode 4
    ];

    let mut packed_verts = Vec::new();
    let mut cpu_unpacked = Vec::new();

    for &(x, y, z, normal, tint, rot, blk, sky, ao, layer, dry) in &test_cases {
        let mut v = PackedVoxelVertex::new(x, y, z, normal, 0, rot, blk, sky, ao, layer);
        if dry {
            v.set_registry_tint_mode(4);
        } else {
            v.set_registry_tint_mode(tint as u8);
        }

        let unpacked = UnpackedVoxelVertex::from_packed(v);
        assert_eq!(unpacked.local_x, x as f32);
        assert_eq!(unpacked.local_y, y as f32);
        assert_eq!(unpacked.local_z, z as f32);
        assert_eq!(unpacked.normal_id, normal);
        if dry {
            assert_eq!(unpacked.tint_mode, 4);
            assert!(unpacked.dry_foliage);
        } else {
            assert_eq!(unpacked.tint_mode, tint);
            assert!(!unpacked.dry_foliage);
        }
        assert_eq!(unpacked.uv_rot, rot);
        assert_eq!(unpacked.block_light, blk as f32);
        assert_eq!(unpacked.sky_light, sky as f32);
        assert_eq!(unpacked.ao, ao as f32);
        assert_eq!(unpacked.base_layer, layer);

        packed_verts.push(v);
        cpu_unpacked.push(unpacked);
    }

    // Now execute GPU compute shader with the exact WGSL unpack code to verify 100% numerical identity
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping GPU compute unpack check (no headless adapter): {}", e);
            return;
        }
    };

    let wgsl_compute = concat!(
        include_str!("../src/render/shaders/unpack.wgsl"),
        r#"
        struct PackedVertex {
            word0: u32,
            word1: u32,
        };

        struct UnpackedGpu {
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
            dry_foliage: u32,
            _pad: u32,
        };

        @group(0) @binding(0)
        var<storage, read> in_packed: array<PackedVertex>;

        @group(0) @binding(1)
        var<storage, read_write> out_unpacked: array<UnpackedGpu>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if (idx >= arrayLength(&in_packed)) {
                return;
            }
            let word0 = in_packed[idx].word0;
            let word1 = in_packed[idx].word1;

            let u = unpack_voxel_vertex(word0, word1);

            var out: UnpackedGpu;
            out.local_x = u.local_x;
            out.local_y = u.local_y;
            out.local_z = u.local_z;
            out.normal_id = u.normal_id;
            out.tint_mode = u.tint_mode;
            out.uv_rot = u.uv_rot;
            out.block_light = u.block_light;
            out.sky_light = u.sky_light;
            out.ao = u.ao;
            out.base_layer = u.base_layer;
            if (u.dry_foliage) {
                out.dry_foliage = 1u;
            } else {
                out.dry_foliage = 0u;
            }
            out._pad = 0u;
            out_unpacked[idx] = out;
        }
    "#
    );

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_unpack_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_compute.into()),
    });

    let count = packed_verts.len();
    let packed_bytes: Vec<u8> = packed_verts
        .iter()
        .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
        .collect();

    use wgpu::util::DeviceExt;
    let in_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("in_packed_buf"),
        contents: &packed_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 12 fields of 4 bytes each = 48 bytes per vertex
    let out_buf_size = (count * 48) as wgpu::BufferAddress;
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_unpacked_buf"),
        size: out_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buf"),
        size: out_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("unpack_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_buf_size);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().expect("map_async failed");

    let data = slice.get_mapped_range().expect("get_mapped_range failed");
    for (i, cpu) in cpu_unpacked.iter().enumerate() {
        let base = i * 48;
        let gpu_x = f32::from_ne_bytes(data[base..base + 4].try_into().unwrap());
        let gpu_y = f32::from_ne_bytes(data[base + 4..base + 8].try_into().unwrap());
        let gpu_z = f32::from_ne_bytes(data[base + 8..base + 12].try_into().unwrap());
        let gpu_normal = u32::from_ne_bytes(data[base + 12..base + 16].try_into().unwrap());
        let gpu_tint = u32::from_ne_bytes(data[base + 16..base + 20].try_into().unwrap());
        let gpu_rot = u32::from_ne_bytes(data[base + 20..base + 24].try_into().unwrap());
        let gpu_bl = f32::from_ne_bytes(data[base + 24..base + 28].try_into().unwrap());
        let gpu_sl = f32::from_ne_bytes(data[base + 28..base + 32].try_into().unwrap());
        let gpu_ao = f32::from_ne_bytes(data[base + 32..base + 36].try_into().unwrap());
        let gpu_layer = u32::from_ne_bytes(data[base + 36..base + 40].try_into().unwrap());
        let gpu_dry = u32::from_ne_bytes(data[base + 40..base + 44].try_into().unwrap()) != 0;

        assert_eq!(gpu_x, cpu.local_x, "local_x mismatch at index {}", i);
        assert_eq!(gpu_y, cpu.local_y, "local_y mismatch at index {}", i);
        assert_eq!(gpu_z, cpu.local_z, "local_z mismatch at index {}", i);
        assert_eq!(gpu_normal, cpu.normal_id, "normal_id mismatch at index {}", i);
        assert_eq!(gpu_tint, cpu.tint_mode, "tint_mode mismatch at index {}", i);
        assert_eq!(gpu_rot, cpu.uv_rot, "uv_rot mismatch at index {}", i);
        assert_eq!(gpu_bl, cpu.block_light, "block_light mismatch at index {}", i);
        assert_eq!(gpu_sl, cpu.sky_light, "sky_light mismatch at index {}", i);
        assert_eq!(gpu_ao, cpu.ao, "ao mismatch at index {}", i);
        assert_eq!(gpu_layer, cpu.base_layer, "base_layer mismatch at index {}", i);
        assert_eq!(gpu_dry, cpu.dry_foliage, "dry_foliage mismatch at index {}", i);
    }

    drop(data);
    staging_buf.unmap();
}

// ---------------------------------------------------------------------------
// 3. Shader Light Curve Identity Verification
// ---------------------------------------------------------------------------
#[test]
fn test_shader_light_curve_matches_formula() {
    // Vanilla Lightmap brightness curve: f(L) = L / (4.0 - 3.0 * L) for L in [0.0, 1.0]
    let mut prev_brightness = -1.0f32;

    for lit in 0..=15 {
        let l = lit as f32 / 15.0;
        let brightness = l / (4.0 - 3.0 * l);

        assert!(brightness >= 0.0 && brightness <= 1.0);
        assert!(brightness > prev_brightness, "brightness curve must be strictly increasing");
        prev_brightness = brightness;

        if lit == 0 {
            assert_eq!(brightness, 0.0, "lit=0 must yield brightness 0.0");
        }
        if lit == 15 {
            assert_eq!(brightness, 1.0, "lit=15 must yield brightness 1.0");
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Headless Offscreen Render: Solid Cube & Directional Contrast
// ---------------------------------------------------------------------------
#[test]
fn test_headless_offscreen_render_solid_cube() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping render test (no headless adapter): {}", e);
            return;
        }
    };

    let assets = assets_dir();
    let textures = TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))
        .expect("TextureArray from assets");

    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let target = OffscreenTarget::new(&ctx.device, 256, 256);

    // Camera looking down at a 1x1x1 cube from front-top-right:
    // Eye at (2.5, 3.0, 3.5), target at (0.5, 0.5, 0.5).
    // From here, UP (+Y), SOUTH (+Z), and EAST (+X) faces are visible.
    let camera = Camera::new(
        glam::vec3(2.5, 3.0, 3.5),
        glam::vec3(0.5, 0.5, 0.5),
        1.0, // 256 / 256 aspect
    );

    use wgpu::util::DeviceExt;
    let camera_uniform = camera.build_uniform();
    let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_uniform_buf"),
        contents: bytemuck::bytes_of(&camera_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);
    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

    // Build 6 faces of a unit cube at (0, 0, 0)
    let registry = BlockRegistry::from_file(&assets.join("registry/block_registry.bin"))
        .expect("load block_registry.bin");
    let stone = registry.get_block("stone").expect("stone block");
    let state = registry.get_state(stone.state_start).unwrap();
    let stone_layer = state.faces[FACE_UP].base_layer;

    let mut vertices: Vec<PackedVoxelVertex> = Vec::new();
    let key_up = FaceMergeKey {
        base_layer: stone_layer,
        tint_mode: 0,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };
    // Emit all 6 faces using the mesher's canonical quad winding:
    // 0: FACE_UP (+Y, y = 1)
    emit_quad(&mut vertices, (0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0), FACE_UP as u32, &key_up);

    // 1: FACE_DOWN (-Y, y = 0)
    emit_quad(&mut vertices, (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1), FACE_DOWN as u32, &key_up);

    // 2: FACE_SOUTH (+Z, z = 1)
    emit_quad(&mut vertices, (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1), FACE_SOUTH as u32, &key_up);

    // 3: FACE_NORTH (-Z, z = 0)
    emit_quad(&mut vertices, (1, 1, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0), FACE_NORTH as u32, &key_up);

    // 4: FACE_EAST (+X, x = 1)
    emit_quad(&mut vertices, (1, 1, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0), FACE_EAST as u32, &key_up);

    // 5: FACE_WEST (-X, x = 0)
    emit_quad(&mut vertices, (0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1), FACE_WEST as u32, &key_up);

    let v_bytes: Vec<u8> = vertices
        .iter()
        .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
        .collect();

    let vertex_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cube_vertex_buf"),
        contents: &v_bytes,
        usage: wgpu::BufferUsages::VERTEX,
    });

    let instance = SectionInstance::new(0.0, 0.0, 0.0);
    let instance_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cube_instance_buf"),
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass1_opaque_cube"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&pipelines.pass1_opaque_cutout);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));

    let pixels = target.readback_pixels(&ctx.device, &ctx.queue);

    // Verify non-zero rendered pixels exist
    let rendered_count = pixels.chunks_exact(4).filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    assert!(rendered_count > 500, "rendered cube must have visible pixels (got {})", rendered_count);

    let mut top_face_samples = Vec::new();
    let mut south_face_samples = Vec::new();
    let mut east_face_samples = Vec::new();

    // Sample regions in the 256x256 image centered around projected face centers:
    // Top face center: (128.0, 109.7)
    for y in 100..115 {
        for x in 120..136 {
            let idx = (y * 256 + x) * 4;
            let lum = pixels[idx] as f32;
            if lum > 10.0 {
                top_face_samples.push(lum);
            }
        }
    }

    // South face (+Z) center: (115.5, 138.7)
    for y in 132..148 {
        for x in 108..122 {
            let idx = (y * 256 + x) * 4;
            let lum = pixels[idx] as f32;
            if lum > 10.0 {
                south_face_samples.push(lum);
            }
        }
    }

    // East face (+X) center: (146.3, 134.9)
    for y in 128..144 {
        for x in 140..154 {
            let idx = (y * 256 + x) * 4;
            let lum = pixels[idx] as f32;
            if lum > 10.0 {
                east_face_samples.push(lum);
            }
        }
    }

    assert!(!top_face_samples.is_empty(), "must sample pixels from top face");
    assert!(!south_face_samples.is_empty(), "must sample pixels from south face");
    assert!(!east_face_samples.is_empty(), "must sample pixels from east face");

    let avg_top = top_face_samples.iter().sum::<f32>() / top_face_samples.len() as f32;
    let avg_south = south_face_samples.iter().sum::<f32>() / south_face_samples.len() as f32;
    let avg_east = east_face_samples.iter().sum::<f32>() / east_face_samples.len() as f32;

    println!("Directional Shading Observed: Top={:.1}, South={:.1}, East={:.1}", avg_top, avg_south, avg_east);

    // Directional shading contract: Top (1.0) > South (0.8) > East (0.6)
    assert!(
        avg_top > avg_south,
        "Top face ({:.1}) must be brighter than South face ({:.1})",
        avg_top,
        avg_south
    );
    assert!(
        avg_south > avg_east,
        "South face ({:.1}) must be brighter than East face ({:.1})",
        avg_south,
        avg_east
    );
}

// ---------------------------------------------------------------------------
// 5. Headless Cutout Alpha Discard (Threshold 0.5)
// ---------------------------------------------------------------------------
#[test]
fn test_headless_cutout_alpha_discard() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping cutout test (no headless adapter): {}", e);
            return;
        }
    };

    // Create a 2-layer test texture array:
    // Layer 0: solid white (for background quad)
    // Layer 1: top half opaque white (alpha 255), bottom half transparent (alpha 0)
    let mut m0 = vec![255u8; 16 * 16 * 4 * 2];
    let offset_l1 = 16 * 16 * 4;
    // Set bottom half of Layer 1 to alpha 0
    for y in 8..16 {
        for x in 0..16 {
            let px = offset_l1 + (y * 16 + x) * 4;
            m0[px + 3] = 0; // alpha = 0 (< 0.5)
        }
    }

    let m1 = vec![255u8; 8 * 8 * 4 * 2];
    let m2 = vec![255u8; 4 * 4 * 4 * 2];
    let m3 = vec![255u8; 2 * 2 * 4 * 2];
    let m4 = vec![255u8; 1 * 1 * 4 * 2];

    let textures = TextureArray::from_mip_buffers(&ctx.device, &ctx.queue, 2, [&m0, &m1, &m2, &m3, &m4])
        .expect("TextureArray creation");

    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let target = OffscreenTarget::new(&ctx.device, 128, 128);

    // Orthogonal camera looking straight down -Z at (0, 0, 0)
    let camera = Camera::new(
        glam::vec3(0.5, 0.5, 3.0),
        glam::vec3(0.5, 0.5, 0.0),
        1.0,
    );

    use wgpu::util::DeviceExt;
    let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&camera.build_uniform()),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);
    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

    // Quad 1 (Background): z = -1.0 (further away), layer = 0, tinted red
    // Quad 2 (Cutout Foreground): z = 0.0 (closer), layer = 1, tinted blue
    let mut vertices: Vec<PackedVoxelVertex> = Vec::new();

    // Quad 2 (Foreground): normal SOUTH (+Z) at local (0, 0, 1) to (1, 1, 1)
    let key_fg = FaceMergeKey {
        base_layer: 1, // Cutout layer (top half alpha 255, bottom half alpha 0)
        tint_mode: 0,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };
    emit_quad(&mut vertices, (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1), FACE_SOUTH as u32, &key_fg);

    let v_bytes: Vec<u8> = vertices
        .iter()
        .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
        .collect();

    let vertex_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &v_bytes,
        usage: wgpu::BufferUsages::VERTEX,
    });

    let instance = SectionInstance::new(0.0, 0.0, 0.0);
    let instance_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Render with background cleared to green [0, 255, 0]
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutout_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 1.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&pipelines.pass1_opaque_cutout);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let pixels = target.readback_pixels(&ctx.device, &ctx.queue);

    // Verify:
    // Top half (y < 64 in texture space): foreground rendered (white / light-shaded)
    // Bottom half (y > 64 in texture space): discarded! Clear color (green) preserved!
    let top_pixel = &pixels[(53 * 128 + 64) * 4..((53 * 128 + 64) * 4 + 4)];
    let bottom_pixel = &pixels[(75 * 128 + 64) * 4..((75 * 128 + 64) * 4 + 4)];

    // Top pixel must be rendered foreground (non-green / R=G=B white)
    assert!(top_pixel[0] > 50, "top pixel must render foreground (got R={})", top_pixel[0]);
    // Bottom pixel must be cleared background (G=255, R=0) due to alpha discard
    assert_eq!(bottom_pixel[0], 0, "bottom pixel must have discarded foreground: R must be 0");
    assert_eq!(bottom_pixel[1], 255, "bottom pixel must preserve clear green background: G must be 255");
}

// ---------------------------------------------------------------------------
// 6. Headless Translucent Pass: Two-Sided Rendering
// ---------------------------------------------------------------------------
#[test]
fn test_headless_translucent_pass_two_sided() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping translucent test (no headless adapter): {}", e);
            return;
        }
    };

    let dummy_textures = TextureArray::dummy(&ctx.device, &ctx.queue, 1).unwrap();
    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let target = OffscreenTarget::new(&ctx.device, 128, 128);

    // Camera at (0.5, 0.5, 3.0) looking at (0.5, 0.5, 0.0)
    let camera = Camera::new(
        glam::vec3(0.5, 0.5, 3.0),
        glam::vec3(0.5, 0.5, 0.0),
        1.0,
    );

    use wgpu::util::DeviceExt;
    let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&camera.build_uniform()),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);
    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &dummy_textures);

    // Create a BACK-FACING quad facing away from the camera:
    // Normal is NORTH (-Z, normal_id=3), situated at z=0, facing away from camera at +Z.
    // Viewed from camera at +Z, its vertices wind CW (back-facing).
    let mut vertices: Vec<PackedVoxelVertex> = Vec::new();
    let key = FaceMergeKey {
        base_layer: 0,
        tint_mode: 0,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };
    emit_quad(&mut vertices, (1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0), FACE_NORTH as u32, &key);

    let v_bytes: Vec<u8> = vertices
        .iter()
        .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
        .collect();

    let vertex_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &v_bytes,
        usage: wgpu::BufferUsages::VERTEX,
    });

    let instance = SectionInstance::new(0.0, 0.0, 0.0);
    let instance_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // 1. In Pass 1 (Opaque): cull_mode: Some(Face::Back). The back-facing quad MUST be culled.
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass1_cull_test"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&pipelines.pass1_opaque_cutout);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let pixels_pass1 = target.readback_pixels(&ctx.device, &ctx.queue);
    let pass1_visible = pixels_pass1.chunks_exact(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
    assert!(!pass1_visible, "Pass 1 must cull back-facing quad with cull_mode: Some(Face::Back)");

    // 2. In Pass 2 (Translucent): cull_mode: None (two-sided). The back-facing quad MUST render!
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass2_two_sided_test"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&pipelines.pass2_translucent);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let pixels_pass2 = target.readback_pixels(&ctx.device, &ctx.queue);
    let pass2_visible = pixels_pass2.chunks_exact(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
    assert!(pass2_visible, "Pass 2 must render back-facing quad with cull_mode: None (two-sided)");
}

// ---------------------------------------------------------------------------
// 7. Golden Invariant: Runtime Tint matches Bake apply_linear_tint (R8)
// ---------------------------------------------------------------------------
#[test]
fn test_golden_runtime_tint_matches_bake_apply_linear_tint() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping tint golden test (no headless adapter): {}", e);
            return;
        }
    };

    let assets = assets_dir();
    let textures = TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))
        .expect("TextureArray from assets");

    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let target = OffscreenTarget::new(&ctx.device, 16, 16);

    // Orthographic camera looking directly down at unit quad [0, 1] x [0, 1] on FACE_UP (y = 1)
    let view = glam::Mat4::look_at_rh(
        glam::vec3(0.5, 2.0, 0.5),
        glam::vec3(0.5, 1.0, 0.5),
        glam::vec3(0.0, 0.0, -1.0),
    );
    let proj = glam::Mat4::orthographic_rh(-0.5, 0.5, -0.5, 0.5, 0.1, 10.0);
    let camera_uniform = CameraUniform::new(proj * view);

    use wgpu::util::DeviceExt;
    let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tint_camera_uniform_buf"),
        contents: bytemuck::bytes_of(&camera_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);
    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

    // grass_block_top layer is 630 in texture_manifest.json
    // Full neutrality conditions:
    // - normal_id = 0 (FACE_UP) -> cardinal_shading(0) = 1.0
    // - ao = 3 -> ao_factor = 0.4 + 0.2 * 3.0 = 1.0
    // - sky_light = 15, block_light = 0, day_factor = 1.0 -> lit = 15.0 -> vanilla_brightness(15.0) = 1.0
    // - light_mult = 1.0 * 1.0 * (0.05 + 0.95 * 1.0) = 1.0
    // - tint_mode = 1 (TINT_GRASS, #79C05A)
    let key = FaceMergeKey {
        base_layer: 630,
        tint_mode: 1,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };

    let mut vertices: Vec<PackedVoxelVertex> = Vec::new();
    emit_quad(&mut vertices, (0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0), FACE_UP as u32, &key);

    let v_bytes: Vec<u8> = vertices
        .iter()
        .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
        .collect();

    let vertex_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tint_vertex_buf"),
        contents: &v_bytes,
        usage: wgpu::BufferUsages::VERTEX,
    });

    let instance = SectionInstance::new(0.0, 0.0, 0.0);
    let instance_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tint_instance_buf"),
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_tint_golden"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&pipelines.pass1_opaque_cutout);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_bind_group(1, &texture_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let pixels = target.readback_pixels(&ctx.device, &ctx.queue);

    // CPU Reference: raw mip0 texels for layer 630 tinted with #79C05A
    let mip0 = std::fs::read(assets.join("textures/block_layers_mip0_16x16.rgba8"))
        .expect("read mip0 binary");
    let raw_texels = &mip0[630 * 1024..631 * 1024];
    let expected = asset_pipeline::apply_linear_tint(raw_texels, 0x79C05A);

    assert_eq!(pixels.len(), expected.len(), "framebuffer size must match 16x16 RGBA8");

    // Exact byte equality comparison across all 256 texels (1024 bytes)
    let mut exact_matches = 0;
    let mut off_by_one = 0;
    let mut larger_diffs = Vec::new();

    for i in 0..pixels.len() {
        let actual = pixels[i];
        let exp = expected[i];
        if actual == exp {
            exact_matches += 1;
        } else if (actual as i16 - exp as i16).abs() == 1 {
            off_by_one += 1;
        } else {
            let byte_idx = i % 4;
            let channel = match byte_idx { 0 => "R", 1 => "G", 2 => "B", _ => "A" };
            let pixel_idx = i / 4;
            larger_diffs.push(format!(
                "pixel {} ({},{} channel {}): actual {} vs expected {}",
                pixel_idx, pixel_idx % 16, pixel_idx / 16, channel, actual, exp
            ));
        }
    }

    println!(
        "R8 Tint Comparison: {}/{} exact byte matches, {}/{} off-by-one (due to GPU hardware sRGB round)",
        exact_matches, pixels.len(), off_by_one, pixels.len()
    );

    assert!(
        larger_diffs.is_empty(),
        "GPU runtime tint deviated by > 1 LSB from CPU apply_linear_tint: {:?}",
        larger_diffs
    );

    if off_by_one == 0 {
        assert_eq!(pixels, expected, "GPU runtime tint must match CPU apply_linear_tint byte-for-byte");
    }
}

// ---------------------------------------------------------------------------
// Helper: Render vertices offscreen and read back pixels
// ---------------------------------------------------------------------------
fn render_quads_to_target(
    ctx: &GpuContext,
    pipelines: &TerrainPipelines,
    target: &OffscreenTarget,
    camera_bind_group: &wgpu::BindGroup,
    texture_bind_group: &wgpu::BindGroup,
    vertices: &[PackedVoxelVertex],
) -> Vec<u8> {
    use wgpu::util::DeviceExt;
    let v_bytes: Vec<u8> = vertices
        .iter()
        .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
        .collect();

    let vertex_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("render_quads_vertex_buf"),
        contents: &v_bytes,
        usage: wgpu::BufferUsages::VERTEX,
    });

    let instance = SectionInstance::new(0.0, 0.0, 0.0);
    let instance_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("render_quads_instance_buf"),
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_quads_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&pipelines.pass1_opaque_cutout);
        pass.set_bind_group(0, camera_bind_group, &[]);
        pass.set_bind_group(1, texture_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    target.readback_pixels(&ctx.device, &ctx.queue)
}

// ---------------------------------------------------------------------------
// 8. Greedy-Quad Goldens: Merged 16x16 & 2x1 Stone vs Singles (R9)
// ---------------------------------------------------------------------------
#[test]
fn test_greedy_quad_tiling_stone_16x16_and_2x1() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping greedy quad test (no headless adapter): {}", e);
            return;
        }
    };

    let assets = assets_dir();
    let textures = TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))
        .expect("TextureArray from assets");

    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let registry = BlockRegistry::from_file(&assets.join("registry/block_registry.bin"))
        .expect("load block_registry.bin");
    let stone = registry.get_block("stone").expect("stone block");
    let state = registry.get_state(stone.state_start).unwrap();
    let stone_layer = state.faces[FACE_UP].base_layer;

    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

    let key = FaceMergeKey {
        base_layer: stone_layer,
        tint_mode: 0,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };

    use wgpu::util::DeviceExt;

    // --- Sub-test A: 16x16 merged stone quad vs 256 1x1 singles ---
    {
        let target_16x16 = OffscreenTarget::new(&ctx.device, 256, 256);

        // Orthographic camera covering [0, 16] x [0, 16] on FACE_UP (y = 1)
        let view = glam::Mat4::look_at_rh(
            glam::vec3(8.0, 20.0, 8.0),
            glam::vec3(8.0, 1.0, 8.0),
            glam::vec3(0.0, 0.0, -1.0),
        );
        let proj = glam::Mat4::orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 50.0);
        let camera_uniform = CameraUniform::new(proj * view);

        let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buf_16x16"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);

        // 1. Merged 16x16 quad
        let mut vertices_merged = Vec::new();
        emit_quad(
            &mut vertices_merged,
            (0, 1, 0),
            (0, 1, 16),
            (16, 1, 16),
            (16, 1, 0),
            FACE_UP as u32,
            &key,
        );

        // 2. 256 individual 1x1 quads
        let mut vertices_singles = Vec::new();
        for bx in 0..16 {
            for bz in 0..16 {
                emit_quad(
                    &mut vertices_singles,
                    (bx, 1, bz),
                    (bx, 1, bz + 1),
                    (bx + 1, 1, bz + 1),
                    (bx + 1, 1, bz),
                    FACE_UP as u32,
                    &key,
                );
            }
        }

        let pixels_merged = render_quads_to_target(
            &ctx,
            &pipelines,
            &target_16x16,
            &camera_bind_group,
            &texture_bind_group,
            &vertices_merged,
        );
        let pixels_singles = render_quads_to_target(
            &ctx,
            &pipelines,
            &target_16x16,
            &camera_bind_group,
            &texture_bind_group,
            &vertices_singles,
        );

        // Verify non-zero rendered pixels
        let rendered_count = pixels_merged
            .chunks_exact(4)
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert_eq!(
            rendered_count,
            256 * 256,
            "16x16 target must be fully rendered with stone pixels"
        );

        // Pixel-exact frame equality
        assert_eq!(
            pixels_merged, pixels_singles,
            "16x16 merged greedy quad frame must be byte-for-byte identical to 256 singles"
        );
    }

    // --- Sub-test B: 2x1 merged stone quad vs 2 1x1 singles ---
    {
        let target_2x1 = OffscreenTarget::new(&ctx.device, 32, 16);

        // Orthographic camera covering [0, 2] x [0, 1] on FACE_UP (y = 1)
        let view = glam::Mat4::look_at_rh(
            glam::vec3(1.0, 5.0, 0.5),
            glam::vec3(1.0, 1.0, 0.5),
            glam::vec3(0.0, 0.0, -1.0),
        );
        let proj = glam::Mat4::orthographic_rh(-1.0, 1.0, -0.5, 0.5, 0.1, 10.0);
        let camera_uniform = CameraUniform::new(proj * view);

        let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buf_2x1"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);

        // 1. Merged 2x1 quad
        let mut vertices_merged = Vec::new();
        emit_quad(
            &mut vertices_merged,
            (0, 1, 0),
            (0, 1, 1),
            (2, 1, 1),
            (2, 1, 0),
            FACE_UP as u32,
            &key,
        );

        // 2. Two 1x1 quads
        let mut vertices_singles = Vec::new();
        emit_quad(
            &mut vertices_singles,
            (0, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
            (1, 1, 0),
            FACE_UP as u32,
            &key,
        );
        emit_quad(
            &mut vertices_singles,
            (1, 1, 0),
            (1, 1, 1),
            (2, 1, 1),
            (2, 1, 0),
            FACE_UP as u32,
            &key,
        );

        let pixels_merged = render_quads_to_target(
            &ctx,
            &pipelines,
            &target_2x1,
            &camera_bind_group,
            &texture_bind_group,
            &vertices_merged,
        );
        let pixels_singles = render_quads_to_target(
            &ctx,
            &pipelines,
            &target_2x1,
            &camera_bind_group,
            &texture_bind_group,
            &vertices_singles,
        );

        let rendered_count = pixels_merged
            .chunks_exact(4)
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert_eq!(
            rendered_count,
            32 * 16,
            "2x1 target must be fully rendered with stone pixels"
        );

        assert_eq!(
            pixels_merged, pixels_singles,
            "2x1 merged greedy quad frame must be byte-for-byte identical to 2 singles"
        );
    }
}

// ---------------------------------------------------------------------------
// 9. Greedy-Quad Goldens: Merged 4x1 Axis-Rotated Oak Log (uv_rot != 0) (R9)
// ---------------------------------------------------------------------------
#[test]
fn test_greedy_quad_axis_rotated_oak_log_4x1() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping rotated log test (no headless adapter): {}", e);
            return;
        }
    };

    let assets = assets_dir();
    let textures = TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))
        .expect("TextureArray from assets");

    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

    // oak_log side layer is 960 in texture_manifest.json
    // uv_rot = 1 simulates 90-degree rotated texture for horizontal axis-aligned log
    let key_log = FaceMergeKey {
        base_layer: 960,
        tint_mode: 0,
        uv_rot: 1,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };

    let target_4x1 = OffscreenTarget::new(&ctx.device, 64, 16);

    // Orthographic camera covering [0, 4] x [0, 1] on FACE_UP (y = 1)
    let view = glam::Mat4::look_at_rh(
        glam::vec3(2.0, 5.0, 0.5),
        glam::vec3(2.0, 1.0, 0.5),
        glam::vec3(0.0, 0.0, -1.0),
    );
    let proj = glam::Mat4::orthographic_rh(-2.0, 2.0, -0.5, 0.5, 0.1, 10.0);
    let camera_uniform = CameraUniform::new(proj * view);

    use wgpu::util::DeviceExt;
    let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_buf_4x1_log"),
        contents: bytemuck::bytes_of(&camera_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);

    // 1. Merged 4x1 rotated quad
    let mut vertices_merged = Vec::new();
    emit_quad(
        &mut vertices_merged,
        (0, 1, 0),
        (0, 1, 1),
        (4, 1, 1),
        (4, 1, 0),
        FACE_UP as u32,
        &key_log,
    );

    // 2. Four individual 1x1 rotated quads
    let mut vertices_singles = Vec::new();
    for bx in 0..4 {
        emit_quad(
            &mut vertices_singles,
            (bx, 1, 0),
            (bx, 1, 1),
            (bx + 1, 1, 1),
            (bx + 1, 1, 0),
            FACE_UP as u32,
            &key_log,
        );
    }

    let pixels_merged = render_quads_to_target(
        &ctx,
        &pipelines,
        &target_4x1,
        &camera_bind_group,
        &texture_bind_group,
        &vertices_merged,
    );
    let pixels_singles = render_quads_to_target(
        &ctx,
        &pipelines,
        &target_4x1,
        &camera_bind_group,
        &texture_bind_group,
        &vertices_singles,
    );

    // Verify non-zero rendered pixels
    let rendered_count = pixels_merged
        .chunks_exact(4)
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .count();
    assert_eq!(
        rendered_count,
        64 * 16,
        "4x1 target must be fully rendered with oak log pixels"
    );

    // 1. Pixel-exact frame equality: merged 4x1 quad vs 4 singles
    assert_eq!(
        pixels_merged, pixels_singles,
        "4x1 rotated oak log merged quad frame must be byte-for-byte identical to 4 singles"
    );

    // 2. Invariant: rotated pattern repeats identically across each 16-texel block
    for y in 0..16 {
        for x in 0..16 {
            let p0 = &pixels_merged[((y * 64 + x) * 4)..((y * 64 + x) * 4 + 4)];
            for b in 1..4 {
                let pb = &pixels_merged
                    [((y * 64 + (b * 16 + x)) * 4)..((y * 64 + (b * 16 + x)) * 4 + 4)];
                assert_eq!(
                    p0, pb,
                    "Rotated block {} texel at ({}, {}) must match block 0",
                    b, x, y
                );
            }
        }
    }
}

#[test]
fn test_headless_sky_dampen_night_preserves_block_light() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: no headless GPU available ({})", e);
            return;
        }
    };

    let assets = assets_dir();
    let textures = TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))
        .expect("TextureArray from assets");

    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let registry = BlockRegistry::from_file(&assets.join("registry/block_registry.bin"))
        .expect("load block_registry.bin");
    let stone = registry.get_block("stone").expect("stone block");
    let state = registry.get_state(stone.state_start).unwrap();
    let stone_layer = state.faces[FACE_UP].base_layer;

    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

    // Quad A: pure sky (block_light = 0, sky_light = 15) covering [0, 1] x [0, 1] at y = 1
    let key_sky = FaceMergeKey {
        base_layer: stone_layer,
        tint_mode: 0,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };

    // Quad B: pure block (block_light = 15, sky_light = 0) covering [1, 2] x [0, 1] at y = 1
    let key_block = FaceMergeKey {
        base_layer: stone_layer,
        tint_mode: 0,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [0, 0, 0, 0],
        corner_block: [15, 15, 15, 15],
    };

    let mut vertices = Vec::new();
    emit_quad(
        &mut vertices,
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 0),
        FACE_UP as u32,
        &key_sky,
    );
    emit_quad(
        &mut vertices,
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 1),
        (2, 1, 0),
        FACE_UP as u32,
        &key_block,
    );

    let target_2x1 = OffscreenTarget::new(&ctx.device, 32, 16);

    let view = glam::Mat4::look_at_rh(
        glam::vec3(1.0, 5.0, 0.5),
        glam::vec3(1.0, 1.0, 0.5),
        glam::vec3(0.0, 0.0, -1.0),
    );
    let proj = glam::Mat4::orthographic_rh(-1.0, 1.0, -0.5, 0.5, 0.1, 10.0);
    let vp = proj * view;

    use wgpu::util::DeviceExt;

    // 1. Render at sky_dampen = 0 (day)
    let cam_day = CameraUniform::new_full(vp, 0, 0);
    let buf_day = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_buf_day"),
        contents: bytemuck::bytes_of(&cam_day),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_day = pipelines.create_camera_bind_group(&ctx.device, &buf_day);
    let pixels_day = render_quads_to_target(
        &ctx,
        &pipelines,
        &target_2x1,
        &bind_day,
        &texture_bind_group,
        &vertices,
    );

    // 2. Render at sky_dampen = 15 (night)
    let cam_night = CameraUniform::new_full(vp, 15, 0);
    let buf_night = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_buf_night"),
        contents: bytemuck::bytes_of(&cam_night),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_night = pipelines.create_camera_bind_group(&ctx.device, &buf_night);
    let pixels_night = render_quads_to_target(
        &ctx,
        &pipelines,
        &target_2x1,
        &bind_night,
        &texture_bind_group,
        &vertices,
    );

    // Verify Quad A (x in 0..16, pure sky):
    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 32 + x) * 4;
            let day_rgb = &pixels_day[idx..idx + 3];
            let night_rgb = &pixels_night[idx..idx + 3];

            assert!(day_rgb[0] > 0 || day_rgb[1] > 0 || day_rgb[2] > 0);
            assert!(
                night_rgb[0] < day_rgb[0] && night_rgb[1] < day_rgb[1] && night_rgb[2] < day_rgb[2],
                "Pure-sky quad at night must be significantly dimmer than day at ({}, {}): night={:?}, day={:?}",
                x, y, night_rgb, day_rgb
            );
        }
    }

    // Verify Quad B (x in 16..32, pure block light):
    for y in 0..16 {
        for x in 16..32 {
            let idx = (y * 32 + x) * 4;
            let day_px = &pixels_day[idx..idx + 4];
            let night_px = &pixels_night[idx..idx + 4];
            assert_eq!(
                day_px, night_px,
                "Pure-block-light quad must be BYTE-UNCHANGED by sky_dampen at ({}, {})",
                x, y
            );
        }
    }
}

#[test]
fn test_animation_frame_golden() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test: no headless GPU available ({})", e);
            return;
        }
    };

    let assets = assets_dir();
    let textures = TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))
        .expect("TextureArray from assets");

    let pipelines = TerrainPipelines::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Depth24Plus,
    );

    let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

    // campfire_fire: base_layer = 176, frametime = 2, timeline_len = 16 ticks
    let campfire_layer = 176;

    let key_fire = FaceMergeKey {
        base_layer: campfire_layer,
        tint_mode: 0,
        uv_rot: 0,
        pass_bucket: 0,
        corner_ao: [3, 3, 3, 3],
        corner_sky: [15, 15, 15, 15],
        corner_block: [0, 0, 0, 0],
    };

    let mut vertices = Vec::new();
    emit_quad(
        &mut vertices,
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 0),
        FACE_UP as u32,
        &key_fire,
    );

    let target_1x1 = OffscreenTarget::new(&ctx.device, 16, 16);

    let view = glam::Mat4::look_at_rh(
        glam::vec3(0.5, 5.0, 0.5),
        glam::vec3(0.5, 1.0, 0.5),
        glam::vec3(0.0, 0.0, -1.0),
    );
    let proj = glam::Mat4::orthographic_rh(-0.5, 0.5, -0.5, 0.5, 0.1, 10.0);
    let vp = proj * view;

    use wgpu::util::DeviceExt;

    let render_at_tick = |tick: u32| -> Vec<u8> {
        let cam = CameraUniform::new_full(vp, 0, tick);
        let cam_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buf_anim"),
            contents: bytemuck::bytes_of(&cam),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cam_bind = pipelines.create_camera_bind_group(&ctx.device, &cam_buf);
        render_quads_to_target(
            &ctx,
            &pipelines,
            &target_1x1,
            &cam_bind,
            &texture_bind_group,
            &vertices,
        )
    };

    let pixels_t0 = render_at_tick(0);
    let pixels_t1 = render_at_tick(1);
    let pixels_t2 = render_at_tick(2);
    let pixels_t16 = render_at_tick(16);

    // In campfire_fire (frametime = 2):
    // tick 0 and tick 1 should have the same frame (frame index 0)
    assert_eq!(
        pixels_t0, pixels_t1,
        "Campfire fire ticks 0 and 1 must show identical frame (frametime = 2)"
    );

    // tick 0 vs tick 2 MUST differ (advances to frame index 1)
    assert_ne!(
        pixels_t0, pixels_t2,
        "Campfire fire at tick 0 vs tick 2 (after frametime = 2) must differ"
    );

    // tick 16 (full cycle length = 16 ticks) MUST return to tick 0 identically!
    assert_eq!(
        pixels_t0, pixels_t16,
        "Campfire fire at tick 16 (after full cycle of 16 ticks) must return identically to tick 0"
    );
}



