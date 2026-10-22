use std::collections::HashSet;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::bytes_of;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use voxel_engine::bfs::{recompute_light_column, LightComputeResult, NeighborChunkLights};
use voxel_engine::light::ChunkVoxelColumn;
use voxel_engine::mesher::{mesh_section, OptionalHorizontalNeighbors, SectionHalo};
use voxel_engine::physics::{
    tick_player, PlayerInput, PlayerPhysicsState, VoxelWorldCollision, EYE_HEIGHT,
};
use voxel_engine::registry::BlockRegistry;
use voxel_engine::render::{
    CameraUniform, GpuContext, SectionInstance, TerrainPipelines, TextureArray,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MovementMode {
    Physics,
    Fly,
}

pub struct ViewerWorld {
    pub voxels: Vec<Vec<ChunkVoxelColumn>>,
    pub reg: Arc<BlockRegistry>,
}

impl VoxelWorldCollision for ViewerWorld {
    fn is_solid_at(&self, x: i32, y: i32, z: i32) -> bool {
        let cx = x.div_euclid(16);
        let cz = z.div_euclid(16);
        if cx >= 0 && cx < 6 && cz >= 0 && cz < 6 && y >= -64 && y < 320 {
            let rel_x = x.rem_euclid(16) as usize;
            let rel_z = z.rem_euclid(16) as usize;
            let state_id = self.voxels[cx as usize][cz as usize].get_state(rel_x, y, rel_z);
            self.reg.is_solid(state_id as u32)
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct AssetDirectoryNotFound {
    pub searched: Vec<PathBuf>,
}

impl std::fmt::Display for AssetDirectoryNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Assets directory not found. Searched locations: {:?}. Pass --assets-dir <path> or set ASSETS_DIR environment variable.",
            self.searched
        )
    }
}

impl std::error::Error for AssetDirectoryNotFound {}

fn assets_dir() -> Result<PathBuf, AssetDirectoryNotFound> {
    let mut searched = Vec::new();

    // 1. Explicit CLI argument --assets-dir <PATH> or --assets-dir=<PATH> or --assets <PATH>
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--assets-dir" || arg == "--assets" {
            if let Some(val) = args.next() {
                let p = PathBuf::from(&val);
                searched.push(p.clone());
                if p.join("registry/block_registry.bin").exists() {
                    return Ok(p);
                }
            }
        } else if let Some(stripped) = arg.strip_prefix("--assets-dir=") {
            let p = PathBuf::from(stripped);
            searched.push(p.clone());
            if p.join("registry/block_registry.bin").exists() {
                return Ok(p);
            }
        }
    }

    // 2. Explicit environment variable ASSETS_DIR
    if let Ok(env_dir) = std::env::var("ASSETS_DIR") {
        let p = PathBuf::from(env_dir);
        searched.push(p.clone());
        if p.join("registry/block_registry.bin").exists() {
            return Ok(p);
        }
    }

    // 3. Ancestors of current_exe()
    if let Ok(exe_path) = std::env::current_exe() {
        for ancestor in exe_path.ancestors() {
            let p = ancestor.join("assets");
            searched.push(p.clone());
            if p.join("registry/block_registry.bin").exists() {
                return Ok(p);
            }
        }
    }

    // 4. CARGO_MANIFEST_DIR ancestors
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest_dir).join("../../assets");
        searched.push(p.clone());
        if p.join("registry/block_registry.bin").exists() {
            return Ok(p);
        }
    }

    // 5. CWD-relative ./assets and ../../assets
    let p_rel = PathBuf::from("assets");
    searched.push(p_rel.clone());
    if p_rel.join("registry/block_registry.bin").exists() {
        return Ok(p_rel);
    }

    let p_parent = PathBuf::from("../../assets");
    searched.push(p_parent.clone());
    if p_parent.join("registry/block_registry.bin").exists() {
        return Ok(p_parent);
    }

    Err(AssetDirectoryNotFound { searched })
}

struct SectionMesh {
    opaque_buf: Option<wgpu::Buffer>,
    opaque_vertex_count: u32,
    translucent_buf: Option<wgpu::Buffer>,
    translucent_vertex_count: u32,
    instance_buf: wgpu::Buffer,
}

struct AppState {
    window: Arc<Window>,
    ctx: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    pipelines: TerrainPipelines,
    texture_bind_group: wgpu::BindGroup,
    camera_buf: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    sections: Vec<SectionMesh>,
    camera_pos: glam::Vec3,
    camera_yaw: f32,
    camera_pitch: f32,
    sky_dampen: u32,
    keys_pressed: HashSet<KeyCode>,
    cursor_grabbed: bool,
    last_frame_time: Instant,
    fps_timer: Instant,
    frame_count: u32,
    startup_line_printed: bool,
    movement_mode: MovementMode,
    player_physics: PlayerPhysicsState,
    world: ViewerWorld,
    tick_accumulator: f64,
    tick_count: u32,
}

impl AppState {
    fn set_cursor_grab(&mut self, grab: bool) {
        if grab {
            match self.window.set_cursor_grab(CursorGrabMode::Locked) {
                Ok(()) => {
                    self.window.set_cursor_visible(false);
                    self.cursor_grabbed = true;
                }
                Err(e) => {
                    eprintln!("Failed to grab cursor: {e}");
                    self.cursor_grabbed = false;
                }
            }
        } else {
            if let Err(e) = self.window.set_cursor_grab(CursorGrabMode::None) {
                eprintln!("Failed to release cursor grab: {e}");
            }
            self.window.set_cursor_visible(true);
            self.cursor_grabbed = false;
        }
    }
    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer_depth_texture"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        (depth_texture, depth_view)
    }

    fn new(window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let (ctx, surface, config) = GpuContext::new_windowed(window.clone())?;
        let (depth_texture, depth_view) = Self::create_depth_texture(&ctx.device, &config);

        let assets = assets_dir()?;
        let textures =
            TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))?;

        let pipelines = TerrainPipelines::new(
            &ctx.device,
            config.format,
            wgpu::TextureFormat::Depth24Plus,
        );

        let texture_bind_group = pipelines.create_texture_bind_group(&ctx.device, &textures);

        let dummy_cam = CameraUniform::new(glam::Mat4::IDENTITY);
        let camera_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_uniform_buf"),
            contents: bytes_of(&dummy_cam),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = pipelines.create_camera_bind_group(&ctx.device, &camera_buf);

        // Load block registry
        let reg = Arc::new(BlockRegistry::from_file(&assets.join("registry/block_registry.bin"))?);
        let air_id = reg.get_block("air").map(|b| b.state_start as u16).unwrap_or(0);
        let stone_id = reg.get_block("stone").map(|b| b.state_start as u16).unwrap_or(0);
        let grass_id = reg
            .get_block("grass_block")
            .map(|b| b.state_start as u16)
            .unwrap_or(stone_id);
        let dirt_id = reg.get_block("dirt").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let glowstone_id = reg
            .get_block("glowstone")
            .map(|b| b.state_start as u16)
            .unwrap_or(stone_id);
        let magma_id = reg
            .get_block("magma_block")
            .map(|b| b.state_start as u16)
            .unwrap_or(stone_id);
        let lava_id = reg.get_block("lava").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let packed_ice_id = reg
            .get_block("packed_ice")
            .map(|b| b.state_start as u16)
            .unwrap_or(stone_id);
        let ice_id = reg.get_block("ice").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let glass_id = reg.get_block("glass").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let stained_glass_id = reg
            .get_block("white_stained_glass")
            .or_else(|| reg.get_block("cyan_stained_glass"))
            .map(|b| b.state_start as u16)
            .unwrap_or(glass_id);

        let furnace_lit_id = {
            if let Some(furnace) = reg.get_block("furnace") {
                let mut found = furnace.state_start as u16;
                for s_id in furnace.state_start..furnace.state_start + furnace.state_count {
                    if let Some(s) = reg.get_state(s_id) {
                        if s.light_emission == 13 {
                            found = s_id as u16;
                            break;
                        }
                    }
                }
                found
            } else {
                stone_id
            }
        };

        let (log_x_id, log_z_id) = {
            if let Some(log) = reg.get_block("oak_log") {
                (log.state_start as u16, (log.state_start + 2) as u16)
            } else {
                (stone_id, stone_id)
            }
        };

        // Construct 6x6 chunk column grid (cx in 0..6, cz in 0..6)
        let mut voxels: Vec<Vec<ChunkVoxelColumn>> = (0..6)
            .map(|_| (0..6).map(|_| ChunkVoxelColumn::new(air_id)).collect())
            .collect();

        // Populate base terrain across all 36 columns:
        // Section 4 corresponds to world_y in 0..16
        // Base layers: y = 0..2 stone, y = 3 dirt, y = 4 grass
        for cx in 0..6 {
            for cz in 0..6 {
                for x in 0..16 {
                    for z in 0..16 {
                        for y in 0..3 {
                            voxels[cx][cz].set_state(x, y, z, stone_id);
                        }
                        voxels[cx][cz].set_state(x, 3, z, dirt_id);
                        voxels[cx][cz].set_state(x, 4, z, grass_id);
                    }
                }
            }
        }

        // Populate showcase features in chunk (2, 2) (inner column):
        // 1. Lava pool & magma block (sunken at y = 4):
        for x in 3..=5 {
            for z in 3..=5 {
                voxels[2][2].set_state(x, 3, z, stone_id); // floor
                voxels[2][2].set_state(x, 4, z, lava_id);  // animated fluid
            }
        }
        // Border magma blocks around lava pool:
        for x in 2..=6 {
            voxels[2][2].set_state(x, 4, 2, magma_id);
            voxels[2][2].set_state(x, 4, 6, magma_id);
        }
        for z in 3..=5 {
            voxels[2][2].set_state(2, 4, z, magma_id);
            voxels[2][2].set_state(6, 4, z, magma_id);
        }

        // 2. Glowstone pillar and lit furnace:
        voxels[2][2].set_state(8, 5, 3, glowstone_id);
        voxels[2][2].set_state(8, 6, 3, glowstone_id);
        voxels[2][2].set_state(9, 5, 3, furnace_lit_id);

        // 3. Packed ice hill with translucent ice peak:
        for x in 11..=14 {
            for z in 11..=14 {
                voxels[2][2].set_state(x, 5, z, packed_ice_id);
            }
        }
        for x in 12..=13 {
            for z in 12..=13 {
                voxels[2][2].set_state(x, 6, z, packed_ice_id);
                voxels[2][2].set_state(x, 7, z, ice_id); // translucent ice peak
            }
        }

        // 4. Clear glass & stained glass structures:
        for x in 2..=5 {
            voxels[2][2].set_state(x, 5, 12, glass_id);
            voxels[2][2].set_state(x, 6, 12, glass_id);
            voxels[2][2].set_state(x, 7, 12, stained_glass_id);
        }

        // 5. Sideways-axis logs:
        // Axis X at z = 8
        for x in 2..=6 {
            voxels[2][2].set_state(x, 5, 8, log_x_id);
        }
        // Axis Z at x = 9
        for z in 7..=11 {
            voxels[2][2].set_state(9, 5, z, log_z_id);
        }

        // 2-pass light recomputation with NeighborChunkLights boundary seeding (AM-4)
        // Pass 1: compute base light per column without neighbors
        let mut lights: Vec<Vec<LightComputeResult>> = (0..6)
            .map(|cx| {
                (0..6)
                    .map(|cz| recompute_light_column(&voxels[cx][cz], &reg, None, None))
                    .collect()
            })
            .collect();

        // Pass 2: seed boundaries across adjacent columns and recompute
        let mut pass2_lights: Vec<Vec<LightComputeResult>> = Vec::with_capacity(6);
        for cx in 0..6 {
            let mut row = Vec::with_capacity(6);
            for cz in 0..6 {
                let nbrs = NeighborChunkLights {
                    west: if cx > 0 { Some(&lights[cx - 1][cz].light) } else { None },
                    east: if cx + 1 < 6 { Some(&lights[cx + 1][cz].light) } else { None },
                    north: if cz > 0 { Some(&lights[cx][cz - 1].light) } else { None },
                    south: if cz + 1 < 6 { Some(&lights[cx][cz + 1].light) } else { None },
                };
                let res = recompute_light_column(
                    &voxels[cx][cz],
                    &reg,
                    Some(&nbrs),
                    Some(&lights[cx][cz].light),
                );
                row.push(res);
            }
            pass2_lights.push(row);
        }
        lights = pass2_lights;

        // Mesh inner 4x4 sections guarded by OptionalHorizontalNeighbors (wait-for-8 halo gate)
        let mut sections = Vec::new();
        let mut total_opaque_vertices = 0u32;
        let mut total_translucent_vertices = 0u32;
        let mut total_quads = 0usize;

        for cx in 0..6 {
            for cz in 0..6 {
                let opt_nbrs = OptionalHorizontalNeighbors {
                    west: if cx > 0 { Some((&voxels[cx - 1][cz], &lights[cx - 1][cz].light)) } else { None },
                    east: if cx + 1 < 6 { Some((&voxels[cx + 1][cz], &lights[cx + 1][cz].light)) } else { None },
                    north: if cz > 0 { Some((&voxels[cx][cz - 1], &lights[cx][cz - 1].light)) } else { None },
                    south: if cz + 1 < 6 { Some((&voxels[cx][cz + 1], &lights[cx][cz + 1].light)) } else { None },
                    nw: if cx > 0 && cz > 0 { Some((&voxels[cx - 1][cz - 1], &lights[cx - 1][cz - 1].light)) } else { None },
                    ne: if cx + 1 < 6 && cz > 0 { Some((&voxels[cx + 1][cz - 1], &lights[cx + 1][cz - 1].light)) } else { None },
                    sw: if cx > 0 && cz + 1 < 6 { Some((&voxels[cx - 1][cz + 1], &lights[cx - 1][cz + 1].light)) } else { None },
                    se: if cx + 1 < 6 && cz + 1 < 6 { Some((&voxels[cx + 1][cz + 1], &lights[cx + 1][cz + 1].light)) } else { None },
                };

                // Wait-for-8 halo gate: mesh ONLY when all 8 horizontal neighbors exist
                if let Some(neighbors) = opt_nbrs.as_complete() {
                    let mut halo = SectionHalo::new();
                    // Section 4 corresponds to y in 0..16 (base_y = 0)
                    halo.populate(4, &voxels[cx][cz], &lights[cx][cz].light, &neighbors, air_id);
                    let mesh_res = mesh_section(&halo, &reg);

                    total_opaque_vertices += mesh_res.opaque.len() as u32;
                    total_translucent_vertices += mesh_res.translucent.len() as u32;
                    total_quads += mesh_res.quad_count();

                    let world_x = cx as f32 * 16.0;
                    let world_z = cz as f32 * 16.0;
                    let instance = SectionInstance::new(world_x, 0.0, world_z);
                    let instance_buf =
                        ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("section_instance_buf_{cx}_{cz}")),
                            contents: bytes_of(&instance),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let opaque_buf = if !mesh_res.opaque.is_empty() {
                        let v_bytes: Vec<u8> = mesh_res
                            .opaque
                            .iter()
                            .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
                            .collect();
                        Some(
                            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some(&format!("opaque_vertex_buf_{cx}_{cz}")),
                                contents: &v_bytes,
                                usage: wgpu::BufferUsages::VERTEX,
                            }),
                        )
                    } else {
                        None
                    };

                    let translucent_buf = if !mesh_res.translucent.is_empty() {
                        let v_bytes: Vec<u8> = mesh_res
                            .translucent
                            .iter()
                            .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
                            .collect();
                        Some(
                            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some(&format!("translucent_vertex_buf_{cx}_{cz}")),
                                contents: &v_bytes,
                                usage: wgpu::BufferUsages::VERTEX,
                            }),
                        )
                    } else {
                        None
                    };

                    sections.push(SectionMesh {
                        opaque_buf,
                        opaque_vertex_count: mesh_res.opaque.len() as u32,
                        translucent_buf,
                        translucent_vertex_count: mesh_res.translucent.len() as u32,
                        instance_buf,
                    });
                }
            }
        }

        println!(
            "Showcase 6x6 grid meshed (4x4 inner sections): {} opaque vertices, {} translucent vertices ({} total quads)",
            total_opaque_vertices, total_translucent_vertices, total_quads
        );

        let world = ViewerWorld {
            voxels,
            reg: reg.clone(),
        };

        let spawn_pos = glam::DVec3::new(48.0, 10.0, 48.0);
        let player_physics = PlayerPhysicsState::new(spawn_pos);
        let camera_pos = glam::vec3(48.0, (10.0 + EYE_HEIGHT) as f32, 48.0);

        let now = Instant::now();
        Ok(Self {
            window,
            ctx,
            surface,
            config,
            depth_texture,
            depth_view,
            pipelines,
            texture_bind_group,
            camera_buf,
            camera_bind_group,
            sections,
            camera_pos,
            camera_yaw: 0.0,
            camera_pitch: -0.2,
            sky_dampen: 0,
            keys_pressed: HashSet::new(),
            cursor_grabbed: false,
            last_frame_time: now,
            fps_timer: now,
            frame_count: 0,
            startup_line_printed: false,
            movement_mode: MovementMode::Physics,
            player_physics,
            world,
            tick_accumulator: 0.0,
            tick_count: 0,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.ctx.device, &self.config);
            let (depth_texture, depth_view) =
                Self::create_depth_texture(&self.ctx.device, &self.config);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;

        // Single tick source: advance physics ticks AND camera.tick_count (20 Hz)
        // Accumulator capped at 5 substeps (5 * 0.05 = 0.25 s); excess discarded
        self.tick_accumulator += dt;
        if self.tick_accumulator > 0.25 {
            self.tick_accumulator = 0.25;
        }

        let mut substeps = 0;
        while self.tick_accumulator >= 0.05 && substeps < 5 {
            if self.movement_mode == MovementMode::Physics {
                let input = PlayerInput {
                    forward: self.keys_pressed.contains(&KeyCode::KeyW),
                    backward: self.keys_pressed.contains(&KeyCode::KeyS),
                    left: self.keys_pressed.contains(&KeyCode::KeyA),
                    right: self.keys_pressed.contains(&KeyCode::KeyD),
                    jump: self.keys_pressed.contains(&KeyCode::Space),
                    sprint: self.keys_pressed.contains(&KeyCode::ControlLeft),
                    yaw: self.camera_yaw as f64,
                };
                tick_player(&mut self.player_physics, &input, &self.world);
            }
            self.tick_count = self.tick_count.wrapping_add(1);
            self.tick_accumulator -= 0.05;
            substeps += 1;
        }

        if self.movement_mode == MovementMode::Physics {
            self.camera_pos = glam::vec3(
                self.player_physics.pos.x as f32,
                (self.player_physics.pos.y + EYE_HEIGHT) as f32,
                self.player_physics.pos.z as f32,
            );
        } else {
            // [FLY] mode keeps the 8625f86 / 45e6edc contract unchanged
            let dt_f32 = (dt as f32).min(0.1);
            let sprint_mult = if self.keys_pressed.contains(&KeyCode::ControlLeft) {
                2.5
            } else {
                1.0
            };
            let speed = 12.0 * dt_f32 * sprint_mult;
            let (yaw_sin, yaw_cos) = self.camera_yaw.sin_cos();
            let forward = glam::vec3(yaw_sin, 0.0, yaw_cos).normalize_or_zero();
            let right = glam::vec3(-yaw_cos, 0.0, yaw_sin).normalize_or_zero();

            if self.keys_pressed.contains(&KeyCode::KeyW) {
                self.camera_pos += forward * speed;
            }
            if self.keys_pressed.contains(&KeyCode::KeyS) {
                self.camera_pos -= forward * speed;
            }
            if self.keys_pressed.contains(&KeyCode::KeyA) {
                self.camera_pos -= right * speed;
            }
            if self.keys_pressed.contains(&KeyCode::KeyD) {
                self.camera_pos += right * speed;
            }
            if self.keys_pressed.contains(&KeyCode::Space) {
                self.camera_pos.y += speed;
            }
            if self.keys_pressed.contains(&KeyCode::ShiftLeft)
                || self.keys_pressed.contains(&KeyCode::ShiftRight)
            {
                self.camera_pos.y -= speed;
            }
        }
    }

    fn render(&mut self) -> Result<(), &'static str> {
        self.update();

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.resize(self.config.width, self.config.height);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("Validation error getting current surface texture");
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let aspect = (self.config.width as f32) / (self.config.height as f32).max(1.0);
        let fov_y = 70.0f32.to_radians();
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, 0.1, 1000.0);

        let (pitch_sin, pitch_cos) = self.camera_pitch.sin_cos();
        let (yaw_sin, yaw_cos) = self.camera_yaw.sin_cos();
        let look_dir = glam::vec3(yaw_sin * pitch_cos, pitch_sin, yaw_cos * pitch_cos).normalize();
        let target = self.camera_pos + look_dir;
        let view_mat = glam::Mat4::look_at_rh(self.camera_pos, target, glam::Vec3::Y);
        let vp = proj * view_mat;

        let camera_uniform = CameraUniform::new_full(vp, self.sky_dampen, self.tick_count);
        self.ctx.queue.write_buffer(
            &self.camera_buf,
            0,
            bytes_of(&camera_uniform),
        );

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewer_render_encoder"),
            });

        let sky_ratio = (1.0 - (self.sky_dampen as f64 / 15.0)).max(0.05);
        let clear_color = wgpu::Color {
            r: 0.52 * sky_ratio,
            g: 0.81 * sky_ratio,
            b: 0.92 * sky_ratio,
            a: 1.0,
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
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

            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_bind_group, &[]);

            // Pass 1: Opaque & Cutout
            pass.set_pipeline(&self.pipelines.pass1_opaque_cutout);
            for sec in &self.sections {
                if let Some(ref buf) = sec.opaque_buf {
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.set_vertex_buffer(1, sec.instance_buf.slice(..));
                    pass.draw(0..sec.opaque_vertex_count, 0..1);
                }
            }

            // Pass 2: Translucent
            pass.set_pipeline(&self.pipelines.pass2_translucent);
            for sec in &self.sections {
                if let Some(ref buf) = sec.translucent_buf {
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.set_vertex_buffer(1, sec.instance_buf.slice(..));
                    pass.draw(0..sec.translucent_vertex_count, 0..1);
                }
            }
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        self.ctx.queue.present(output);

        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f64();
        if elapsed >= 0.5 {
            let fps = (self.frame_count as f64) / elapsed;
            if !self.startup_line_printed {
                let info = self.ctx.adapter.get_info();
                println!(
                    "Startup: {} | {:?} | {:?} | {:.1} FPS",
                    info.name, info.backend, self.config.format, fps
                );
                self.startup_line_printed = true;
            }
            let mode_str = match self.movement_mode {
                MovementMode::Physics => "PHYSICS",
                MovementMode::Fly => "FLY",
            };
            let title = format!(
                "Minecraft 26.3 Voxel Viewer | [{}] | FPS: {:.1} | Dampen: {} | Tick: {}",
                mode_str, fps, self.sky_dampen, self.tick_count
            );
            self.window.set_title(&title);
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }

        Ok(())
    }
}

#[derive(Default)]
struct ViewerApp {
    state: Option<AppState>,
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let mut win_attrs = Window::default_attributes()
                .with_title("Minecraft 26.3 Voxel Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            #[cfg(target_os = "linux")]
            {
                use winit::platform::wayland::WindowAttributesExtWayland;
                use winit::platform::x11::WindowAttributesExtX11;
                win_attrs = WindowAttributesExtWayland::with_name(
                    win_attrs,
                    "voxel_viewer",
                    "voxel_viewer",
                );
                win_attrs = WindowAttributesExtX11::with_name(
                    win_attrs,
                    "voxel_viewer",
                    "voxel_viewer",
                );
            }
            let window = match event_loop.create_window(win_attrs) {
                Ok(w) => Arc::new(w),
                Err(e) => {
                    eprintln!("Failed to create window: {e}");
                    event_loop.exit();
                    return;
                }
            };
            match AppState::new(window) {
                Ok(state) => self.state = Some(state),
                Err(err) => {
                    eprintln!("Error initializing voxel_viewer: {err}");
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size.width, physical_size.height);
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: winit::event::MouseButton::Left | winit::event::MouseButton::Right,
                ..
            } => {
                // Click = grab pointer (AM-2)
                if button_state == ElementState::Pressed && !state.cursor_grabbed {
                    state.set_cursor_grab(true);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                if key_state == ElementState::Pressed {
                    state.keys_pressed.insert(key_code);
                    match key_code {
                        // Esc #1 = release grab; Esc #2 while ungrabbed = exit
                        KeyCode::Escape => {
                            if state.cursor_grabbed {
                                state.set_cursor_grab(false);
                            } else {
                                event_loop.exit();
                            }
                        }
                        // Controls: Dampen keys are N (+1) / M (-1)
                        KeyCode::KeyN => {
                            state.sky_dampen = (state.sky_dampen + 1).min(15);
                            println!("Sky dampen: {} / 15", state.sky_dampen);
                        }
                        KeyCode::KeyM => {
                            state.sky_dampen = state.sky_dampen.saturating_sub(1);
                            println!("Sky dampen: {} / 15", state.sky_dampen);
                        }
                        // F toggles [PHYSICS] / [FLY]
                        KeyCode::KeyF => {
                            state.movement_mode = match state.movement_mode {
                                MovementMode::Physics => {
                                    println!("Movement mode: [FLY]");
                                    MovementMode::Fly
                                }
                                MovementMode::Fly => {
                                    state.player_physics.pos = glam::DVec3::new(
                                        state.camera_pos.x as f64,
                                        (state.camera_pos.y - EYE_HEIGHT as f32) as f64,
                                        state.camera_pos.z as f64,
                                    );
                                    state.player_physics.vel = glam::DVec3::ZERO;
                                    state.player_physics.on_ground = false;
                                    println!("Movement mode: [PHYSICS]");
                                    MovementMode::Physics
                                }
                            };
                        }
                        _ => {}
                    }
                } else {
                    state.keys_pressed.remove(&key_code);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = state.render() {
                    eprintln!("Render error: {}", e);
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            // Look = pointer deltas while grabbed (Wayland/Hyprland compatible)
            if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
                if state.cursor_grabbed {
                    let sensitivity = 0.003;
                    // FPS feel: mouse right (dx>0) swings view toward screen-right (-X), i.e. yaw decreases
                    state.camera_yaw -= (dx as f32) * sensitivity;
                    state.camera_pitch -= (dy as f32) * sensitivity;
                    state.camera_pitch = state.camera_pitch.clamp(-1.5, 1.5);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting Minecraft 26.3 Voxel Viewer...");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = ViewerApp::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
