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
use winit::window::{Window, WindowId};

use voxel_engine::bfs::recompute_light_column;
use voxel_engine::light::{ChunkLightColumn, ChunkVoxelColumn};
use voxel_engine::mesher::{mesh_section, HorizontalNeighbors, SectionHalo};
use voxel_engine::registry::BlockRegistry;
use voxel_engine::render::{
    CameraUniform, GpuContext, SectionInstance, TerrainPipelines, TextureArray,
};

fn assets_dir() -> PathBuf {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest_dir).join("../../assets");
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from("assets");
    if p.exists() {
        return p;
    }
    PathBuf::from("../../assets")
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
    opaque_buf: Option<wgpu::Buffer>,
    opaque_vertex_count: u32,
    translucent_buf: Option<wgpu::Buffer>,
    translucent_vertex_count: u32,
    instance_buf: wgpu::Buffer,
    camera_pos: glam::Vec3,
    camera_yaw: f32,
    camera_pitch: f32,
    sky_dampen: u32,
    keys_pressed: HashSet<KeyCode>,
    mouse_dragging: bool,
    start_time: Instant,
    last_frame_time: Instant,
    fps_timer: Instant,
    frame_count: u32,
    startup_line_printed: bool,
}

impl AppState {
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

        let assets = assets_dir();
        let textures = TextureArray::from_assets(&ctx.device, &ctx.queue, &assets.join("textures"))?;

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

        let instance = SectionInstance::new(0.0, 0.0, 0.0);
        let instance_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("section_instance_buf"),
            contents: bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Load block registry and generate showcase chunk
        let reg = BlockRegistry::from_file(&assets.join("registry/block_registry.bin"))?;
        let air_id = reg.get_block("air").map(|b| b.state_start as u16).unwrap_or(0);
        let stone_id = reg.get_block("stone").map(|b| b.state_start as u16).unwrap_or(0);
        let grass_id = reg.get_block("grass_block").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let dirt_id = reg.get_block("dirt").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let glowstone_id = reg.get_block("glowstone").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let magma_id = reg.get_block("magma_block").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let lava_id = reg.get_block("lava").map(|b| b.state_start as u16).unwrap_or(stone_id);
        let packed_ice_id = reg.get_block("packed_ice").map(|b| b.state_start as u16).unwrap_or(stone_id);
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

        // Construct 16x16 chunk voxel column
        let mut voxels = ChunkVoxelColumn::new(air_id);

        // Section 4 is world_y in 0..16
        // Base layers: y = 0..2 stone, y = 3 dirt, y = 4 grass
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..3 {
                    voxels.set_state(x, y, z, stone_id);
                }
                voxels.set_state(x, 3, z, dirt_id);
                voxels.set_state(x, 4, z, grass_id);
            }
        }

        // Lava pool & magma block (sunken at y = 4):
        for x in 3..=5 {
            for z in 3..=5 {
                voxels.set_state(x, 3, z, stone_id); // floor
                voxels.set_state(x, 4, z, lava_id);  // animated fluid
            }
        }
        // Border magma blocks around lava pool:
        for x in 2..=6 {
            voxels.set_state(x, 4, 2, magma_id);
            voxels.set_state(x, 4, 6, magma_id);
        }
        for z in 3..=5 {
            voxels.set_state(2, 4, z, magma_id);
            voxels.set_state(6, 4, z, magma_id);
        }

        // Glowstone pillar and lit furnace:
        voxels.set_state(8, 5, 3, glowstone_id);
        voxels.set_state(8, 6, 3, glowstone_id);
        voxels.set_state(9, 5, 3, furnace_lit_id);

        // Packed ice hill with translucent ice peak:
        for x in 11..=14 {
            for z in 11..=14 {
                voxels.set_state(x, 5, z, packed_ice_id);
            }
        }
        for x in 12..=13 {
            for z in 12..=13 {
                voxels.set_state(x, 6, z, packed_ice_id);
                voxels.set_state(x, 7, z, ice_id); // translucent ice peak
            }
        }

        // Clear glass & stained glass structures:
        for x in 2..=5 {
            voxels.set_state(x, 5, 12, glass_id);
            voxels.set_state(x, 6, 12, glass_id);
            voxels.set_state(x, 7, 12, stained_glass_id);
        }

        // Sideways-axis logs:
        // Axis X at z = 8
        for x in 2..=6 {
            voxels.set_state(x, 5, 8, log_x_id);
        }
        // Axis Z at x = 9
        for z in 7..=11 {
            voxels.set_state(9, 5, z, log_z_id);
        }

        // Compute light column with sunlight from top & BFS emission
        let light_res = recompute_light_column(&voxels, &reg, None, None);

        // Build halo with empty neighbors (sunlight 15)
        let empty_vox = ChunkVoxelColumn::new(air_id);
        let mut empty_light = ChunkLightColumn::new();
        for y in -64..320 {
            for z in 0..16 {
                for x in 0..16 {
                    empty_light.set_sky_light(x, y, z, 15);
                }
            }
        }

        let nbr_col = (&empty_vox, &empty_light);
        let neighbors = HorizontalNeighbors {
            west: nbr_col,
            east: nbr_col,
            north: nbr_col,
            south: nbr_col,
            nw: nbr_col,
            ne: nbr_col,
            sw: nbr_col,
            se: nbr_col,
        };

        let mut halo = SectionHalo::new();
        // Section 4 corresponds to y in 0..16 (since base_y = -64 + 4*16 = 0)
        halo.populate(4, &voxels, &light_res.light, &neighbors, air_id);

        let mesh_res = mesh_section(&halo, &reg);
        println!(
            "Showcase chunk meshed: {} opaque vertices, {} translucent vertices ({} total quads)",
            mesh_res.opaque.len(),
            mesh_res.translucent.len(),
            mesh_res.quad_count()
        );

        let opaque_buf = if !mesh_res.opaque.is_empty() {
            let v_bytes: Vec<u8> = mesh_res
                .opaque
                .iter()
                .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
                .collect();
            Some(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opaque_vertex_buf"),
                contents: &v_bytes,
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };

        let translucent_buf = if !mesh_res.translucent.is_empty() {
            let v_bytes: Vec<u8> = mesh_res
                .translucent
                .iter()
                .flat_map(|v| [v.word0.to_ne_bytes(), v.word1.to_ne_bytes()].concat())
                .collect();
            Some(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("translucent_vertex_buf"),
                contents: &v_bytes,
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };

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
            opaque_buf,
            opaque_vertex_count: mesh_res.opaque.len() as u32,
            translucent_buf,
            translucent_vertex_count: mesh_res.translucent.len() as u32,
            instance_buf,
            camera_pos: glam::vec3(8.0, 12.0, -8.0),
            camera_yaw: 0.78, // ~45 deg
            camera_pitch: -0.4,
            sky_dampen: 0,
            keys_pressed: HashSet::new(),
            mouse_dragging: false,
            start_time: now,
            last_frame_time: now,
            fps_timer: now,
            frame_count: 0,
            startup_line_printed: false,
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
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;

        let speed = 12.0 * dt;
        let (yaw_sin, yaw_cos) = self.camera_yaw.sin_cos();
        let forward = glam::vec3(yaw_sin, 0.0, yaw_cos).normalize_or_zero();
        let right = glam::vec3(yaw_cos, 0.0, -yaw_sin).normalize_or_zero();

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
        // Controls: Space stays ASCEND
        if self.keys_pressed.contains(&KeyCode::Space) {
            self.camera_pos.y += speed;
        }
        if self.keys_pressed.contains(&KeyCode::ShiftLeft)
            || self.keys_pressed.contains(&KeyCode::ShiftRight)
        {
            self.camera_pos.y -= speed;
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

        let tick_count = (self.start_time.elapsed().as_secs_f64() * 20.0) as u32;
        let camera_uniform = CameraUniform::new_full(vp, self.sky_dampen, tick_count);
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
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));

            // Pass 1: Opaque & Cutout
            if let Some(ref buf) = self.opaque_buf {
                pass.set_pipeline(&self.pipelines.pass1_opaque_cutout);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..self.opaque_vertex_count, 0..1);
            }

            // Pass 2: Translucent
            if let Some(ref buf) = self.translucent_buf {
                pass.set_pipeline(&self.pipelines.pass2_translucent);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..self.translucent_vertex_count, 0..1);
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
            let title = format!(
                "Minecraft 26.3 Voxel Viewer | FPS: {:.1} | Dampen: {} | Tick: {}",
                fps, self.sky_dampen, tick_count
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
            let win_attrs = Window::default_attributes()
                .with_title("Minecraft 26.3 Voxel Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = Arc::new(event_loop.create_window(win_attrs).expect("create window"));
            let state = AppState::new(window).expect("initialize AppState");
            self.state = Some(state);
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
                state.mouse_dragging = button_state == ElementState::Pressed;
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
                        KeyCode::Escape => event_loop.exit(),
                        // Controls: Dampen keys are N (+1) / M (-1)
                        KeyCode::KeyN => {
                            state.sky_dampen = (state.sky_dampen + 1).min(15);
                            println!("Sky dampen: {} / 15", state.sky_dampen);
                        }
                        KeyCode::KeyM => {
                            state.sky_dampen = state.sky_dampen.saturating_sub(1);
                            println!("Sky dampen: {} / 15", state.sky_dampen);
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
            if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
                if state.mouse_dragging {
                    let sensitivity = 0.003;
                    state.camera_yaw += (dx as f32) * sensitivity;
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
