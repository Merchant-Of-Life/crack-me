use std::error::Error;

/// Validates that an adapter satisfies the minimum hardware limits required by the engine.
/// Specifically, texture array support for >= 2048 layers is mandatory for texture array batching.
pub fn validate_adapter_limits(limits: &wgpu::Limits) -> Result<(), String> {
    const REQUIRED_ARRAY_LAYERS: u32 = 2048;
    if limits.max_texture_array_layers < REQUIRED_ARRAY_LAYERS {
        return Err(format!(
            "wgpu adapter limit failure: max_texture_array_layers is {} (required >= {})",
            limits.max_texture_array_layers, REQUIRED_ARRAY_LAYERS
        ));
    }
    Ok(())
}

/// Headless GPU context wrapping wgpu Instance, Adapter, Device, and Queue.
/// Executes with zero window / display dependencies.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn new_headless() -> Result<Self, Box<dyn Error>> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(desc);

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: true,
                ..Default::default()
            }))
        })?;

        let limits = adapter.limits();
        validate_adapter_limits(&limits)?;

        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_texture_array_layers = 2048;
        let required_limits = required_limits.using_resolution(adapter.limits());

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("voxel_engine_headless_device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Windowed GPU context with surface initialization, strict sRGB validation,
    /// and 2048 texture array layer limits discipline.
    pub fn new_windowed(
        window: std::sync::Arc<winit::window::Window>,
    ) -> Result<(Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration), Box<dyn Error>> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: Some(&surface),
                force_fallback_adapter: true,
                ..Default::default()
            }))
        })?;

        let limits = adapter.limits();
        validate_adapter_limits(&limits)?;

        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_texture_array_layers = 2048;
        let required_limits = required_limits.using_resolution(adapter.limits());

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("voxel_engine_windowed_device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        assert!(format.is_srgb(), "Surface format {:?} must be sRGB", format);

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::AutoVsync) {
            wgpu::PresentMode::AutoVsync
        } else {
            caps.present_modes[0]
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: Default::default(),
        };
        surface.configure(&device, &config);

        let adapter_info = adapter.get_info();
        println!(
            "Startup: {} | {:?} | {:?}",
            adapter_info.name, adapter_info.backend, format
        );

        Ok((
            Self {
                instance,
                adapter,
                device,
                queue,
            },
            surface,
            config,
        ))
    }
}
