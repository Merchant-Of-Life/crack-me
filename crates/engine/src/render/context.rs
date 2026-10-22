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
}
