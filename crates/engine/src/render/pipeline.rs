use crate::PackedVoxelVertex;
use crate::render::TextureArray;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct SectionInstance {
    pub world_offset: [f32; 3],
    pub _pad: u32,
}

impl SectionInstance {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            world_offset: [x, y, z],
            _pad: 0,
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
            offset: 0,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x3,
        }];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SectionInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }
}

pub fn voxel_vertex_desc() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Uint32, // word0
        },
        wgpu::VertexAttribute {
            offset: 4,
            shader_location: 1,
            format: wgpu::VertexFormat::Uint32, // word1
        },
    ];
    wgpu::VertexBufferLayout {
        array_stride: 8 as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}

/// CPU reference unpack mirroring the WGSL shader unpack bit-for-bit (SPECS §4).
#[derive(Debug, Clone, PartialEq)]
pub struct UnpackedVoxelVertex {
    pub local_x: f32,
    pub local_y: f32,
    pub local_z: f32,
    pub normal_id: u32,
    pub tint_mode: u32,
    pub uv_rot: u32,
    pub block_light: f32,
    pub sky_light: f32,
    pub ao: f32,
    pub base_layer: u32,
    pub dry_foliage: bool,
}

impl UnpackedVoxelVertex {
    pub fn from_packed(v: PackedVoxelVertex) -> Self {
        let dry_foliage = v.dry_foliage();
        let tint_mode = if dry_foliage { 4 } else { v.tint_mode() };
        Self {
            local_x: (v.word0 & 0x1F) as f32,
            local_y: ((v.word0 >> 5) & 0x1F) as f32,
            local_z: ((v.word0 >> 10) & 0x1F) as f32,
            normal_id: (v.word0 >> 15) & 0x07,
            tint_mode,
            uv_rot: (v.word0 >> 20) & 0x03,
            block_light: (v.word1 & 0x0F) as f32,
            sky_light: ((v.word1 >> 4) & 0x0F) as f32,
            ao: ((v.word1 >> 8) & 0x03) as f32,
            base_layer: (v.word1 >> 10) & 0x07FF,
            dry_foliage,
        }
    }
}

pub struct TerrainPipelines {
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pass1_opaque_cutout: wgpu::RenderPipeline,
    pub pass2_translucent: wgpu::RenderPipeline,
}

impl TerrainPipelines {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/unpack.wgsl"),
                    "\n",
                    include_str!("shaders/terrain.wgsl")
                )
                .into(),
            ),
        });

        // Group 0: Camera Uniform
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Group 1: Texture2DArray + Sampler
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pipeline_layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let vertex_buffers = [Some(voxel_vertex_desc()), Some(SectionInstance::desc())];

        // Pass 1: Opaque & Cutout
        // - CCW front-face, Back culling
        // - Depth write enabled, LessEqual compare
        // - Alpha cutoff 0.5 evaluated in fragment shader
        let pass1_opaque_cutout = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pass1_opaque_cutout_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_pass1_opaque_cutout"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Pass 2: Translucent
        // - CCW front-face, cull_mode: None (two-sided)
        // - Depth write disabled, LessEqual compare
        // - Standard alpha blending
        let pass2_translucent = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pass2_translucent_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Two-sided rendering for translucent surfaces
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_pass2_translucent"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            camera_bind_group_layout,
            texture_bind_group_layout,
            pipeline_layout,
            pass1_opaque_cutout,
            pass2_translucent,
        }
    }

    pub fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &self.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        })
    }

    pub fn create_texture_bind_group(
        &self,
        device: &wgpu::Device,
        texture_array: &TextureArray,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_array.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_array.sampler),
                },
            ],
        })
    }
}
