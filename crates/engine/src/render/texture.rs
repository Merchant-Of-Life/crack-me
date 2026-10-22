use std::error::Error;
use std::fs;
use std::path::Path;

/// Texture2DArray resource containing the 5 pre-baked mipmap levels of all block textures.
pub struct TextureArray {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub total_layers: u32,
}

impl TextureArray {
    /// Loads the 5 pre-baked mip binaries from `assets/textures/` and uploads them to the GPU.
    pub fn from_assets(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures_dir: &Path,
    ) -> Result<Self, Box<dyn Error>> {
        let manifest_path = textures_dir.join("texture_manifest.json");
        let manifest_text = fs::read_to_string(&manifest_path)?;
        let manifest_json: serde_json::Value = serde_json::from_str(&manifest_text)?;
        let total_layers = manifest_json
            .get("total_layers")
            .and_then(|v| v.as_u64())
            .ok_or("total_layers missing in texture_manifest.json")? as u32;

        let mip0 = fs::read(textures_dir.join("block_layers_mip0_16x16.rgba8"))?;
        let mip1 = fs::read(textures_dir.join("block_layers_mip1_8x8.rgba8"))?;
        let mip2 = fs::read(textures_dir.join("block_layers_mip2_4x4.rgba8"))?;
        let mip3 = fs::read(textures_dir.join("block_layers_mip3_2x2.rgba8"))?;
        let mip4 = fs::read(textures_dir.join("block_layers_mip4_1x1.rgba8"))?;

        Self::from_mip_buffers(
            device,
            queue,
            total_layers,
            [&mip0, &mip1, &mip2, &mip3, &mip4],
        )
    }

    /// Uploads raw bytes for the 5 mipmap levels [16x16, 8x8, 4x4, 2x2, 1x1] into a Texture2DArray.
    pub fn from_mip_buffers(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        total_layers: u32,
        mips: [&[u8]; 5],
    ) -> Result<Self, Box<dyn Error>> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("block_textures_2d_array"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: total_layers,
            },
            mip_level_count: 5,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (m, mip_data) in mips.iter().enumerate() {
            let m = m as u32;
            let dim = 16 >> m;
            let expected_bytes = (dim * dim * 4 * total_layers) as usize;
            if mip_data.len() != expected_bytes {
                return Err(format!(
                    "Mip level {} size mismatch: expected {} bytes ({} layers x {}x{}x4), got {}",
                    m, expected_bytes, total_layers, dim, dim, mip_data.len()
                )
                .into());
            }

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: m,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                mip_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * dim),
                    rows_per_image: Some(dim),
                },
                wgpu::Extent3d {
                    width: dim,
                    height: dim,
                    depth_or_array_layers: total_layers,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("block_textures_view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(5),
            base_array_layer: 0,
            array_layer_count: Some(total_layers),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("block_textures_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
            total_layers,
        })
    }

    /// Creates a dummy single-layer or small test TextureArray for isolated testing.
    pub fn dummy(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        total_layers: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let m0 = vec![255u8; (16 * 16 * 4 * total_layers) as usize];
        let m1 = vec![255u8; (8 * 8 * 4 * total_layers) as usize];
        let m2 = vec![255u8; (4 * 4 * 4 * total_layers) as usize];
        let m3 = vec![255u8; (2 * 2 * 4 * total_layers) as usize];
        let m4 = vec![255u8; (1 * 1 * 4 * total_layers) as usize];
        Self::from_mip_buffers(device, queue, total_layers, [&m0, &m1, &m2, &m3, &m4])
    }
}
