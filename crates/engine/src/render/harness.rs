/// Offscreen rendering target and CPU pixel readback harness for headless testing.
pub struct OffscreenTarget {
    pub width: u32,
    pub height: u32,
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub readback_buffer: wgpu::Buffer,
    pub padded_bytes_per_row: u32,
}

impl OffscreenTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_color_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_depth_target"),
            size: wgpu::Extent3d {
                width,
                height,
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

        // wgpu requires buffer copy bytes_per_row to be aligned to 256 bytes
        let unpadded_bytes = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes + align - 1) & !(align - 1);
        let buffer_size = (padded_bytes_per_row * height) as wgpu::BufferAddress;

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen_readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            width,
            height,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            readback_buffer,
            padded_bytes_per_row,
        }
    }

    /// Copies the rendered color attachment to the staging buffer, waits for completion,
    /// and returns the unpadded RGBA8 pixels as a Vec<u8> of length width * height * 4.
    pub fn readback_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_command_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().expect("buffer map_async failed");

        let data = buffer_slice.get_mapped_range().expect("get_mapped_range failed");
        let unpadded_bytes_per_row = (self.width * 4) as usize;
        let mut pixels = Vec::with_capacity(unpadded_bytes_per_row * self.height as usize);

        for row in 0..self.height as usize {
            let start = row * self.padded_bytes_per_row as usize;
            pixels.extend_from_slice(&data[start..start + unpadded_bytes_per_row]);
        }

        drop(data);
        self.readback_buffer.unmap();

        pixels
    }
}
