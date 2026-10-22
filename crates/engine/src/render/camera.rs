#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new(matrix: glam::Mat4) -> Self {
        Self {
            view_proj: matrix.to_cols_array_2d(),
        }
    }
}

pub struct Camera {
    pub eye: glam::Vec3,
    pub target: glam::Vec3,
    pub up: glam::Vec3,
    pub aspect: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(eye: glam::Vec3, target: glam::Vec3, aspect: f32) -> Self {
        Self {
            eye,
            target,
            up: glam::Vec3::Y,
            aspect,
            fov_y: 70.0f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
        }
    }

    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = glam::Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far);
        proj * view
    }

    pub fn build_uniform(&self) -> CameraUniform {
        CameraUniform::new(self.build_view_projection_matrix())
    }
}
