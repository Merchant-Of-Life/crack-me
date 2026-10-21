use bytemuck::{Pod, Zeroable};

/// Exactly 64-bit packed voxel vertex according to SPECS.md
///
/// Word 0:
/// Bits 0..4 (5b): Local X (0..=16)
/// Bits 5..9 (5b): Local Y (0..=16)
/// Bits 10..14 (5b): Local Z (0..=16)
/// Bits 15..17 (3b): Normal ID (0..5: +Y, -Y, +Z, -Z, +X, -X)
/// Bits 18..19 (2b): Tint mode (0..3: None, Grass, Foliage, Water)
/// Bits 20..31 (12b): Spare / UV rotation / flags
///
/// Word 1:
/// Bits 0..3 (4b): Block Light (0..15)
/// Bits 4..7 (4b): Sky Light (0..15)
/// Bits 8..9 (2b): Ambient Occlusion (0..3)
/// Bits 10..20 (11b): Base Texture Layer (0..2047)
/// Bits 21..31 (11b): Spare
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct PackedVoxelVertex {
    pub word0: u32,
    pub word1: u32,
}

impl PackedVoxelVertex {
    #[inline(always)]
    pub fn new(
        x: u32,
        y: u32,
        z: u32,
        normal_id: u32,
        tint_mode: u32,
        flags: u32,
        block_light: u32,
        sky_light: u32,
        ao: u32,
        base_layer: u32,
    ) -> Self {
        debug_assert!(x <= 16, "X coordinate must be <= 16");
        debug_assert!(y <= 16, "Y coordinate must be <= 16");
        debug_assert!(z <= 16, "Z coordinate must be <= 16");
        debug_assert!(normal_id < 6, "Normal ID must be in 0..5");
        debug_assert!(tint_mode < 4, "Tint mode must be in 0..3");
        debug_assert!(flags < 4096, "Flags must be < 4096");
        debug_assert!(block_light < 16, "Block light must be < 16");
        debug_assert!(sky_light < 16, "Sky light must be < 16");
        debug_assert!(ao < 4, "AO must be in 0..3");
        debug_assert!(base_layer < 2048, "Base layer must be < 2048");

        let word0 = (x & 0x1F)
            | ((y & 0x1F) << 5)
            | ((z & 0x1F) << 10)
            | ((normal_id & 0x07) << 15)
            | ((tint_mode & 0x03) << 18)
            | ((flags & 0x0FFF) << 20);

        let word1 = (block_light & 0x0F)
            | ((sky_light & 0x0F) << 4)
            | ((ao & 0x03) << 8)
            | ((base_layer & 0x07FF) << 10);

        Self { word0, word1 }
    }

    #[inline(always)]
    pub fn x(&self) -> u32 {
        self.word0 & 0x1F
    }

    #[inline(always)]
    pub fn y(&self) -> u32 {
        (self.word0 >> 5) & 0x1F
    }

    #[inline(always)]
    pub fn z(&self) -> u32 {
        (self.word0 >> 10) & 0x1F
    }

    #[inline(always)]
    pub fn normal_id(&self) -> u32 {
        (self.word0 >> 15) & 0x07
    }

    #[inline(always)]
    pub fn tint_mode(&self) -> u32 {
        (self.word0 >> 18) & 0x03
    }

    #[inline(always)]
    pub fn flags(&self) -> u32 {
        (self.word0 >> 20) & 0x0FFF
    }

    #[inline(always)]
    pub fn block_light(&self) -> u32 {
        self.word1 & 0x0F
    }

    #[inline(always)]
    pub fn sky_light(&self) -> u32 {
        (self.word1 >> 4) & 0x0F
    }

    #[inline(always)]
    pub fn ao(&self) -> u32 {
        (self.word1 >> 8) & 0x03
    }

    #[inline(always)]
    pub fn base_layer(&self) -> u32 {
        (self.word1 >> 10) & 0x07FF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_packing_roundtrip_64bit() {
        assert_eq!(std::mem::size_of::<PackedVoxelVertex>(), 8);
        assert_eq!(std::mem::align_of::<PackedVoxelVertex>(), 4);

        let v = PackedVoxelVertex::new(16, 8, 16, 4, 2, 512, 14, 15, 3, 1842);
        assert_eq!(v.x(), 16);
        assert_eq!(v.y(), 8);
        assert_eq!(v.z(), 16);
        assert_eq!(v.normal_id(), 4);
        assert_eq!(v.tint_mode(), 2);
        assert_eq!(v.flags(), 512);
        assert_eq!(v.block_light(), 14);
        assert_eq!(v.sky_light(), 15);
        assert_eq!(v.ao(), 3);
        assert_eq!(v.base_layer(), 1842);
    }
}

