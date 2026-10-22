//! Asset pipeline shared color transformation and tinting utilities.

#[inline(always)]
pub fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[inline(always)]
pub fn linear_to_srgb(l: f32) -> f32 {
    let l = l.clamp(0.0, 1.0);
    if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// Applies fixed color tint in linear RGB space according to SPECS §7 / R8.
pub fn apply_linear_tint(src_16x16: &[u8], tint_hex: u32) -> Vec<u8> {
    let tr = srgb_to_linear(((tint_hex >> 16) & 0xFF) as f32 / 255.0);
    let tg = srgb_to_linear(((tint_hex >> 8) & 0xFF) as f32 / 255.0);
    let tb = srgb_to_linear((tint_hex & 0xFF) as f32 / 255.0);

    let mut out = Vec::with_capacity(src_16x16.len());
    for chunk in src_16x16.chunks_exact(4) {
        let r = srgb_to_linear(chunk[0] as f32 / 255.0);
        let g = srgb_to_linear(chunk[1] as f32 / 255.0);
        let b = srgb_to_linear(chunk[2] as f32 / 255.0);
        let a = chunk[3];

        let tinted_r = linear_to_srgb(r * tr);
        let tinted_g = linear_to_srgb(g * tg);
        let tinted_b = linear_to_srgb(b * tb);

        out.push((tinted_r * 255.0).round() as u8);
        out.push((tinted_g * 255.0).round() as u8);
        out.push((tinted_b * 255.0).round() as u8);
        out.push(a);
    }
    out
}
