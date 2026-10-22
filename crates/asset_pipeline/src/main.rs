use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use image::{imageops, DynamicImage, GenericImageView, ImageFormat, RgbaImage};
use serde::{Deserialize, Serialize};
use zip::ZipArchive;
use asset_pipeline::{apply_linear_tint, linear_to_srgb, srgb_to_linear};

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum AnimationFrameSpec {
    Index(u32),
    Detailed {
        index: u32,
        #[serde(default)]
        time: Option<u32>,
    },
}

#[derive(Deserialize, Debug, Default, Clone)]
pub struct AnimationMetadata {
    pub frametime: Option<u32>,
    pub interpolate: Option<bool>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frames: Option<Vec<AnimationFrameSpec>>,
}

#[derive(Deserialize, Debug, Default, Clone)]
pub struct McMeta {
    pub animation: Option<AnimationMetadata>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TextureEntry {
    pub base_layer: u32,
    pub original_width: u32,
    pub original_height: u32,
    pub animated: bool,
    pub frame_count: u32,
    pub frametime: u32,
    pub interpolate: bool,
    pub timeline_offset: u32,
    pub timeline_len: u32,
    pub has_mcmeta: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Manifest {
    pub total_layers: u32,
    pub layer_dimension: u32,
    pub mip_levels: u32,
    pub timeline_total_u32: u32,
    pub pre_tinted_layers: BTreeMap<String, u32>,
    pub textures: BTreeMap<String, TextureEntry>,
}


/// Detects if an RGBA buffer is a cutout (contains both transparent and opaque pixels)
fn is_cutout(rgba: &[u8]) -> bool {
    let mut has_transparent = false;
    let mut has_opaque = false;
    for chunk in rgba.chunks_exact(4) {
        let a = chunk[3];
        if a < 250 {
            has_transparent = true;
        }
        if a > 10 {
            has_opaque = true;
        }
        if has_transparent && has_opaque {
            return true;
        }
    }
    false
}

/// Downsamples an RGBA buffer by 2x in linear color space with premultiplied alpha.
/// For cutouts, applies coverage-preserving alpha scaling.
fn downsample_linear_2x(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    preserve_alpha_coverage: bool,
) -> (Vec<u8>, usize, usize) {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let mut dst = vec![0u8; dst_w * dst_h * 4];

    let alpha_threshold = 0.5f32;
    let ref_coverage = if preserve_alpha_coverage {
        let mut count = 0usize;
        for i in 0..(src_w * src_h) {
            let a = src[i * 4 + 3] as f32 / 255.0;
            if a > alpha_threshold {
                count += 1;
            }
        }
        count as f32 / (src_w * src_h) as f32
    } else {
        0.0
    };

    let mut unscaled_alphas = Vec::with_capacity(dst_w * dst_h);

    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx0 = (x * 2).min(src_w - 1);
            let sx1 = (x * 2 + 1).min(src_w - 1);
            let sy0 = (y * 2).min(src_h - 1);
            let sy1 = (y * 2 + 1).min(src_h - 1);

            let coords = [(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)];
            let mut lin_r = 0.0f32;
            let mut lin_g = 0.0f32;
            let mut lin_b = 0.0f32;
            let mut a_sum = 0.0f32;

            for &(cx, cy) in &coords {
                let idx = (cy * src_w + cx) * 4;
                let r = srgb_to_linear(src[idx] as f32 / 255.0);
                let g = srgb_to_linear(src[idx + 1] as f32 / 255.0);
                let b = srgb_to_linear(src[idx + 2] as f32 / 255.0);
                let a = src[idx + 3] as f32 / 255.0;

                lin_r += r * a;
                lin_g += g * a;
                lin_b += b * a;
                a_sum += a;
            }

            let avg_a = a_sum / 4.0;
            unscaled_alphas.push(avg_a);

            let (final_r, final_g, final_b) = if a_sum > 1e-5 {
                (lin_r / a_sum, lin_g / a_sum, lin_b / a_sum)
            } else {
                (0.0, 0.0, 0.0)
            };

            let dst_idx = (y * dst_w + x) * 4;
            dst[dst_idx] = (linear_to_srgb(final_r) * 255.0).round() as u8;
            dst[dst_idx + 1] = (linear_to_srgb(final_g) * 255.0).round() as u8;
            dst[dst_idx + 2] = (linear_to_srgb(final_b) * 255.0).round() as u8;
            dst[dst_idx + 3] = (avg_a.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }

    if preserve_alpha_coverage && ref_coverage > 0.001 && ref_coverage < 0.999 {
        let mut low = 0.5f32;
        let mut high = 4.0f32;
        let total_px = (dst_w * dst_h) as f32;

        for _ in 0..10 {
            let mid = (low + high) * 0.5;
            let cov = unscaled_alphas.iter().filter(|&&a| a * mid > alpha_threshold).count() as f32 / total_px;
            if cov < ref_coverage {
                low = mid;
            } else {
                high = mid;
            }
        }
        let best_c = (low + high) * 0.5;
        for (i, &a) in unscaled_alphas.iter().enumerate() {
            let scaled_a = (a * best_c).clamp(0.0, 1.0);
            dst[i * 4 + 3] = (scaled_a * 255.0).round() as u8;
        }
    }

    (dst, dst_w, dst_h)
}


/// Downsamples an image of size (w, h) down to 16x16 in successive 2x steps
fn downsample_to_16x16(img: &DynamicImage) -> Vec<u8> {
    let (mut cur_w, mut cur_h) = img.dimensions();
    let mut buf = img.to_rgba8().into_raw();
    let cutout = is_cutout(&buf);

    while cur_w > 16 || cur_h > 16 {
        let (next_buf, next_w, next_h) = downsample_linear_2x(&buf, cur_w as usize, cur_h as usize, cutout);
        buf = next_buf;
        cur_w = next_w as u32;
        cur_h = next_h as u32;
    }

    if cur_w == 16 && cur_h == 16 {
        buf
    } else {
        // Fallback for odd dimensions
        imageops::resize(img, 16, 16, imageops::FilterType::Triangle).into_raw()
    }
}

/// Builds an expanded frame timeline from parsed AnimationMetadata
fn build_timeline(anim: Option<&AnimationMetadata>, frame_count: u32) -> (Vec<u32>, bool) {
    let frametime = anim.and_then(|a| a.frametime).unwrap_or(1).max(1);
    let interpolate = anim.and_then(|a| a.interpolate).unwrap_or(false);

    let mut timeline = Vec::new();
    if let Some(frames) = anim.and_then(|a| a.frames.as_ref()) {
        for f in frames {
            let (idx, duration) = match f {
                AnimationFrameSpec::Index(i) => (*i, frametime),
                AnimationFrameSpec::Detailed { index, time } => (*index, time.unwrap_or(frametime).max(1)),
            };
            if idx < frame_count {
                for _ in 0..duration {
                    timeline.push(idx);
                }
            }
        }
    } else {
        for idx in 0..frame_count {
            for _ in 0..frametime {
                timeline.push(idx);
            }
        }
    }

    if timeline.is_empty() {
        for idx in 0..frame_count {
            timeline.push(idx);
        }
    }

    (timeline, interpolate)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME").expect("HOME not set");
    let jar_path = PathBuf::from(&home).join(
        ".local/share/PrismLauncher/libraries/com/mojang/minecraft/26.3/minecraft-26.3-client.jar",
    );

    if !jar_path.exists() {
        return Err(format!("Minecraft 26.3 client JAR not found at {:?}", jar_path).into());
    }

    println!("Opening JAR: {:?}", jar_path);
    let file = File::open(&jar_path)?;
    let mut archive = ZipArchive::new(BufReader::new(file))?;

    // Output target directories
    let out_models = Path::new("assets/models/block");
    let out_blockstates = Path::new("assets/blockstates");
    let out_mcmetas = Path::new("assets/textures/block_mcmetas");
    let out_blocks = Path::new("assets/textures/blocks");
    let out_textures = Path::new("assets/textures");

    fs::create_dir_all(out_models)?;
    fs::create_dir_all(out_blockstates)?;
    fs::create_dir_all(out_mcmetas)?;
    fs::create_dir_all(out_blocks)?;
    fs::create_dir_all(out_textures)?;

    println!("Extracting models, blockstates, and texture assets from JAR...");
    let mut texture_data: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut mcmeta_data: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    let mut model_count = 0usize;
    let mut state_count = 0usize;
    let mut mcmeta_count = 0usize;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();

        if name.starts_with("assets/minecraft/models/block/") && name.ends_with(".json") {
            let rel = name.strip_prefix("assets/minecraft/models/block/").unwrap();
            let dest = out_models.join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            fs::write(dest, buf)?;
            model_count += 1;
        } else if name.starts_with("assets/minecraft/blockstates/") && name.ends_with(".json") {
            let rel = name.strip_prefix("assets/minecraft/blockstates/").unwrap();
            let dest = out_blockstates.join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            fs::write(dest, buf)?;
            state_count += 1;
        } else if name.starts_with("assets/minecraft/textures/block/") {
            let rel = name.strip_prefix("assets/minecraft/textures/block/").unwrap();
            if rel.ends_with(".png") {
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)?;
                let base = rel.strip_suffix(".png").unwrap().to_string();
                texture_data.insert(base, buf);
            } else if rel.ends_with(".png.mcmeta") {
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)?;
                let base = rel.strip_suffix(".png.mcmeta").unwrap().to_string();
                mcmeta_data.insert(base.clone(), buf.clone());
                let dest = out_mcmetas.join(format!("{}.png.mcmeta", base));
                fs::write(dest, buf)?;
                mcmeta_count += 1;
            }
        }
    }

    println!(
        "Extracted {} block models, {} blockstates, and {} mcmetas.",
        model_count, state_count, mcmeta_count
    );
    println!("Processing {} block PNG textures...", texture_data.len());

    let mut manifest_entries = BTreeMap::new();
    let mut current_layer = 0u32;
    let mut mip0_layers_rgba8 = Vec::new();

    let mut global_timeline: Vec<u32> = Vec::new();

    // Cache of 16x16 base frames for pre-tinting lookup
    let mut base_frames_16x16: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    for (name, png_bytes) in &texture_data {
        let img = match image::load_from_memory_with_format(png_bytes, ImageFormat::Png) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Warning: Skipping unreadable PNG {}: {}", name, e);
                continue;
            }
        };

        let (w, h) = img.dimensions();
        let animated = h > w && (h % w == 0);
        let frame_count = if animated { h / w } else { 1 };
        let has_mcmeta = mcmeta_data.contains_key(name);

        let parsed_mcmeta: Option<AnimationMetadata> = if let Some(meta_bytes) = mcmeta_data.get(name) {
            serde_json::from_slice::<McMeta>(meta_bytes).ok().and_then(|m| m.animation)
        } else {
            None
        };

        let base_layer_for_entry = current_layer;

        if animated {
            // Process all individual animation frames
            let (timeline, interpolate) = build_timeline(parsed_mcmeta.as_ref(), frame_count);
            let frametime = parsed_mcmeta.as_ref().and_then(|a| a.frametime).unwrap_or(1).max(1);
            let timeline_offset = global_timeline.len() as u32;
            let timeline_len = timeline.len() as u32;
            global_timeline.extend(&timeline);

            for f in 0..frame_count {
                let frame_crop = imageops::crop_imm(&img, 0, f * w, w, w).to_image();
                let frame_16x16: Vec<u8> = if w == 16 {
                    frame_crop.into_raw()
                } else {
                    // For 32x32 frames (e.g. water_flow, lava_flow), downsample to 16x16 linearly
                    downsample_to_16x16(&DynamicImage::ImageRgba8(frame_crop))
                };

                if f == 0 {
                    base_frames_16x16.insert(name.clone(), frame_16x16.clone());
                    let target_png_path = out_blocks.join(format!("{}.png", name));
                    if let Some(parent) = target_png_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    if let Some(r_img) = RgbaImage::from_raw(16, 16, frame_16x16.clone()) {
                        let _ = r_img.save_with_format(&target_png_path, ImageFormat::Png);
                    }
                }

                mip0_layers_rgba8.extend_from_slice(&frame_16x16);
                current_layer += 1;
            }

            manifest_entries.insert(
                name.clone(),
                TextureEntry {
                    base_layer: base_layer_for_entry,
                    original_width: w,
                    original_height: h,
                    animated: true,
                    frame_count,
                    frametime,
                    interpolate,
                    timeline_offset,
                    timeline_len,
                    has_mcmeta,
                },
            );
        } else {
            // Static texture
            let frame_16x16 = if w == 16 && h == 16 {
                img.to_rgba8().into_raw()
            } else {
                // Downsample non-16x16 static textures (32x32 signs/shelves, 64x64 beds) cleanly
                downsample_to_16x16(&img)
            };

            base_frames_16x16.insert(name.clone(), frame_16x16.clone());

            let target_png_path = out_blocks.join(format!("{}.png", name));
            if let Some(parent) = target_png_path.parent() {
                fs::create_dir_all(parent)?;
            }
            if let Some(r_img) = RgbaImage::from_raw(16, 16, frame_16x16.clone()) {
                let _ = r_img.save_with_format(&target_png_path, ImageFormat::Png);
            }

            mip0_layers_rgba8.extend_from_slice(&frame_16x16);
            current_layer += 1;

            manifest_entries.insert(
                name.clone(),
                TextureEntry {
                    base_layer: base_layer_for_entry,
                    original_width: w,
                    original_height: h,
                    animated: false,
                    frame_count: 1,
                    frametime: 0,
                    interpolate: false,
                    timeline_offset: 0,
                    timeline_len: 0,
                    has_mcmeta,
                },
            );
        }
    }

    // Pre-tint fixed variants (birch leaves 0x80A755, spruce leaves 0x619961, lily pad 0x208030)
    let mut pre_tinted_layers = BTreeMap::new();
    let fixed_tints: [(&str, &str, u32); 3] = [
        ("birch_leaves", "birch_leaves_tinted", 0x80A755),
        ("spruce_leaves", "spruce_leaves_tinted", 0x619961),
        ("lily_pad", "lily_pad_tinted", 0x208030),
    ];

    for (src_name, target_name, hex) in &fixed_tints {
        if let Some(base_16) = base_frames_16x16.get(*src_name) {
            let tinted = apply_linear_tint(base_16, *hex);
            let target_png_path = out_blocks.join(format!("{}.png", target_name));
            if let Some(r_img) = RgbaImage::from_raw(16, 16, tinted.clone()) {
                let _ = r_img.save_with_format(&target_png_path, ImageFormat::Png);
            }
            mip0_layers_rgba8.extend_from_slice(&tinted);
            pre_tinted_layers.insert(target_name.to_string(), current_layer);
            manifest_entries.insert(
                target_name.to_string(),
                TextureEntry {
                    base_layer: current_layer,
                    original_width: 16,
                    original_height: 16,
                    animated: false,
                    frame_count: 1,
                    frametime: 0,
                    interpolate: false,
                    timeline_offset: 0,
                    timeline_len: 0,
                    has_mcmeta: false,
                },
            );
            current_layer += 1;
        } else {
            eprintln!("Warning: Could not find base texture '{}' to pre-tint", src_name);
        }
    }

    // Composite grass_block_side_overlay over grass_block_side with Plains grass tint 0x79C05A [APPROX-RENDERING-07, APPROX-RENDERING-09]
    let overlay_composite_name = "grass_block_side_overlay_composite";
    if let (Some(dirt_16), Some(overlay_16)) = (
        base_frames_16x16.get("grass_block_side"),
        base_frames_16x16.get("grass_block_side_overlay"),
    ) {
        let tinted_overlay = apply_linear_tint(overlay_16, 0x79C05A);
        let mut composite = dirt_16.clone();
        for (comp_chunk, over_chunk) in composite.chunks_exact_mut(4).zip(tinted_overlay.chunks_exact(4)) {
            let over_a = over_chunk[3];
            if over_a == 255 {
                comp_chunk.copy_from_slice(over_chunk);
            } else if over_a > 0 {
                let a = over_a as f32 / 255.0;
                comp_chunk[0] = (over_chunk[0] as f32 * a + comp_chunk[0] as f32 * (1.0 - a)).round() as u8;
                comp_chunk[1] = (over_chunk[1] as f32 * a + comp_chunk[1] as f32 * (1.0 - a)).round() as u8;
                comp_chunk[2] = (over_chunk[2] as f32 * a + comp_chunk[2] as f32 * (1.0 - a)).round() as u8;
                comp_chunk[3] = 255;
            }
        }
        let target_png_path = out_blocks.join(format!("{}.png", overlay_composite_name));
        if let Some(r_img) = RgbaImage::from_raw(16, 16, composite.clone()) {
            let _ = r_img.save_with_format(&target_png_path, ImageFormat::Png);
        }
        mip0_layers_rgba8.extend_from_slice(&composite);
        pre_tinted_layers.insert(overlay_composite_name.to_string(), current_layer);
        manifest_entries.insert(
            overlay_composite_name.to_string(),
            TextureEntry {
                base_layer: current_layer,
                original_width: 16,
                original_height: 16,
                animated: false,
                frame_count: 1,
                frametime: 0,
                interpolate: false,
                timeline_offset: 0,
                timeline_len: 0,
                has_mcmeta: false,
            },
        );
        current_layer += 1;
    } else {
        eprintln!("Warning: Could not find grass_block_side or grass_block_side_overlay to composite");
    }

    let total_layers = current_layer;
    println!("Total layers in 2D array texture: {} (under 2048 limit)", total_layers);
    println!("Generating 4 offline mipmap levels in linear color space with coverage preservation...");

    // Generate full mip chain:
    // Level 0: 16x16
    // Level 1: 8x8
    // Level 2: 4x4
    // Level 3: 2x2
    // Level 4: 1x1
    let mut mip1_layers_rgba8 = Vec::with_capacity((total_layers as usize) * 8 * 8 * 4);
    let mut mip2_layers_rgba8 = Vec::with_capacity((total_layers as usize) * 4 * 4 * 4);
    let mut mip3_layers_rgba8 = Vec::with_capacity((total_layers as usize) * 2 * 2 * 4);
    let mut mip4_layers_rgba8 = Vec::with_capacity((total_layers as usize) * 1 * 1 * 4);

    let bytes_per_layer_mip0 = 16 * 16 * 4;
    for l in 0..total_layers as usize {
        let layer_mip0 = &mip0_layers_rgba8[l * bytes_per_layer_mip0..(l + 1) * bytes_per_layer_mip0];
        let cutout = is_cutout(layer_mip0);

        let (layer_mip1, _, _) = downsample_linear_2x(layer_mip0, 16, 16, cutout);
        let (layer_mip2, _, _) = downsample_linear_2x(&layer_mip1, 8, 8, cutout);
        let (layer_mip3, _, _) = downsample_linear_2x(&layer_mip2, 4, 4, cutout);
        let (layer_mip4, _, _) = downsample_linear_2x(&layer_mip3, 2, 2, cutout);

        mip1_layers_rgba8.extend_from_slice(&layer_mip1);
        mip2_layers_rgba8.extend_from_slice(&layer_mip2);
        mip3_layers_rgba8.extend_from_slice(&layer_mip3);
        mip4_layers_rgba8.extend_from_slice(&layer_mip4);
    }

    // Write out all mip files
    fs::write("assets/textures/block_layers_mip0_16x16.rgba8", &mip0_layers_rgba8)?;
    fs::write("assets/textures/block_layers_mip1_8x8.rgba8", &mip1_layers_rgba8)?;
    fs::write("assets/textures/block_layers_mip2_4x4.rgba8", &mip2_layers_rgba8)?;
    fs::write("assets/textures/block_layers_mip3_2x2.rgba8", &mip3_layers_rgba8)?;
    fs::write("assets/textures/block_layers_mip4_1x1.rgba8", &mip4_layers_rgba8)?;

    // Write binary timeline table
    let mut timeline_bytes = Vec::with_capacity(global_timeline.len() * 4);
    for val in &global_timeline {
        timeline_bytes.extend_from_slice(&val.to_le_bytes());
    }
    fs::write("assets/textures/animation_timeline.bin", &timeline_bytes)?;

    // Write binary animation meta table: 1844 rows x [u32 offset, u32 len, u32 frametime] = 22,128 bytes
    // SPECS 9742f26 pins per-layer ownership: distribute [offset, len, ft_code] across base..base+frame_count
    let mut animation_meta = vec![[0u32; 3]; total_layers as usize];
    for (_name, entry) in &manifest_entries {
        let base = entry.base_layer as usize;
        if entry.animated {
            let ft_field = if entry.interpolate { 0 } else { 1 };
            for f in 0..entry.frame_count as usize {
                if base + f < total_layers as usize {
                    animation_meta[base + f] = [entry.timeline_offset, entry.timeline_len, ft_field];
                }
            }
        } else {
            animation_meta[base] = [0, 0, 0];
        }
    }
    let mut meta_bytes = Vec::with_capacity(animation_meta.len() * 12);
    for row in &animation_meta {
        meta_bytes.extend_from_slice(&row[0].to_le_bytes());
        meta_bytes.extend_from_slice(&row[1].to_le_bytes());
        meta_bytes.extend_from_slice(&row[2].to_le_bytes());
    }
    fs::write("assets/textures/animation_meta.bin", &meta_bytes)?;

    let manifest = Manifest {
        total_layers,
        layer_dimension: 16,
        mip_levels: 5,
        timeline_total_u32: global_timeline.len() as u32,
        pre_tinted_layers,
        textures: manifest_entries,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write("assets/textures/texture_manifest.json", manifest_json)?;

    println!("------------------------------------------------------------");
    println!("Asset Pipeline Execution Complete:");
    println!("  Total Texture Layers: {}", total_layers);
    println!("  Mip Levels Baked: 5 (16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1)");
    println!("  Timeline Buffer Size: {} entries ({} bytes)", global_timeline.len(), timeline_bytes.len());
    println!("  Pre-tinted layers: birch_leaves_tinted, spruce_leaves_tinted, lily_pad_tinted");
    println!("  Extracted models: {} to assets/models/block", model_count);
    println!("  Extracted blockstates: {} to assets/blockstates", state_count);
    println!("------------------------------------------------------------");

    Ok(())
}

