//! Clean-room noise synthesis layer (SPECS.md §2 pin `Terrain Generation & Synthesis`).
//!
//! Faithful f64 port of decompiled Minecraft 26.3 classes (Vineflower decompile,
//! spec-side reference only):
//! - `net.minecraft.world.level.levelgen.synth.SimplexNoise` (169 lines)
//! - `net.minecraft.world.level.levelgen.synth.GradientNoise` (86 lines) — the
//!   `SimplexNoise` super class holding the permutation table, offsets and gradients
//! - `net.minecraft.world.level.levelgen.synth.NormalNoise` (377 lines)
//!
//! Standing SPECS.md §2 pins replicated here:
//! - RANGE output range: symmetric ±2.0 (`SimplexNoise.java:8`, `Interval.ofSymmetric(2.0F)`)
//! - STANDARD_DEVIATION normalization: `0.42544` (`SimplexNoise.java:9`)
//! - `F2 = (sqrt(3) - 1) / 2`, `G2 = (3 - sqrt(3)) / 6` (`SimplexNoise.java:10-12`)
//!
//! Provenance caveat (open verification point): the decompile quarantine contains
//! the `RandomSource` *interface* only — no `LegacyRandomSource` / `WorldgenRandom`
//! implementation class was found under `/~/.minecraft-decomp/src`. RNG consumption
//! is therefore served by [`JavaRandomCompat`], which replicates the documented
//! `java.util.Random` algorithm (48-bit masked LCG), flagged for reviewer
//! verification against the actual runtime RNG used by 26.3 worldgen.
//!
//! Omissions from `NormalNoise.java` (unused-by-this-port branches, listed as
//! required by the task): the fluent `Builder` (behavioral duplicate of
//! [`NormalNoiseParameters`]), serialization `Codec`s and parameter validation
//! (datafixer/codec plumbing), `hashCode`/`equals`/`toString`,
//! `parityConfigString`, `createForLegacyNetherBiome` + `LegacyFbmInitializer`
//! (deprecated legacy nether-biome path), and the class constants `SAMPLE_COUNT`,
//! `MAX_OCTAVE_COUNT`, `MAX_AMPLITUDE`, `PERSISTENCE`, `LACUNARITY` (declared but
//! never read in the decompiled fragment). The 2D `NoiseStack.get(x, y)` overload
//! is likewise omitted — [`NormalNoise::get_value`] covers the 3D evaluation path.

// ---------------------------------------------------------------------------
// RNG compatibility layer (see module docs: provenance flagged)
// ---------------------------------------------------------------------------

// --- Core Constants (java.util.Random documented algorithm) ---

/// LCG multiplier `0x5DEECE66D` (java.util.Random.setSeed/next).
const MULT: u64 = 0x5DEECE66D;

/// LCG addend `0xB` (java.util.Random.next).
const ADDEND: u64 = 0xB;

/// 48-bit state mask `(1L << 48) - 1` (java.util.Random.setSeed/next).
const MASK: u64 = (1u64 << 48) - 1;

/// Replication of the documented `java.util.Random` algorithm, providing the
/// minimal surface `GradientNoise`'s constructor consumes
/// (`GradientNoise.java:36-50`: three `nextDouble()` calls, then 256 ×
/// `nextInt(256 - i)`).
///
/// Provenance: **documented java.util.Random algorithm; flagged for reviewer
/// verification** — the decompile quarantine ships no `RandomSource`
/// implementation class (see module docs).
pub struct JavaRandomCompat {
    seed: u64,
}

impl JavaRandomCompat {
    /// Mirrors `new Random(seed)` → `setSeed`:
    /// `this.seed = (seed ^ 0x5DEECE66DL) & ((1L << 48) - 1)` (java.util.Random docs).
    pub fn new(seed: i64) -> Self {
        Self {
            seed: ((seed as u64) ^ MULT) & MASK,
        }
    }

    /// Mirrors `protected int next(int bits)`:
    /// seed update `seed = (seed * 0x5DEECE66D + 0xB) & ((1L << 48) - 1)`, then
    /// `return (int)(seed >>> (48 - bits))` — `>>>` is a *logical* shift; the u64
    /// state is stored unsigned so `>>` here is already logical, and the
    /// `as i32` cast replicates Java's narrowing `(int)` truncation (relevant
    /// when `bits == 32`, where the top bit yields a negative value).
    pub fn next_bits(&mut self, bits: u32) -> i32 {
        debug_assert!(
            bits > 0 && bits <= 32,
            "java.util.Random.next requires 1 <= bits <= 32"
        );
        self.seed = (self.seed.wrapping_mul(MULT).wrapping_add(ADDEND)) & MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// Mirrors `nextDouble()`: `(((long)next(26) << 27) + next(27)) * DOUBLE_UNIT`
    /// where `DOUBLE_UNIT = 0x1.0p-53 = 2^-53`. Call order (next(26) then
    /// next(27)) is load-bearing for parity — do not reorder.
    pub fn next_double(&mut self) -> f64 {
        let top = self.next_bits(26) as i64;
        let bot = self.next_bits(27) as i64;
        ((top << 27) + bot) as f64 * 1.1102230246251565e-16 // DOUBLE_UNIT == 2^-53 exactly
    }

    /// Mirrors `nextInt(bound)` exactly, including the rejection-sampling loop
    /// (bound not a power of two) and the power-of-two shortcut
    /// `(int)((bound * (long)next(31)) >> 31)` — the i64 multiply cannot
    /// overflow for `bound <= 2^30` and the shift is logical on a
    /// non-negative value.
    pub fn next_int(&mut self, bound: i32) -> i32 {
        debug_assert!(bound > 0, "java.util.Random.nextInt requires bound > 0");
        if (bound & -bound) == bound {
            // bound is a power of two — java.util.Random.nextInt shortcut path.
            return (((bound as i64) * (self.next_bits(31) as i64)) >> 31) as i32;
        }
        loop {
            let bits = self.next_bits(31);
            let val = bits % bound;
            if ((bits as i64) - (val as i64) + ((bound - 1) as i64)) < (1i64 << 31) {
                return val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimplexNoise / GradientNoise
// ---------------------------------------------------------------------------

/// Symmetric output interval `[-2.0, 2.0]` (`SimplexNoise.java:8`,
/// `Interval.ofSymmetric(2.0F)`; SPECS.md §2 pin). Kept private per task spec —
/// exercised by the range-envelope sanity tests below. `Interval.java` itself is
/// not present in the decompile quarantine, so only the symmetric magnitude is
/// carried.
///
/// Unused-in-engine constant; retained for decompile parity.
#[allow(dead_code)]
const RANGE: f64 = 2.0;

/// Standard deviation constant (`SimplexNoise.java:9`, SPECS.md §2 pin).
/// Unused by `SimplexNoise` itself in the decompile (declared for downstream
/// users); retained for decompile parity.
#[allow(dead_code)]
const STANDARD_DEVIATION: f64 = 0.42544;

/// `SQRT_3 = Math.sqrt(3.0)` (`SimplexNoise.java:10`).
const SQRT_3: f64 = 1.7320508075688772; // f64: correctly-rounded sqrt(3.0)

/// `F2 = 0.5 * (SQRT_3 - 1.0)` (`SimplexNoise.java:11`, SPECS.md §2 pin
/// F2 = (sqrt(3)-1)/2 = 0.3660254037844386).
const F2: f64 = 0.5 * (SQRT_3 - 1.0);

/// `G2 = (3.0 - SQRT_3) / 6.0` (`SimplexNoise.java:12`, SPECS.md §2 pin
/// G2 = (3-sqrt(3))/6 = 0.21132486540518713).
const G2: f64 = (3.0 - SQRT_3) / 6.0;

/// 3D skew factor literal used verbatim by the decompile (`SimplexNoise.java:82,83`).
const F3: f64 = 0.3333333333333333;

/// 3D unskew factor literal used verbatim by the decompile (`SimplexNoise.java:87,88,147-155`).
const G3: f64 = 0.16666666666666666;

/// The 16 gradient vectors (`GradientNoise.java:7-24`).
const GRADIENT: [(i32, i32, i32); 16] = [
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
    (1, 1, 0),
    (0, -1, 1),
    (-1, 1, 0),
    (0, -1, -1),
];

/// `Gradient.dot(x, y, z)` (`GradientNoise.java:74-76`).
#[inline]
fn gradient_dot(g: (i32, i32, i32), x: f64, y: f64, z: f64) -> f64 {
    (g.0 as f64) * x + (g.1 as f64) * y + (g.2 as f64) * z
}

/// Port of `Mth#floor(double)` as consumed by `SimplexNoise.get`
/// (`SimplexNoise.java:45,46,84-86`): floors to an integral double, then narrows
/// to Java `int`. Java's double-to-int cast *clamps* at `Integer.MAX_VALUE` /
/// *MIN_VALUE*, NaN → 0 — exactly Rust's saturating `as i32` on an integer f64.
/// Negative operands receive true floor semantics (floor(-1.2) = -2), not
/// truncation.
#[inline]
fn mth_floor(value: f64) -> i32 {
    value.floor() as i32
}

/// Simplex noise with the MC 26.3 `GradientNoise` layout: 256-entry permutation
/// byte table plus owner-independent sample offsets drawn at construction
/// (`GradientNoise.java:27-30`).
pub struct SimplexNoise {
    /// `protected final byte[] perms = new byte[256]` (`GradientNoise.java:27`).
    perms: [u8; 256],
    /// `protected final double offsetX/Y/Z` (`GradientNoise.java:28-30`).
    offset_x: f64,
    offset_y: f64,
    offset_z: f64,
}

impl SimplexNoise {
    /// `SimplexNoise(RandomSource random)` ⇒ `GradientNoise(random, 256.0)`
    /// (`SimplexNoise.java:14-16`, `GradientNoise.java:32-34`).
    pub fn new(random: &mut JavaRandomCompat) -> Self {
        Self::from_gradient_noise_ctor(random, 256.0)
    }

    /// `SimplexNoise(RandomSource random, boolean discardNoiseOffset)` with
    /// `discardNoiseOffset == true` ⇒ `GradientNoise(random, 0.0)`
    /// (`SimplexNoise.java:18-20`). NOTE the decompile subtlety: the offsets
    /// are still *drawn* (`offset = random.nextDouble() * 0.0` —
    /// `GradientNoise.java:37-39`), so this constructor advances the RNG by
    /// exactly the same three `nextDouble()` calls as [`SimplexNoise::new`];
    /// only the resulting offsets become exact `0.0`.
    pub fn with_offset_handling(random: &mut JavaRandomCompat) -> Self {
        Self::from_gradient_noise_ctor(random, 0.0)
    }

    /// `protected GradientNoise(RandomSource random, double noiseOffsetScale)`
    /// (`GradientNoise.java:36-51`). Operation order: three `nextDouble()`
    /// draws, identity fill, then the in-place shuffle consuming
    /// `nextInt(256 - i)` for `i` in `0..256`.
    fn from_gradient_noise_ctor(random: &mut JavaRandomCompat, noise_offset_scale: f64) -> Self {
        let offset_x = random.next_double() * noise_offset_scale;
        let offset_y = random.next_double() * noise_offset_scale;
        let offset_z = random.next_double() * noise_offset_scale;

        let mut perms = [0u8; 256];
        for i in 0..256usize {
            perms[i] = i as u8;
        }

        for i in 0..256usize {
            let offset = random.next_int((256 - i) as i32) as usize;
            let tmp = perms[i];
            perms[i] = perms[offset + i];
            perms[offset + i] = tmp;
        }

        Self {
            perms,
            offset_x,
            offset_y,
            offset_z,
        }
    }

    /// `protected int permute(int x)` (`GradientNoise.java:53-55`):
    /// `return this.perms[x & 0xFF] & 0xFF;`
    #[inline]
    fn permute(&self, x: i32) -> usize {
        self.perms[(x & 0xFF) as usize] as usize
    }

    /// `private double getCornerNoise3D(...)` (`SimplexNoise.java:27-38`).
    #[inline]
    fn get_corner_noise_3d(&self, index: usize, x: f64, y: f64, z: f64, base: f64) -> f64 {
        // double t0 = base - x * x - y * y - z * z;
        let mut t0 = base - x * x - y * y - z * z;
        // if (t0 < 0.0) { n0 = 0.0; } else { t0 *= t0; n0 = t0 * t0 * GRADIENT[index].dot(x, y, z); }
        if t0 < 0.0 {
            0.0
        } else {
            t0 *= t0;
            t0 * t0 * gradient_dot(GRADIENT[index], x, y, z)
        }
    }

    /// `public float get(double _xin, double _yin)` (`SimplexNoise.java:40-75`).
    ///
    /// The 2D path routes through the shared 3D corner-noise helper with `z`
    /// pinned to `0.0` and corner decay factor `base = 0.5`, returning
    /// `70.0 * (n0 + n1 + n2)`. The Java method's return type is `float`, so the
    /// f64 expression is narrowed to f32 and widened again — the widened value
    /// is bit-identical to what every Java caller observes.
    pub fn get_value_2d(&self, _xin: f64, _yin: f64) -> f64 {
        let xin = _xin + self.offset_x;
        let yin = _yin + self.offset_y;
        // double s = (xin + yin) * F2;
        let s = (xin + yin) * F2;
        let i = mth_floor(xin + s);
        let j = mth_floor(yin + s);
        // double t = (i + j) * G2;
        let t = ((i as f64) + (j as f64)) * G2;
        let x0 = xin - ((i as f64) - t);
        let y0 = yin - ((j as f64) - t);
        // if (x0 > y0) { i1 = 1; j1 = 0; } else { i1 = 0; j1 = 1; }
        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };
        let x1 = x0 - (i1 as f64) + G2;
        let y1 = y0 - (j1 as f64) + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;
        let ii = i & 0xFF;
        let jj = j & 0xFF;
        // int gi0 = this.permute(ii + this.permute(jj)) % 12; — operands are
        // non-negative (permute ≦ 255, ii/jj ≦ 255 and +1), so Java `%` == rem.
        let gi0 = (self.permute(ii + self.permute(jj) as i32) % 12) as usize;
        let gi1 = (self.permute(ii + i1 + self.permute(jj + j1) as i32) % 12) as usize;
        let gi2 = (self.permute(ii + 1 + self.permute(jj + 1) as i32) % 12) as usize;
        let n0 = self.get_corner_noise_3d(gi0, x0, y0, 0.0, 0.5);
        let n1 = self.get_corner_noise_3d(gi1, x1, y1, 0.0, 0.5);
        let n2 = self.get_corner_noise_3d(gi2, x2, y2, 0.0, 0.5);
        // return (float)(70.0 * (n0 + n1 + n2));
        (70.0 * (n0 + n1 + n2)) as f32 as f64
    }

    /// `public float get(double _xin, double _yin, double _zin)`
    /// (`SimplexNoise.java:77-168`).
    pub fn get_value_3d(&self, _xin: f64, _yin: f64, _zin: f64) -> f64 {
        let xin = _xin + self.offset_x;
        let yin = _yin + self.offset_y;
        let zin = _zin + self.offset_z;
        // double F3 = 0.3333333333333333; double s = (xin + yin + zin) * 0.3333333333333333;
        let s = (xin + yin + zin) * F3;
        let i = mth_floor(xin + s);
        let j = mth_floor(yin + s);
        let k = mth_floor(zin + s);
        // double G3 = 0.16666666666666666; double t = (i + j + k) * 0.16666666666666666;
        let t = ((i as f64) + (j as f64) + (k as f64)) * G3;
        let x0 = xin - ((i as f64) - t);
        let y0 = yin - ((j as f64) - t);
        let z0 = zin - ((k as f64) - t);
        // Rank ordering branch lattice (`SimplexNoise.java:101-145`), decompile order preserved.
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };
        let x1 = x0 - (i1 as f64) + G3;
        let y1 = y0 - (j1 as f64) + G3;
        let z1 = z0 - (k1 as f64) + G3;
        let x2 = x0 - (i2 as f64) + F3;
        let y2 = y0 - (j2 as f64) + F3;
        let z2 = z0 - (k2 as f64) + F3;
        let x3 = x0 - 1.0 + 0.5;
        let y3 = y0 - 1.0 + 0.5;
        let z3 = z0 - 1.0 + 0.5;
        let ii = i & 0xFF;
        let jj = j & 0xFF;
        let kk = k & 0xFF;
        let gi0 =
            (self.permute(ii + self.permute(jj + self.permute(kk) as i32) as i32) % 12) as usize;
        let gi1 = (self
            .permute(ii + i1 + self.permute(jj + j1 + self.permute(kk + k1) as i32) as i32)
            % 12) as usize;
        let gi2 = (self
            .permute(ii + i2 + self.permute(jj + j2 + self.permute(kk + k2) as i32) as i32)
            % 12) as usize;
        let gi3 = (self.permute(ii + 1 + self.permute(jj + 1 + self.permute(kk + 1) as i32) as i32)
            % 12) as usize;
        let n0 = self.get_corner_noise_3d(gi0, x0, y0, z0, 0.6);
        let n1 = self.get_corner_noise_3d(gi1, x1, y1, z1, 0.6);
        let n2 = self.get_corner_noise_3d(gi2, x2, y2, z2, 0.6);
        let n3 = self.get_corner_noise_3d(gi3, x3, y3, z3, 0.6);
        // return (float)(32.0 * (n0 + n1 + n2 + n3));
        (32.0 * (n0 + n1 + n2 + n3)) as f32 as f64
    }

    /// Test-only constructor injecting a known permutation table and offsets for
    /// hand-traced micro goldens (`#[cfg(test)]` surface per task spec).
    #[cfg(test)]
    fn from_parts(perms: [u8; 256], offset_x: f64, offset_y: f64, offset_z: f64) -> Self {
        Self {
            perms,
            offset_x,
            offset_y,
            offset_z,
        }
    }

    #[cfg(test)]
    fn perms_ref(&self) -> &[u8; 256] {
        &self.perms
    }
}

// ---------------------------------------------------------------------------
// NormalNoise (26.3 octave-stack parameter wrapper)
// ---------------------------------------------------------------------------

// --- Core Constants (NormalNoise.java, 26.3) ---

/// `TARGET_DEVIATION = 0.3333333333333333` (`NormalNoise.java:28-29`, used at
/// lines 98, 165, 59).
const TARGET_DEVIATION: f64 = 0.3333333333333333;

/// `INPUT_FACTOR = 1.0181268882175227` (`NormalNoise.java:26-27`, used at
/// line 183 for the second Perlin stack's frequency).
const INPUT_FACTOR: f64 = 1.0181268882175227;

/// Per-octave expected standard deviation of the base Perlin layer
/// (`NormalNoise.java:106`, literal `0.2702247831245211`).
const PERLIN_LAYER_DEVIATION: f64 = 0.2702247831245211;

/// `NormalNoise.Normalization` enum value space (`NormalNoise.java:323-341`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// `DISABLED` — no amplitude pre-scaling and no normalization pass.
    Disabled,
    /// `ENABLED` — Builder default (`NormalNoise.java:262`).
    Enabled,
    /// `LEGACY` (`NormalNoise.java:292-296`) — enables the parity
    /// normalization rewrite at `NormalNoise.java:52-56`.
    Legacy,
}

/// One octave slot: `NormalNoise.OctaveInfo(...)`
/// (`NormalNoise.java:343-351`).
#[derive(Clone, Debug, PartialEq)]
pub struct NormalNoiseOctave {
    /// `octaveIndex` — `baseOctave + i` (`NormalNoise.java:77`).
    pub octave_index: i32,
    /// `frequency` — doubling per octave starting at `2^baseOctave`
    /// (`NormalNoise.java:65,80`).
    pub frequency: f64,
    /// `amplitude` — base amplitude (optionally normalized) halved per octave,
    /// scaled by the per-index amplitude modifier (`NormalNoise.java:74-81`).
    pub amplitude: f64,
}

impl NormalNoiseOctave {
    /// `OctaveInfo::absAmplitude()` (`NormalNoise.java:348-350`).
    #[inline]
    pub fn abs_amplitude(&self) -> f64 {
        self.amplitude.abs()
    }
}

/// `NormalNoise.Parameters` (`NormalNoise.java:353-376`) with the fluent
/// `Builder`'s default values (`NormalNoise.java:258-264`).
#[derive(Clone, Debug, PartialEq)]
pub struct NormalNoiseParameters {
    /// `baseAmplitude`, Builder default `1.0`.
    pub base_amplitude: f64,
    /// `baseOctave`, Builder default `0` (createParity overrides with firstOctave).
    pub base_octave: i32,
    /// `octaveCount`, Builder default `1`.
    pub octave_count: i32,
    /// `normalize`, Builder default `ENABLED`.
    pub normalize: Normalization,
    /// `amplitudeModifiers` — empty replicates Java's `DoubleList.of()`, which
    /// makes `getAmplitudeModifier` return `1.0` for every index
    /// (`NormalNoise.java:87-89`). When non-empty its length must equal
    /// `octave_count` (codec validation, `NormalNoise.java:369-375`, enforced
    /// via `debug_assert` here).
    pub amplitude_modifiers: Vec<f64>,
}

impl Default for NormalNoiseParameters {
    /// Builder field defaults (`NormalNoise.java:259-263`).
    fn default() -> Self {
        Self {
            base_amplitude: 1.0,
            base_octave: 0,
            octave_count: 1,
            normalize: Normalization::Enabled,
            amplitude_modifiers: Vec::new(),
        }
    }
}

/// Port of the decompiled `NormalNoise` octave-stack wrapper. In 26.3 this class
/// carries the octave construction/normalization parameters and builds a
/// `NoiseStack` of `PerlinNoise` instances (`NormalNoise.java:41-60, 172-187`);
/// because `PerlinNoise` is outside this module's oracle set, evaluation is
/// reified by [`NormalNoise::get_value`] over an injected octave evaluator with
/// generation-order slots.
pub struct NormalNoise {
    /// `this.parameters` (`NormalNoise.java:36`).
    parameters: NormalNoiseParameters,
    /// `this.octaves` (`NormalNoise.java:37`), `buildOctaves` output
    /// (`NormalNoise.java:62-85`).
    octaves: Vec<NormalNoiseOctave>,
    /// `this.normalizationFactor` (`NormalNoise.java:38`).
    normalization_factor: f64,
    /// `this.range` = `Interval.ofSymmetric((float)(targetAmplitude * 1/3 * 6))`
    /// (`NormalNoise.java:59`), stored as the symmetric f64 magnitude accurate
    /// to the Java float return.
    range: f64,
}

impl NormalNoise {
    /// `this.parameters` view (`NormalNoise.java:36`).
    pub fn parameters(&self) -> &NormalNoiseParameters {
        &self.parameters
    }

    /// Private decompiled constructor `NormalNoise(Parameters)` over the
    /// Builder-equivalent field set (`NormalNoise.java:41-60`).
    pub fn from_parameters(parameters: NormalNoiseParameters) -> Self {
        debug_assert!(
            parameters.amplitude_modifiers.is_empty()
                || parameters.amplitude_modifiers.len() as i32 == parameters.octave_count,
            "amplitude_modifiers size must match octave_count (NormalNoise.java:369-375)"
        );
        // this.octaves = buildOctaves(..., normalize != DISABLED, ...)
        let octaves = Self::build_octaves(
            parameters.base_octave,
            parameters.base_amplitude,
            parameters.octave_count,
            parameters.normalize != Normalization::Disabled,
            &parameters.amplitude_modifiers,
        );
        // double targetAmplitude = octaves.stream().mapToDouble(OctaveInfo::absAmplitude).sum();
        let mut target_amplitude = octaves.iter().map(|o| o.abs_amplitude()).sum::<f64>();
        // double normalizationFactor = computeNormalizationFactor(targetAmplitude, octaves);
        let mut normalization_factor =
            Self::compute_normalization_factor(target_amplitude, &octaves);
        // LEGACY rewrite (`NormalNoise.java:52-56`).
        if parameters.normalize == Normalization::Legacy && normalization_factor != 0.0 {
            let parity_normalization_factor = Self::compute_parity_normalization_factor(
                parameters.base_amplitude,
                parameters.octave_count,
                &parameters.amplitude_modifiers,
            );
            target_amplitude *= parity_normalization_factor / normalization_factor;
            normalization_factor = parity_normalization_factor;
        }

        // this.range = Interval.ofSymmetric((float)(targetAmplitude * 1/3 * 6.0));
        let range = (target_amplitude * TARGET_DEVIATION * 6.0) as f32 as f64;

        Self {
            parameters,
            octaves,
            normalization_factor,
            range,
        }
    }

    /// `public static NormalNoise createParity(int firstOctave, DoubleList amplitudes)`
    /// (`NormalNoise.java:117-135`) — the constructor surface required by the
    /// task: `(first_octave, amplitudes)`.
    ///
    /// # Panics
    /// Panics exactly like the decompile's `IllegalArgumentException` when the
    /// amplitude list is empty ("Need at least 1 amplitude").
    pub fn create_parity(first_octave: i32, amplitudes: &[f64]) -> Self {
        if amplitudes.is_empty() {
            panic!("Need at least 1 amplitude");
        }

        // Builder-equivalent field assembly with Builder defaults:
        // baseOctave <- firstOctave, octaveCount <- amplitudes.size(),
        // baseAmplitude <- computeParityBaseAmplitude(...), normalize stays ENABLED.
        let base_amplitude = Self::compute_parity_base_amplitude(first_octave, amplitudes);
        let mut amplitude_modifiers: Vec<f64> = Vec::new();
        for (i, &amplitude) in amplitudes.iter().enumerate() {
            if amplitude != 1.0 {
                if amplitude_modifiers.is_empty() {
                    // Builder.setAmplitudeModifier materializes an all-1.0 list
                    // of octaveCount entries on first set (`NormalNoise.java:298-313`).
                    amplitude_modifiers = vec![1.0; amplitudes.len()];
                }
                amplitude_modifiers[i] = amplitude;
            }
        }
        Self::from_parameters(NormalNoiseParameters {
            base_amplitude,
            base_octave: first_octave,
            octave_count: amplitudes.len() as i32,
            normalize: Normalization::Enabled,
            amplitude_modifiers,
        })
    }

    /// `private static List<OctaveInfo> buildOctaves(...)` (`NormalNoise.java:62-85`).
    ///
    /// `Math.pow(2.0, baseOctave)` and `Math.pow(0.5, ±n)` take exact integral
    /// exponents (|…| ≤ 32 by the codec clamp), so `f64::powi` reproduces the
    /// exact representable value bit-for-bit.
    fn build_octaves(
        base_octave: i32,
        base_amplitude: f64,
        octave_count: i32,
        normalize: bool,
        amplitude_modifiers: &[f64],
    ) -> Vec<NormalNoiseOctave> {
        // double frequency = Math.pow(2.0, baseOctave);
        let mut frequency = 2.0f64.powi(base_octave);
        let mut amplitude = base_amplitude;
        if normalize {
            // amplitude *= Math.pow(0.5, -(octaveCount - 1)) / (Math.pow(0.5, -octaveCount) - 1.0);
            // Both powers take exact integral exponents; 0.5^(-n) == 2^n exactly,
            // so powi reproduces Java Math.pow bit-for-bit here.
            amplitude *= 0.5f64.powi(-(octave_count - 1)) / (0.5f64.powi(-octave_count) - 1.0);
        }
        let mut octaves = Vec::with_capacity(octave_count.max(0) as usize);
        for i in 0..octave_count {
            // double amplitudeModifier = getAmplitudeModifier(amplitudeModifiers, i);
            let amplitude_modifier = Self::get_amplitude_modifier(amplitude_modifiers, i);
            if amplitude_modifier != 0.0 {
                // double modifiedAmplitude = amplitude * amplitudeModifier;
                let modified_amplitude = amplitude * amplitude_modifier;
                octaves.push(NormalNoiseOctave {
                    octave_index: base_octave + i,
                    frequency,
                    amplitude: modified_amplitude,
                });
            }

            // frequency *= 2.0; amplitude *= 0.5;  (advance every iteration, decompile order)
            frequency *= 2.0;
            amplitude *= 0.5;
        }
        octaves
    }

    /// `private static double getAmplitudeModifier(DoubleList, int)`
    /// (`NormalNoise.java:87-89`).
    #[inline]
    fn get_amplitude_modifier(amplitude_modifiers: &[f64], index: i32) -> f64 {
        if amplitude_modifiers.is_empty() {
            1.0
        } else {
            amplitude_modifiers[index as usize]
        }
    }

    /// `private static double computeNormalizationFactor(double, List<OctaveInfo>)`
    /// (`NormalNoise.java:91-100`).
    fn compute_normalization_factor(target_amplitude: f64, octaves: &[NormalNoiseOctave]) -> f64 {
        let input_deviation = Self::estimate_deviation(octaves);
        if input_deviation == 0.0 {
            return 0.0;
        }
        // double inputSumDeviation = inputDeviation * Math.sqrt(2.0);
        let input_sum_deviation = input_deviation * 2.0f64.sqrt();
        // double targetDeviation = targetAmplitude * TARGET_DEVIATION;
        let target_deviation = target_amplitude * TARGET_DEVIATION;
        target_deviation / input_sum_deviation
    }

    /// `private static double estimateDeviation(List<OctaveInfo>)`
    /// (`NormalNoise.java:102-111`).
    fn estimate_deviation(octaves: &[NormalNoiseOctave]) -> f64 {
        let mut variance = 0.0;
        for octave in octaves {
            // double layerDeviation = 0.2702247831245211 * octave.absAmplitude();
            let layer_deviation = PERLIN_LAYER_DEVIATION * octave.abs_amplitude();
            // variance += Mth.square(layerDeviation) — `Mth.square(x)` is `x * x`.
            variance += layer_deviation * layer_deviation;
        }
        variance.sqrt()
    }

    /// `private static double computeParityBaseAmplitude(int, DoubleList)`
    /// (`NormalNoise.java:141-151`).
    fn compute_parity_base_amplitude(base_octave: i32, amplitudes: &[f64]) -> f64 {
        // List<OctaveInfo> octaves = buildOctaves(baseOctave, 1.0, amplitudes.size(), true, amplitudes);
        let octaves =
            Self::build_octaves(base_octave, 1.0, amplitudes.len() as i32, true, amplitudes);
        let target_amplitude: f64 = octaves.iter().map(|o| o.abs_amplitude()).sum();
        let new_normalization_factor =
            Self::compute_normalization_factor(target_amplitude, &octaves);
        if new_normalization_factor == 0.0 {
            return 1.0;
        }

        // double oldNormalizationFactor = computeParityNormalizationFactor(1.0, amplitudes.size(), amplitudes);
        let old_normalization_factor =
            Self::compute_parity_normalization_factor(1.0, amplitudes.len() as i32, amplitudes);
        old_normalization_factor / new_normalization_factor
    }

    /// `private static double computeParityNormalizationFactor(double, int, DoubleList)`
    /// (`NormalNoise.java:153-166`).
    fn compute_parity_normalization_factor(
        base_amplitude: f64,
        octave_count: i32,
        amplitude_modifiers: &[f64],
    ) -> f64 {
        // int minOctave = Integer.MAX_VALUE; int maxOctave = Integer.MIN_VALUE;
        let mut min_octave = i32::MAX;
        let mut max_octave = i32::MIN;

        for i in 0..octave_count {
            let modifier = Self::get_amplitude_modifier(amplitude_modifiers, i);
            if modifier != 0.0 {
                min_octave = min_octave.min(i);
                max_octave = max_octave.max(i);
            }
        }

        // If every modifier were 0.0, `maxOctave - minOctave` would overflow a
        // Java int (JLS 15.17.1 wrap): MIN - MAX == MIN wrapping to +1 here.
        // Deliberate `wrapping_sub` replicates that; normal configs have min/max
        // set (span = octaveRange) and never overflow.
        let octave_span = max_octave.wrapping_sub(min_octave);
        // return baseAmplitude * 0.5 * 0.3333333333333333 / parityExpectedDeviation(maxOctave - minOctave);
        base_amplitude * 0.5 * TARGET_DEVIATION / Self::parity_expected_deviation(octave_span)
    }

    /// `private static double parityExpectedDeviation(int octaveSpan)`
    /// (`NormalNoise.java:168-170`).
    #[inline]
    fn parity_expected_deviation(octave_span: i32) -> f64 {
        0.1 * (1.0 + 1.0 / ((octave_span as f64) + 1.0))
    }

    /// `this.octaves` view (construction output, `NormalNoise.java:43-49`).
    pub fn octaves(&self) -> &[NormalNoiseOctave] {
        &self.octaves
    }

    /// `this.normalizationFactor` (`NormalNoise.java:38, 58`).
    pub fn normalization_factor(&self) -> f64 {
        self.normalization_factor
    }

    /// `range()` (`NormalNoise.java:202-204`) — the symmetric magnitude of the
    /// `Interval.ofSymmetric((float)(targetAmplitude * 1/3 * 6.0))` construction.
    pub fn range(&self) -> f64 {
        self.range
    }

    /// `NormalNoise.create(random)` wiring (`NormalNoise.java:172-187`, with the
    /// stack evaluation of `NoiseStack.get` from `NoiseStack.java:36-45`).
    ///
    /// The decompile instantiates a `NoiseStack` of `PerlinNoise` layers: for
    /// each octave, a "first" Perlin at `octave.frequency` and a "second" at
    /// `octave.frequency * INPUT_FACTOR`, both weighted by the single f32
    /// `valueFactor = (float)(normalizationFactor * octave.amplitude)`
    /// (`NormalNoise.java:181-183`). Because `PerlinNoise` is outside this
    /// module's oracle set, the per-layer `Noise.get(x*frequency, …)` values are
    /// injected: `octave_noise(slot, x, y, z)` receives the layer's
    /// *frequency-scaled* coordinates and a dense layer ordinal `slot`
    /// (octave 0 contributes slots 0 first / 1 second, octave 1 → slots 2/3, …).
    /// Callers must build their Perlin instances — seeded
    /// `"octave_" + octave_index` — in exactly this `octaves()` order.
    ///
    /// Accumulation replicates the decompile's f32 arithmetic (`value += (float)
    /// (float amplitude * (float) noise)`), returning the f64-widened result of
    /// the Java `float` return.
    pub fn get_value<F>(&self, octave_noise: F, x: f64, y: f64, z: f64) -> f64
    where
        F: Fn(usize, f64, f64, f64) -> f64,
    {
        // float value = 0.0F;
        let mut value: f32 = 0.0;

        for (slot_index, octave) in self.octaves.iter().enumerate() {
            // float valueFactor = (float)(this.normalizationFactor * octave.amplitude);
            let value_factor = (self.normalization_factor * octave.amplitude) as f32;
            // stack.add(firstNoise, octave.frequency, valueFactor);
            let frequency = octave.frequency;
            value += value_factor
                * (octave_noise(2 * slot_index, x * frequency, y * frequency, z * frequency)
                    as f32);
            // stack.add(secondNoise, octave.frequency * INPUT_FACTOR, valueFactor);
            let second_frequency = octave.frequency * INPUT_FACTOR;
            value += value_factor
                * (octave_noise(
                    2 * slot_index + 1,
                    x * second_frequency,
                    y * second_frequency,
                    z * second_frequency,
                ) as f32);
        }

        value as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic u32 LCG used to drive the sweep helpers (`test_noise_*`):
    /// `state.wrapping_mul(1664525).wrapping_add(1013904223)`.
    struct TestRng(u32);

    impl TestRng {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0
        }

        /// Maps to ±1000.0 (2000.0 span), evenly injective across the range.
        fn next_coord(&mut self) -> f64 {
            (self.next_u32() as f64 / (u32::MAX as f64 + 1.0) - 0.5) * 2000.0
        }
    }

    /// Minimal-Simplex hand-rolled reference used ONLY as an independent test
    /// oracle for the permutation-table shuffle sequence (not for value goldens;
    /// those are hand-derived constants below).
    fn reference_shuffle_sequence(seed: i64) -> [u8; 256] {
        let mut rng = JavaRandomCompat::new(seed);
        // GradientNoise consumes three nextDouble offsets BEFORE the shuffle
        // (GradientNoise.java:37-50) — replicate the full draw order.
        let _ = rng.next_double() * 256.0;
        let _ = rng.next_double() * 256.0;
        let _ = rng.next_double() * 256.0;
        let mut perms = [0u8; 256];
        for i in 0..256usize {
            perms[i] = i as u8;
        }
        for i in 0..256usize {
            let offset = rng.next_int((256 - i) as i32) as usize;
            perms.swap(i, offset + i);
        }
        perms
    }

    #[test]
    fn test_noise_javarandomcompat_lcg_goldens() {
        // Canonical documented java.util.Random values, seed 0 (reviewer note:
        // provenance flagged — these trace the *documented* LCG, see module docs).
        // seed = (0 ^ 0x5DEECE66D) & (2^48 - 1); first next(48) state = 0x117C3D35B37E176...
        let mut rng = JavaRandomCompat::new(0);
        assert_eq!(rng.next_double(), 0.7309677873766570);
        assert_eq!(rng.next_double(), 0.24053641567148587);
        assert_eq!(rng.next_double(), 0.6374174253501083);

        // Power-of-two shortcut path: (int)((bound * (long)next(31)) >> 31).
        let mut rng = JavaRandomCompat::new(0);
        assert_eq!(rng.next_int(256), 187);
        assert_eq!(rng.next_int(256), 212);
        assert_eq!(rng.next_int(256), 61);
        assert_eq!(rng.next_int(256), 155);
        assert_eq!(rng.next_int(256), 163);

        // Non-power-of-two rejection path: bound = 7.
        let mut rng = JavaRandomCompat::new(0);
        assert_eq!(rng.next_int(7), 5);
        assert_eq!(rng.next_int(7), 2);
        assert_eq!(rng.next_int(7), 4);
        assert_eq!(rng.next_int(7), 2);
        assert_eq!(rng.next_int(7), 4);

        // Seed 12345: (seed ^ 0x5DEECE66D) & mask chains the same LCG.
        let mut rng = JavaRandomCompat::new(12345);
        assert_eq!(rng.next_double(), 0.3618031071604718);

        // nextInt(1): power-of-two shortcut collapses to 0 regardless of draw.
        let mut rng = JavaRandomCompat::new(42);
        assert_eq!(rng.next_int(1), 0);

        // Bounds are honoured.
        let mut rng = JavaRandomCompat::new(9);
        for _ in 0..1000 {
            let v = rng.next_int(13);
            assert!((0..13).contains(&v), "nextInt(13) out of bounds: {v}");
        }

        // `>>>` logical-shift semantics: next(24) must be non-negative for the
        // full unsigned range (top bit of 24-bit draw is 1 for ~half the draws).
        let mut rng = JavaRandomCompat::new(-777);
        let mut saw_top_bit = false;
        for _ in 0..200 {
            let v = rng.next_bits(24);
            assert!(v >= 0, ">>> must yield non-negative for bits < 32: {v}");
            saw_top_bit |= (v & 0x800000) != 0;
        }
        assert!(saw_top_bit, "negative seed must still exercise high draws");
    }

    #[test]
    fn test_noise_simplex_seeded_permutation_sequence() {
        // The shuffle consumes: 3 × nextDouble (offsets), then nextInt(256 - i)
        // per index (GradientNoise.java:37-50).
        //
        // Hand trace, seed 0x123456789ABC (documented LCG; flagged provenance):
        //   seed = (0x123456789ABC ^ 0x5DEECE66D) & (2^48-1)
        //   nextDouble(1) = ((next(26) << 27) + next(27)) * 2^-53 = 0.36346...
        //   offsets drawn: ox = nextDouble * 256.0, oy, oz
        //     ox = 93.00729481091008, oy = 250.29202523217708, oz = 170.86961024019678
        //   first shuffle draws (nextInt(256 - i)): 249, 164, 4, 230, 27
        //   swap 1: offset 249 → perms[0] <-> perms[249]: p[0]=249, p[249]=0
        //   swap 2: offset 164 → perms[1] <-> perms[165]: p[1]=165, p[165]=164
        let seed = 0x123456789ABCi64;
        let noise = SimplexNoise::new(&mut JavaRandomCompat::new(seed));
        let perms = noise.perms_ref();

        // Independently re-derived reference (same documented algorithm, in-test).
        assert_eq!(perms, &reference_shuffle_sequence(seed));

        // Spot constants (python-traced against the same op order):
        assert_eq!(perms[0], 140);
        assert_eq!(perms[1], 168);
        assert_eq!(perms[2], 49);
        assert_eq!(perms[3], 118);
        assert_eq!(perms[4], 92);
        assert_eq!(perms[126], 24);
        assert_eq!(perms[127], 28);
        assert_eq!(perms[128], 91);
        assert_eq!(perms[253], 208);
        assert_eq!(perms[254], 77);
        assert_eq!(perms[255], 175);
        // Still a permutation of 0..255 (bijective byte table).
        let mut seen = [false; 256];
        for &b in perms {
            seen[b as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "perm table must remain bijective");
    }

    #[test]
    fn test_noise_simplex_offset_handling_branches() {
        // with_offset_handling == GradientNoise(random, 0.0): offsets become
        // exact 0.0 while the RNG consumption stays IDENTICAL to new()
        // (offsets are still drawn before multiplication — SimplexNoise.java:18-20).
        let seed = 0x1BADB002i64;
        let mut r_a = JavaRandomCompat::new(seed);
        let mut r_b = JavaRandomCompat::new(seed);
        let with_zero = SimplexNoise::with_offset_handling(&mut r_b);
        let _default = SimplexNoise::new(&mut r_a);

        // Identical draw counts: next states must coincide after construction.
        assert_eq!(r_a.next_double(), r_b.next_double());
        assert_eq!(r_a.next_int(97), r_b.next_int(97));

        // Zero-offset instance samples without any offset translation: compare
        // against a hand-built zero-offset instance sharing the same table.
        let zero_twin = SimplexNoise::from_parts(*with_zero.perms_ref(), 0.0, 0.0, 0.0);
        assert_eq!(
            with_zero.get_value_2d(3.5, 4.5),
            zero_twin.get_value_2d(3.5, 4.5),
            "with_offset_handling must sample with exactly-zero offsets"
        );
        // And the default constructor must translate: same table, different sample.
        let noise_default = SimplexNoise::new(&mut JavaRandomCompat::new(seed));
        assert_ne!(
            noise_default.get_value_2d(3.5, 4.5),
            zero_twin.get_value_2d(3.5, 4.5),
            "default 256.0 offsets must shift sampled values"
        );
        assert_eq!(
            with_zero.perms_ref(),
            noise_default.perms_ref(),
            "same seed => same shuffle, offsets do not change consumption"
        );
    }

    #[test]
    fn test_noise_simplex_fixed_seed_determinism() {
        // Two same-seeded instances must be bit-identical everywhere.
        let seed = 0xABCDEF12i64;
        let noise_a = SimplexNoise::new(&mut JavaRandomCompat::new(seed));
        let noise_b = SimplexNoise::new(&mut JavaRandomCompat::new(seed));
        assert_eq!(noise_a.perms_ref(), noise_b.perms_ref());

        let mut rng = TestRng(0xDEADBEEF);
        // 250 2D points and 250 3D points, swept ±1000 (task-mandated spread).
        for _ in 0..250 {
            let x = rng.next_coord();
            let y = rng.next_coord();
            let z = rng.next_coord();
            let v2_a = noise_a.get_value_2d(x, y);
            let v2_b = noise_b.get_value_2d(x, y);
            assert!(
                v2_a.to_bits() == v2_b.to_bits(),
                "2D bit mismatch at ({x}, {y}): {v2_a} vs {v2_b}"
            );
            let v3_a = noise_a.get_value_3d(x, y, z);
            let v3_b = noise_b.get_value_3d(x, y, z);
            assert!(
                v3_a.to_bits() == v3_b.to_bits(),
                "3D bit mismatch at ({x}, {y}, {z}): {v3_a} vs {v3_b}"
            );
        }
    }

    #[test]
    fn test_noise_simplex_2d_microtrace_golden() {
        // Hand-traced single 2D sample. Construction: identity permutation table
        // injected via the test-only `from_parts`, offsets pinned to 0.0.
        // Sample: get_value_2d(1.234, -0.567). Arithmetic follows
        // SimplexNoise.java:41-74 with f64 = Java double, in exact order:
        //
        //   xin = 1.234,  yin = -0.567
        //   SQRT_3 = 1.7320508075688772, F2 = 0.3660254037844386, G2 = 0.21132486540518713
        //   s = (1.234 + -0.567) * F2 = 0.667 * 0.36602540378443864 = 0.24413894432422056
        //   i = floor(1.234 + 0.24413894432422056) = floor(1.4781389443242206) = 1
        //   j = floor(-0.567 + 0.24413894432422056) = floor(-0.32286105567577944) = -1   (negative floor!)
        //   t = (1 + -1) * G2 = 0.0
        //   x0 = 1.234 - (1 - 0.0) = 0.23399999999999999
        //   y0 = -0.567 - (-1 - 0.0) = 0.43300000000000005
        //   x0 > y0? 0.234 > 0.433 → NO → i1 = 0, j1 = 1
        //   x1 = x0 - 0 + G2 = 0.23399999999999999 + 0.21132486540518713 = 0.4453248654051871
        //   y1 = y0 - 1 + G2 = -0.56699999999999995 + 0.21132486540518713 = -0.3556751345948128
        //   x2 = x0 - 1.0 + 2.0*G2 = -0.76...+0.42264973081037426 = -0.34335026918962575
        //   y2 = y0 - 1.0 + 0.42264973081037426 = -0.14435026918962568
        //   ii = 1 & 0xFF = 1, jj = -1 & 0xFF = 255
        //   identity perms: permute(x) = x & 0xFF
        //   gi0 = permute(1 + permute(255)) % 12 = permute(256) % 12 = 0 % 12 = 0
        //   gi1 = permute(1 + 0 + permute(255+1)) % 12 = permute(1 + 0) % 12 = 1
        //   gi2 = permute(1 + 1 + permute(256)) % 12 = permute(2) % 12 = 2
        //   corner(gi0, x0, y0, 0, 0.5):  t0 = 0.5 - x0² - y0² = 0.25775499999999996
        //     t0 *= t0 → 0.06643764002499998; t0*t0 = 0.0044139600120914795
        //     grad[0] = (1,1,0), dot = x0 + y0 = 0.667
        //     n0 = 0.0044139600120914795 * 0.667 = 0.002944111328065017
        //   corner(gi1, x1, y1, 0, 0.5):  t0 = 0.1751809628828138 →⁴ 0.0009417760383147193
        //     grad[1] = (-1,1,0), dot = -x1 + y1 = -0.8009999999999999
        //     n1 = -0.00075436260669009
        //   corner(gi2, x2, y2, 0, 0.5):  t0 = 0.3612735924322941 →⁴ 0.01703510718799171
        //     grad[2] = (1,-1,0), dot = x2 - y2 = -0.19900000000000007
        //     n2 = -0.003389986330410351
        //   sum = -0.0012002376090354244; 70.0 * sum = -0.0840166326324797
        //   (float) cast → f32 −0.08401663601398468 (0xbfB5821D40000000 widened)
        let identity = {
            let mut perms = [0u8; 256];
            for (i, p) in perms.iter_mut().enumerate() {
                *p = i as u8;
            }
            perms
        };
        let noise = SimplexNoise::from_parts(identity, 0.0, 0.0, 0.0);
        let golden = -0.08401663601398468f64;
        let v = noise.get_value_2d(1.234, -0.567);
        assert!(
            v.to_bits() == golden.to_bits(),
            "microtrace golden mismatch: {v} (expected {golden})"
        );

        // Second trace with a genuinely permuted table: perms[i] = (17i + 31) & 0xFF
        // (gcd(17, 256) == 1 → bijective). Same geometry (x0/y0/x1/y1/x2/y2
        // identical to the identity trace); gradients resolve differently:
        //   permute(255) = (17*255+31)&255 = 255; permute(256) = perms[0] = 31
        //   gi0 = permute(1 + permute(255)) % 12 = permute(256) % 12 = 31 % 12 = 6
        //   gi1 = permute(1 + 0 + permute(256)) % 12 = perms[(1+31)&255=(17*32+31)&255=231]
        //         → permute(1+0+31) = (17*32+31)&255 = 231 → 231 % 12 = 3
        //   gi2 = permute(1 + 1 + permute(256)) % 12 = permute(2 + 31) = (17*33+31)&255
        //         = 248 → 248 % 12 = 8
        //   n0 = 0.0010328666428294061, n1 = -8.442996831857535e-05, n2 = -0.0024590223082607307
        //   sum = -0.0015105856337498998; 70.0 * sum = -0.10574099436249298
        //   (float) cast → f32 widened −0.1057409942150116
        let derived: [u8; 256] = core::array::from_fn(|i| ((17 * i + 31) & 0xFF) as u8);
        let noise_p = SimplexNoise::from_parts(derived, 0.0, 0.0, 0.0);
        let golden_p = -0.1057409942150116f64;
        let v_p = noise_p.get_value_2d(1.234, -0.567);
        assert!(
            v_p.to_bits() == golden_p.to_bits(),
            "permuted-table golden mismatch: {v_p} (expected {golden_p})"
        );
    }

    /// Sanity net (NOT an oracle): 100k pseudo-random samples must stay inside
    /// the SPECS.md §2 RANGE pin ±2.0 and roughly zero-centered.
    fn assert_envelope_2d_3d() {
        let noise = SimplexNoise::new(&mut JavaRandomCompat::new(0x123456789ABC));
        let mut rng = TestRng(12345678);
        let n = 100_000;
        let mut sum2 = 0.0f64;
        let mut max2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let mut max3 = 0.0f64;
        for _ in 0..n {
            let x = rng.next_coord();
            let y = rng.next_coord();
            let z = rng.next_coord();
            let v2 = noise.get_value_2d(x, y);
            let v3 = noise.get_value_3d(x, y, z);
            assert!(
                v2.is_finite() && v3.is_finite(),
                "noise must stay finite: {v2} / {v3} at ({x},{y},{z})"
            );
            // RANGE pin (SimplexNoise.java:8): symmetric ±2.0 (sanity net).
            assert!(
                v2.abs() <= RANGE,
                "2D range pin violated: {v2} at ({x},{y})"
            );
            assert!(
                v3.abs() <= RANGE,
                "3D range pin violated: {v3} at ({x},{y},{z})"
            );
            sum2 += v2;
            max2 = max2.max(v2.abs());
            sum3 += v3;
            max3 = max3.max(v3.abs());
        }
        // Distribution check: rough mean window ±0.05 (sanity net, not oracle).
        let mean2 = sum2 / (n as f64);
        let mean3 = sum3 / (n as f64);
        assert!(
            mean2.abs() < 0.05,
            "2D mean drifted: {mean2} (max |v| {max2})"
        );
        assert!(
            mean3.abs() < 0.05,
            "3D mean drifted: {mean3} (max |v| {max3})"
        );
    }

    #[test]
    fn test_noise_simplex_range_envelope_2d_3d() {
        assert_envelope_2d_3d();
    }

    #[test]
    fn test_noise_normalnoise_parity_construction_golden() {
        // createParity(-8, [1,1,1,0]) — hand derivation (f64, exact; python f64
        // trace of decompile order):
        //   computeParityBaseAmplitude(-8, amps):
        //     buildOctaves(-8, 1.0, 4, true, amps):
        //       normalize: amplitude *= 0.5^-(3) / (0.5^-4 - 1.0) = 8.0/15.0
        //         = 0.5333333333333333
        //       i=0: (-8, freq 0.25, amp 1.0*0.5333333333333333) ... i=2: amp halves
        //       i=3: modifier 0.0 → octave SKIPPED (freq still doubles)
        //         amps: [0.5333333333333333, 0.26666666666666666, 0.13333333333333333]
        //     target_amplitude = Σ|amp| = 0.9333333333333333
        //     new_nf = cnf: inputDeviation = sqrt((0.2702247831245211*(0.5333..+0.2667..+0.1333..))...)
        //       = sqrt(Σ (0.2702247831245211*ampi)²) = 0.2104... ; * √2 divide:
        //       nf = (0.9333333333333333 * 1/3)/(dev * √2) = 1.3323764968871814
        //     old_nf = cpnf: min=0, max=2 (modifier[3] = 0.0 excluded), span 2,
        //       dev = 0.1*(1 + 1/3) = 0.13333333333333333 →
        //       cpnf = 1.0 * 0.5 * (1/3) / 0.13333333333333333 = 1.25
        //     base_amplitude = 1.25 / 1.3323764968871814 = 0.9381732587751005
        //   from_parameters(base_amplitude, -8, 4, ENABLED, amps):
        //     octaves scaled by base_amplitude; nf recomputed = 1.3323764968871814
        //     target_amplitude = 0.8756283748567606
        //     range = f32(0.8756..*1/3*6) = f32(1.7512567043304441) → 1.7512567043304443
        let noise = NormalNoise::create_parity(-8, &[1.0, 1.0, 1.0, 0.0]);
        assert_eq!(noise.octaves().len(), 3);
        assert_eq!(noise.octaves()[0].octave_index, -8);
        assert_eq!(noise.octaves()[2].octave_index, -6);
        assert_eq!(
            noise.octaves()[0].frequency.to_bits(),
            0.00390625f64.to_bits(), // 2^baseOctave = 2^-8 exactly
            "frequency = 2^baseOctave exactly"
        );
        assert_eq!(
            noise.octaves()[1].frequency.to_bits(),
            0.0078125f64.to_bits(), // 2^-7: doubles per octave exactly
            "frequency doubles per octave exactly"
        );
        assert_eq!(
            noise.normalization_factor().to_bits(),
            f64::from_bits(0x3ff5516a04816e8d).to_bits(),
            "normalization factor golden 1.3323764968871814"
        );
        assert_eq!(
            noise.range().to_bits(),
            f64::from_bits(0x3ffc0525c0000000).to_bits(),
            "range golden f32-rounded 1.7512567043304443"
        );

        // Amplitude modifier path: createParity(-15, [1,1,1,1,1,1,2,0]):
        //   octaves: index 6 gets modifier 2.0 (1.5x default amp), index 7 (0.0) skipped:
        //   base_amplitude golden 0.96552518795937 (bits 0x3feee5951437fdee);
        //   nf golden 1.5104042354560523 (bits 0x3ff82a9da1b060c8);
        //   range golden 1.938623070716858 (bits 0x3fff0499a0000000).
        let amps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0];
        let noise = NormalNoise::create_parity(-15, &amps);
        assert_eq!(noise.octaves().len(), 7);
        assert_eq!(
            noise.normalization_factor().to_bits(),
            f64::from_bits(0x3ff82a9da1b060c8).to_bits(),
            "legacy modified-amplitude parity golden"
        );
        assert_eq!(
            noise.range().to_bits(),
            f64::from_bits(0x3fff0499a0000000).to_bits(),
        );
    }

    #[test]
    fn test_noise_normalnoise_legacy_normalization_path() {
        // LEGACY rewrite (NormalNoise.java:52-56): normalization factor is
        // replaced by the parity factor and target amplitude rescaled.
        // from_parameters(1.0, -15, 4, LEGACY, [1,1,1,0]):
        //   buildOctaves normalizes base amplitude: 8/15 = 0.5333333333333333 …
        //   nf before rewrite = 1.3323764968871818
        //   parity_nf: min=0, max=2, span 2, cpnf = 0.5*⅓/0.13333333333333333 = 1.25
        //   target *= 1.25/1.3323764968871818 = 0.8756283748567605
        //   final nf = 1.25 (bits 0x3ff4000000000000)
        //   range = f32(0.8756… * 1/3 * 6) = 1.7512567043304443
        let noise = NormalNoise::from_parameters(NormalNoiseParameters {
            base_amplitude: 1.0,
            base_octave: -15,
            octave_count: 4,
            normalize: Normalization::Legacy,
            amplitude_modifiers: vec![1.0, 1.0, 1.0, 0.0],
        });
        assert_eq!(
            noise.normalization_factor().to_bits(),
            f64::from_bits(0x3ff4000000000000).to_bits(),
            "legacy normalization must substitute the parity factor (1.25)"
        );
        assert_eq!(
            noise.range().to_bits(),
            f64::from_bits(0x3ffc0525c0000000).to_bits(),
            "legacy target rescale preserves range golden"
        );

        // DISABLED path with no modifiers: nf = 1.3323764968871812
        // (bits 0x3ff5516a04816e8c, last ulp differs from the ENABLED/normalized
        // variant), target = 1.75 (halving without 8/15 scale-up), range = 3.5.
        let noise = NormalNoise::from_parameters(NormalNoiseParameters {
            base_amplitude: 1.0,
            base_octave: -2,
            octave_count: 3,
            normalize: Normalization::Disabled,
            amplitude_modifiers: Vec::new(),
        });
        assert_eq!(
            noise.normalization_factor().to_bits(),
            f64::from_bits(0x3ff5516a04816e8c).to_bits(),
            "disabled-normalization factor golden"
        );
        assert_eq!(
            noise.range().to_bits(),
            f64::from_bits(0x400c000000000000).to_bits(),
            "disabled-normalization range golden (3.5)"
        );
    }

    #[test]
    fn test_noise_normalnoise_octave_decay_and_determinism() {
        // Octave decay property with a bounded deterministic oct-the-evaluator:
        // |value| ≤ Σ valueFactor (mock |output| ≤ 1.0), finite everywhere,
        // bit-stable across identical calls.
        let noise = NormalNoise::create_parity(-8, &[1.0, 1.0, 1.0, 0.0]);
        let noise_negative =
            NormalNoise::create_parity(-15, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0]);

        // Deterministic bounded evaluator (mock, NOT a decompile port): three
        // trig terms normalized to [-1, 1] with slot-dependent phase.
        let mock = |slot: usize, x: f64, y: f64, z: f64| -> f64 {
            let s = slot as f64;
            ((x * 1.7 + s * 0.3).sin() + (y * 2.3 + s * 0.7).cos() + (z * 0.9 + s * 1.1).sin())
                / 3.0
        };

        // Bound = Σ |valueFactor| over both stack layers per active octave
        // (|value| ≤ Σ vf · 1.0 exactly, mock bounded by ±1).
        let bound_for = |n: &NormalNoise| -> f64 {
            n.octaves()
                .iter()
                .map(|o| (n.normalization_factor() * o.amplitude).abs() * 2.0)
                .sum::<f64>()
                + 1.0e-4 // f32 rounding headroom
        };

        let mut rng = TestRng(0xFEEDFACE);
        for _ in 0..4096 {
            let x = rng.next_coord();
            let y = rng.next_coord();
            let z = rng.next_coord();
            let v = noise.get_value(&mock, x, y, z);
            assert!(
                v.is_finite(),
                "value must be finite, got {v} at ({x},{y},{z})"
            );
            assert!(
                v.abs() <= bound_for(&noise),
                "octave decay bound violated: |{v}| > {}",
                bound_for(&noise)
            );
            let vn = noise_negative.get_value(&mock, x, y, z);
            assert!(vn.is_finite(), "value must be finite, got {vn}");
            assert!(
                vn.abs() <= bound_for(&noise_negative),
                "bound violated: {vn}"
            );

            // Determinism: same inputs → bit-identical re-evaluation.
            let v_again = noise.get_value(&mock, x, y, z);
            assert!(
                v.to_bits() == v_again.to_bits(),
                "NormalNoise evaluation not deterministic at ({x},{y},{z})"
            );
        }
    }

    #[test]
    fn test_noise_normalnoise_parity_panic_on_empty() {
        let result = std::panic::catch_unwind(|| NormalNoise::create_parity(-1, &[]));
        assert!(
            result.is_err(),
            "empty amplitude list must panic (decompile: IllegalArgumentException)"
        );
    }
}
