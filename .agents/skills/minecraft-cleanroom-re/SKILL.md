---
name: minecraft-cleanroom-re
description: Clean-room reverse engineering and specification extraction for Minecraft 26.3. Uses single-class Vineflower decompilation, out-of-repo bytecode isolation (~/.minecraft-decomp/), shaderpack analysis, and Data-Oriented Design (DOD) translation for the Rust + wgpu voxel engine. Use when extracting physics constants, block models, noise parameters, or inspecting shader algorithms from Prism Launcher.
---

# Minecraft 26.3 Clean-Room Reverse Engineering & Specification Skill

## Core Philosophy: Clean-Room & Data-Oriented Design (DOD)

1. **Decompiled Code is an Oracle / Spec, Not Source Code**:
   Never copy or port Java OOP structures (`BlockPos`, class inheritance, heap-allocated entities) into the Rust engine. Extract only mathematical invariants, state transitions, bounding box formulas, and tick timings into `SPECS.md`.
2. **Strict Out-of-Repo Isolation**:
   All decompiled `.java` files, raw extracted `.class` files, and intermediate decompiler dumps MUST remain outside the git workspace at `~/.minecraft-decomp/`. Only finalized documentation (`SPECS.md`), assets (textures/models), and native Rust/WGSL code belong in the repository.

---

## Environment & Key Locations

- **Client JAR (26.3)**: `~/.local/share/PrismLauncher/libraries/com/mojang/minecraft/26.3/minecraft-26.3-client.jar`
- **Reference Shaders**: `~/.local/share/PrismLauncher/instances/26.3/minecraft/shaderpacks/`
- **Reference Mod Optimizers**: `~/.local/share/PrismLauncher/instances/26.3/minecraft/mods/` (`sodium`, `lithium`, `DistantHorizons`)
- **Decompiler Binary**: `~/.local/bin/vineflower` (wrapped Java CLI)
- **Isolated Decompile Cache**: `~/.minecraft-decomp/`

---

## Playbooks

### Playbook 1: Targeted Single-Class Extraction & Decompilation

Do not decompile the entire 50MB+ client JAR at once. Extract and decompile only the specific class needed.

```bash
# 1. Locate the class path inside the JAR
unzip -l ~/.local/share/PrismLauncher/libraries/com/mojang/minecraft/26.3/minecraft-26.3-client.jar | grep -i "Player.class"

# 2. Extract single target class into out-of-repo scratch dir
mkdir -p ~/.minecraft-decomp/classes
unzip -p ~/.local/share/PrismLauncher/libraries/com/mojang/minecraft/26.3/minecraft-26.3-client.jar "net/minecraft/world/entity/player/Player.class" > ~/.minecraft-decomp/classes/Player.class

# 3. Decompile with Vineflower
mkdir -p ~/.minecraft-decomp/src
vineflower ~/.minecraft-decomp/classes/Player.class ~/.minecraft-decomp/src/

# 4. View and analyze
cat ~/.minecraft-decomp/src/Player.java | grep -E "(jump|gravity|motion|deltaMovement)"
```

---

### Playbook 2: Shaderpack Pass & Uniform Analysis

Inspect GLSL shaders from `Complementary`, `photon`, or `rethinking-voxels` to design matching WGSL pipelines:

```bash
# Example: Inspect Rethinking Voxels raytraced lighting pipeline
unzip -l ~/.local/share/PrismLauncher/instances/26.3/minecraft/shaderpacks/rethinking-voxels_r0.1-beta9.zip | grep -E "\.(vsh|fsh|csh)"

# Unpack a specific shader pass to out-of-repo scratch
unzip -p ~/.local/share/PrismLauncher/instances/26.3/minecraft/shaderpacks/rethinking-voxels_r0.1-beta9.zip "shaders/deferred1.fsh" > ~/.minecraft-decomp/deferred1.glsl
```

---

### Playbook 3: Optimizer Mod Architecture (Sodium / Lithium)

Decompile Sodium's vertex packing and indirect draw routines for direct reference:

```bash
# Unpack Sodium chunk vertex builder
mkdir -p ~/.minecraft-decomp/sodium
vineflower ~/.local/share/PrismLauncher/instances/26.3/minecraft/mods/sodium-fabric-*.jar ~/.minecraft-decomp/sodium/
find ~/.minecraft-decomp/sodium -name "*Vertex*.java" -o -name "*ChunkBuilder*.java"
```

---

### Playbook 4: Translating OOP to Rust DOD

| Java Minecraft OOP Pattern | Rust Voxel DOD Equivalent | Rationale |
| :--- | :--- | :--- |
| `new BlockPos(x, y, z)` | Flattened array index: `(y << 8) | (z << 4) | x` or `u16` | Eliminates heap allocations and pointer chasing; fits in CPU L1 cache. |
| `BlockState state = chunk.getBlock(x, y, z)` | Direct palette index lookup: `palette[chunk.blocks[index]]` | Constant O(1) memory access, SIMD vectorizable. |
| Quad Vertices: 4x 32-byte floats (128 bytes/quad) | 4x 4-byte bit-packed `u32` (16 bytes/quad) | 8x reduction in GPU bus transfer bandwidth and VRAM usage. |
| Synchronized World Tick Loop | Rayon work-stealing threadpool + Channel messages (`flume` / `crossbeam`) | Lock-free concurrency with deterministic simulation states. |

---

### Playbook 5: Logging Findings to `SPECS.md`

Whenever constants or mechanics are extracted, append them immediately to `SPECS.md` under the appropriate subsystem:
- Physics constants (gravity, drag, jump velocity)
- Bounding box sizes (width, height, eye height)
- Block light falloff equations (15 -> 0 attenuation curves)
- Noise octaves and frequency scales for biomes
