//! Clean-Room Minecraft 26.3 Physics & AABB Collision Engine.
//!
//! Implements discrete 20 Hz tick simulation (§1 of SPECS.md) decompile-verified
//! against `minecraft-26.3-client.jar`.

use glam::DVec3;

// --- Core Constants (Minecraft 26.3 Oracle) ---

/// Base gravity registered on `Attributes.GRAVITY` (`Attributes.java:56-58`).
pub const GRAVITY: f64 = 0.08;

/// Base jump power registered on `Attributes.JUMP_STRENGTH` (`Attributes.java:59-61`, `LivingEntity.java:2377-2403`).
pub const JUMP_IMPULSE: f64 = 0.42;

/// Sprint-jump horizontal impulse boost (`LivingEntity.java:2397-2400`).
pub const SPRINT_JUMP_BOOST: f64 = 0.2;

/// Vertical air drag multiplier (`LivingEntity.java:210, 2465-2498`).
pub const VERTICAL_DRAG: f64 = 0.98;

/// Horizontal air drag multiplier (`LivingEntity.java:209, 2493-2497`).
pub const HORIZONTAL_DRAG_AIR: f64 = 0.91;

/// Terminal fall velocity: closed form -0.08 * 0.98 / (1.0 - 0.98) = -3.92 blocks/tick.
pub const TERMINAL_VELOCITY: f64 = -3.92;

/// Player base walk speed (`Player.java:213`, `Attributes.java:78-80`).
pub const WALK_SPEED: f64 = 0.1;

/// Player sprint speed: 0.1 * 1.3 = 0.13 (`LivingEntity.java:168`).
pub const SPRINT_SPEED: f64 = 0.13;

/// Air steering speed walking (`Player.java:1873-1877`).
pub const AIR_STEERING_WALK: f64 = 0.02;

/// Air steering speed sprinting (`Player.java:1873-1877`).
pub const AIR_STEERING_SPRINT: f64 = 0.025999999;

/// Ground acceleration multiplier constant (`LivingEntity.java:2723-2729`).
pub const GROUND_ACCEL_CONSTANT: f64 = 0.21600002;

/// Collision precision epsilon (`Shapes.java:23`).
pub const COLLISION_EPSILON: f64 = 1.0e-7;

/// Player width (`EntityTypes.java:1105-1115`).
pub const PLAYER_WIDTH: f64 = 0.6;

/// Player standing height (`EntityTypes.java:1105-1115`).
pub const PLAYER_HEIGHT: f64 = 1.8;

/// Player eye height above foot position (`EntityTypes.java:1105-1115`).
pub const EYE_HEIGHT: f64 = 1.62;

/// Max step-up height attribute default (`Attributes.java:99-101`, `Entity.java:1196-1214`).
pub const MAX_UP_STEP: f64 = 0.6;

/// Negligible horizontal velocity squared threshold for players (`LivingEntity.java:3071-3074`).
pub const PLAYER_NEGLIGIBLE_HORIZ_SQ: f64 = 9.0e-6;

/// Negligible vertical velocity absolute threshold for all entities (`LivingEntity.java:3084-3086`).
pub const NEGLIGIBLE_VERT: f64 = 0.003;

/// Ground friction probe offset below feet (`Entity.java:217, 1102`).
pub const GROUND_PROBE_OFFSET: f64 = 0.500001;

/// Grounded-box negative nudge for step-up candidate collection (`Entity.java:1200`).
pub const STEP_UP_NUDGE: f64 = 1.0e-5;

/// Axis-Aligned Bounding Box (AABB) in world space (f64 precision).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl Aabb {
    #[inline(always)]
    pub fn new(min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64) -> Self {
        Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    /// Constructs the player bounding box around foot position `pos`.
    /// Dimensions: 0.6 x 1.8 x 0.6 centered horizontally (`[-0.3, 0.0, -0.3] .. [0.3, 1.8, 0.3]`).
    #[inline(always)]
    pub fn player_box(pos: DVec3) -> Self {
        let half_w = PLAYER_WIDTH * 0.5;
        Self::new(
            pos.x - half_w,
            pos.y,
            pos.z - half_w,
            pos.x + half_w,
            pos.y + PLAYER_HEIGHT,
            pos.z + half_w,
        )
    }

    /// Translates this AABB by `delta`.
    #[inline(always)]
    pub fn move_by(&self, delta: DVec3) -> Self {
        Self::new(
            self.min_x + delta.x,
            self.min_y + delta.y,
            self.min_z + delta.z,
            self.max_x + delta.x,
            self.max_y + delta.y,
            self.max_z + delta.z,
        )
    }

    /// Expands this AABB to enclose the movement `delta`.
    #[inline(always)]
    pub fn expand_towards(&self, delta: DVec3) -> Self {
        let min_x = if delta.x < 0.0 {
            self.min_x + delta.x
        } else {
            self.min_x
        };
        let max_x = if delta.x > 0.0 {
            self.max_x + delta.x
        } else {
            self.max_x
        };
        let min_y = if delta.y < 0.0 {
            self.min_y + delta.y
        } else {
            self.min_y
        };
        let max_y = if delta.y > 0.0 {
            self.max_y + delta.y
        } else {
            self.max_y
        };
        let min_z = if delta.z < 0.0 {
            self.min_z + delta.z
        } else {
            self.min_z
        };
        let max_z = if delta.z > 0.0 {
            self.max_z + delta.z
        } else {
            self.max_z
        };
        Self::new(min_x, min_y, min_z, max_x, max_y, max_z)
    }

    /// Checks if this AABB overlaps with `other`.
    #[inline(always)]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.max_x > other.min_x
            && self.min_x < other.max_x
            && self.max_y > other.min_y
            && self.min_y < other.max_y
            && self.max_z > other.min_z
            && self.min_z < other.max_z
    }

    /// Collides this moving AABB along X against an obstacle AABB (`VoxelShape.java:248-266`).
    pub fn collide_x(&self, obstacle: &Aabb, mut dx: f64) -> f64 {
        // Must overlap in transverse axes Y and Z
        if self.max_y > obstacle.min_y
            && self.min_y < obstacle.max_y
            && self.max_z > obstacle.min_z
            && self.min_z < obstacle.max_z
        {
            if dx > 0.0 && self.max_x <= obstacle.min_x {
                let dist = obstacle.min_x - self.max_x;
                if dist >= -COLLISION_EPSILON {
                    dx = dx.min(dist);
                }
            } else if dx < 0.0 && self.min_x >= obstacle.max_x {
                let dist = obstacle.max_x - self.min_x;
                if dist <= COLLISION_EPSILON {
                    dx = dx.max(dist);
                }
            }
        }
        dx
    }

    /// Collides this moving AABB along Y against an obstacle AABB (`VoxelShape.java:248-266`).
    pub fn collide_y(&self, obstacle: &Aabb, mut dy: f64) -> f64 {
        // Must overlap in transverse axes X and Z
        if self.max_x > obstacle.min_x
            && self.min_x < obstacle.max_x
            && self.max_z > obstacle.min_z
            && self.min_z < obstacle.max_z
        {
            if dy > 0.0 && self.max_y <= obstacle.min_y {
                let dist = obstacle.min_y - self.max_y;
                if dist >= -COLLISION_EPSILON {
                    dy = dy.min(dist);
                }
            } else if dy < 0.0 && self.min_y >= obstacle.max_y {
                let dist = obstacle.max_y - self.min_y;
                if dist <= COLLISION_EPSILON {
                    dy = dy.max(dist);
                }
            }
        }
        dy
    }

    /// Collides this moving AABB along Z against an obstacle AABB (`VoxelShape.java:248-266`).
    pub fn collide_z(&self, obstacle: &Aabb, mut dz: f64) -> f64 {
        // Must overlap in transverse axes X and Y
        if self.max_x > obstacle.min_x
            && self.min_x < obstacle.max_x
            && self.max_y > obstacle.min_y
            && self.min_y < obstacle.max_y
        {
            if dz > 0.0 && self.max_z <= obstacle.min_z {
                let dist = obstacle.min_z - self.max_z;
                if dist >= -COLLISION_EPSILON {
                    dz = dz.min(dist);
                }
            } else if dz < 0.0 && self.min_z >= obstacle.max_z {
                let dist = obstacle.max_z - self.min_z;
                if dist <= COLLISION_EPSILON {
                    dz = dz.max(dist);
                }
            }
        }
        dz
    }
}

/// World collision provider interface.
pub trait VoxelWorldCollision {
    /// Returns true if voxel at integer coordinates `(x, y, z)` is solid (`is_solid == true`).
    fn is_solid_at(&self, x: i32, y: i32, z: i32) -> bool;

    /// Ground friction at integer coordinates `(x, y, z)`.
    /// [APPROX-PHYSICS-01]: Defaults to 0.6 for standard solid blocks.
    fn friction_at(&self, _x: i32, _y: i32, _z: i32) -> f64 {
        0.6
    }
}

/// Player input state for a 20 Hz tick.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PlayerInput {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub sprint: bool,
    /// Viewer yaw in radians (phi).
    pub yaw: f64,
}

/// Player physics dynamic state.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PlayerPhysicsState {
    /// Foot position in world space.
    pub pos: DVec3,
    /// Foot position at the start of the current tick ([APPROX-VIEW-01]: captured
    /// on the first line of every tick so the renderer can interpolate the camera
    /// between 20 Hz ticks — vanilla `partialTicks` semantics).
    pub prev_pos: DVec3,
    /// Velocity vector (blocks/tick).
    pub vel: DVec3,
    /// Ground latch (recomputed every move).
    pub on_ground: bool,
    /// Sprinting flag.
    pub is_sprinting: bool,
}

impl PlayerPhysicsState {
    pub fn new(pos: DVec3) -> Self {
        Self {
            pos,
            prev_pos: pos,
            vel: DVec3::ZERO,
            on_ground: false,
            is_sprinting: false,
        }
    }

    /// Camera eye position at this exact tick boundary: `pos + (0, 1.62, 0)`.
    #[inline(always)]
    pub fn eye_pos(&self) -> DVec3 {
        DVec3::new(self.pos.x, self.pos.y + EYE_HEIGHT, self.pos.z)
    }

    /// [APPROX-VIEW-01] Camera eye interpolated between the previous tick boundary
    /// and the current one: `lerp(prev_pos, pos, alpha) + (0, 1.62, 0)`. `alpha`
    /// clamps to [0, 1] (viewer feeds `tick_accumulator / 0.05` each render frame).
    #[inline(always)]
    pub fn interpolated_eye_pos(&self, alpha: f64) -> DVec3 {
        let alpha = alpha.clamp(0.0, 1.0);
        self.prev_pos.lerp(self.pos, alpha) + DVec3::new(0.0, EYE_HEIGHT, 0.0)
    }

    /// Player bounding box.
    #[inline(always)]
    pub fn aabb(&self) -> Aabb {
        Aabb::player_box(self.pos)
    }
}

/// Collects solid voxel AABBs that overlap `broadphase_box`.
pub fn collect_solid_colliders(
    world: &impl VoxelWorldCollision,
    broadphase_box: &Aabb,
) -> Vec<Aabb> {
    let min_bx = (broadphase_box.min_x - COLLISION_EPSILON).floor() as i32;
    let max_bx = (broadphase_box.max_x + COLLISION_EPSILON).floor() as i32;
    let min_by = (broadphase_box.min_y - COLLISION_EPSILON).floor() as i32;
    let max_by = (broadphase_box.max_y + COLLISION_EPSILON).floor() as i32;
    let min_bz = (broadphase_box.min_z - COLLISION_EPSILON).floor() as i32;
    let max_bz = (broadphase_box.max_z + COLLISION_EPSILON).floor() as i32;

    let mut colliders = Vec::new();
    for by in min_by..=max_by {
        for bz in min_bz..=max_bz {
            for bx in min_bx..=max_bx {
                if world.is_solid_at(bx, by, bz) {
                    colliders.push(Aabb::new(
                        bx as f64,
                        by as f64,
                        bz as f64,
                        (bx + 1) as f64,
                        (by + 1) as f64,
                        (bz + 1) as f64,
                    ));
                }
            }
        }
    }
    colliders
}

/// Sweeps an AABB against a collection of obstacle boxes with dynamic axis ordering.
/// Resolution order is Y first, then larger horizontal axis, then smaller horizontal axis.
/// (`Direction.java:48-49, 389`, `Shapes.java:222-232`).
pub fn collide_with_shapes(movement: DVec3, mut aabb: Aabb, colliders: &[Aabb]) -> DVec3 {
    if colliders.is_empty() {
        return movement;
    }

    let mut resolved = DVec3::ZERO;

    // 1. Y Axis always first
    if movement.y != 0.0 {
        let mut dy = movement.y;
        for c in colliders {
            if dy.abs() < COLLISION_EPSILON {
                dy = 0.0;
                break;
            }
            dy = aabb.collide_y(c, dy);
        }
        if dy.abs() < COLLISION_EPSILON {
            dy = 0.0;
        }
        resolved.y = dy;
        aabb = aabb.move_by(DVec3::new(0.0, dy, 0.0));
    }

    // 2. Horizontal axes: |dx| < |dz| -> Z then X; else X then Z
    let (first_is_x, first_delta, second_delta) = if movement.x.abs() < movement.z.abs() {
        (false, movement.z, movement.x)
    } else {
        (true, movement.x, movement.z)
    };

    // First horizontal axis
    if first_delta != 0.0 {
        let mut d = first_delta;
        for c in colliders {
            if d.abs() < COLLISION_EPSILON {
                d = 0.0;
                break;
            }
            d = if first_is_x {
                aabb.collide_x(c, d)
            } else {
                aabb.collide_z(c, d)
            };
        }
        if d.abs() < COLLISION_EPSILON {
            d = 0.0;
        }
        if first_is_x {
            resolved.x = d;
            aabb = aabb.move_by(DVec3::new(d, 0.0, 0.0));
        } else {
            resolved.z = d;
            aabb = aabb.move_by(DVec3::new(0.0, 0.0, d));
        }
    }

    // Second horizontal axis
    if second_delta != 0.0 {
        let mut d = second_delta;
        for c in colliders {
            if d.abs() < COLLISION_EPSILON {
                d = 0.0;
                break;
            }
            d = if first_is_x {
                aabb.collide_z(c, d)
            } else {
                aabb.collide_x(c, d)
            };
        }
        if d.abs() < COLLISION_EPSILON {
            d = 0.0;
        }
        if first_is_x {
            resolved.z = d;
        } else {
            resolved.x = d;
        }
    }

    resolved
}

/// Full movement resolution including step-up mechanics (`Entity.java:1188-1245`).
pub fn move_and_collide(
    aabb: Aabb,
    movement: DVec3,
    was_on_ground: bool,
    colliders: &[Aabb],
) -> (DVec3, bool, bool, bool) {
    let movement_step = collide_with_shapes(movement, aabb, colliders);

    let x_collision = (movement.x - movement_step.x).abs() > COLLISION_EPSILON;
    let y_collision = (movement.y - movement_step.y).abs() > COLLISION_EPSILON;
    let z_collision = (movement.z - movement_step.z).abs() > COLLISION_EPSILON;
    let on_ground_after_collision = y_collision && movement.y < 0.0;

    // Step-up evaluation: only when (landed-this-move || onGround) && horizontal collision
    if MAX_UP_STEP > 0.0
        && (on_ground_after_collision || was_on_ground)
        && (x_collision || z_collision)
    {
        let grounded_aabb = if on_ground_after_collision {
            aabb.move_by(DVec3::new(0.0, movement_step.y, 0.0))
        } else {
            aabb
        };

        let mut step_up_aabb =
            grounded_aabb.expand_towards(DVec3::new(movement.x, MAX_UP_STEP, movement.z));
        if !on_ground_after_collision {
            step_up_aabb = step_up_aabb.expand_towards(DVec3::new(0.0, -STEP_UP_NUDGE, 0.0));
        }

        // Collect candidate step heights: collider top faces relative to grounded_aabb.min_y
        let mut candidates: Vec<f64> = Vec::new();
        let step_height_to_skip = movement_step.y;
        for c in colliders {
            if !c.intersects(&step_up_aabb) {
                continue;
            }
            let rel_top = c.max_y - grounded_aabb.min_y;
            if rel_top > 0.0
                && (rel_top - step_height_to_skip).abs() > COLLISION_EPSILON
                && rel_top <= MAX_UP_STEP
            {
                if !candidates
                    .iter()
                    .any(|&val| (val - rel_top).abs() < COLLISION_EPSILON)
                {
                    candidates.push(rel_top);
                }
            }
        }
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for candidate_h in candidates {
            let step_from_ground = collide_with_shapes(
                DVec3::new(movement.x, candidate_h, movement.z),
                grounded_aabb,
                colliders,
            );
            let step_horiz_sq =
                step_from_ground.x * step_from_ground.x + step_from_ground.z * step_from_ground.z;
            let step_base_sq =
                movement_step.x * movement_step.x + movement_step.z * movement_step.z;

            // Strictly improves horizontal progress
            if step_horiz_sq > step_base_sq {
                let distance_to_ground = aabb.min_y - grounded_aabb.min_y;
                let final_movement = step_from_ground - DVec3::new(0.0, distance_to_ground, 0.0);

                let x_col = (movement.x - final_movement.x).abs() > COLLISION_EPSILON;
                let y_col = (movement.y - final_movement.y).abs() > COLLISION_EPSILON;
                let z_col = (movement.z - final_movement.z).abs() > COLLISION_EPSILON;
                return (final_movement, x_col, y_col, z_col);
            }
        }
    }

    (movement_step, x_collision, y_collision, z_collision)
}

/// Executes a single discrete 20 Hz simulation tick (§1 of SPECS.md).
pub fn tick_player(
    state: &mut PlayerPhysicsState,
    input: &PlayerInput,
    world: &impl VoxelWorldCollision,
) {
    // [APPROX-VIEW-01] Capture the pre-tick position FIRST (renderer interpolates
    // the camera between this and the end-of-tick position).
    state.prev_pos = state.pos;

    // --- 1. aiStep prologue: negligible-velocity zeroing ---
    // Player horizontal: horizontalDistanceSqr < 9.0E-6 -> vx = vz = 0
    if state.vel.x * state.vel.x + state.vel.z * state.vel.z < PLAYER_NEGLIGIBLE_HORIZ_SQ {
        state.vel.x = 0.0;
        state.vel.z = 0.0;
    }
    // Vertical (all entities): |vy| < 0.003 -> vy = 0
    if state.vel.y.abs() < NEGLIGIBLE_VERT {
        state.vel.y = 0.0;
    }

    state.is_sprinting = input.sprint;

    // --- 2. Jump handling in aiStep before travel ---
    // Gated by onGround; Space held repeats on landing
    let was_on_ground = state.on_ground;
    if was_on_ground && input.jump {
        state.vel.y = state.vel.y.max(JUMP_IMPULSE);
        if state.is_sprinting {
            // Sprint-jump horizontal impulse: +0.2 * forward
            let (yaw_sin, yaw_cos) = input.yaw.sin_cos();
            let forward = DVec3::new(yaw_sin, 0.0, yaw_cos);
            state.vel += SPRINT_JUMP_BOOST * forward;
        }
    }

    // --- 3. travel: moveRelative input acceleration ---
    // Ground friction probe at y - 0.500001
    let probe_bx = state.pos.x.floor() as i32;
    let probe_by = (state.pos.y - GROUND_PROBE_OFFSET).floor() as i32;
    let probe_bz = state.pos.z.floor() as i32;
    let mu = world.friction_at(probe_bx, probe_by, probe_bz);

    // Accel factor: uses pre-move was_on_ground latch
    let base_speed = if state.is_sprinting {
        SPRINT_SPEED
    } else {
        WALK_SPEED
    };

    let accel = if was_on_ground {
        if mu > 0.6 {
            base_speed * (GROUND_ACCEL_CONSTANT / (mu * mu * mu))
        } else {
            base_speed
        }
    } else if state.is_sprinting {
        AIR_STEERING_SPRINT
    } else {
        AIR_STEERING_WALK
    };

    // Input matrix in viewer convention (SPECS line 58):
    // s = +1 on A, -1 on D; f = +1 on W, -1 on S
    let mut s = 0.0f64;
    if input.left {
        s += 1.0;
    }
    if input.right {
        s -= 1.0;
    }

    let mut f = 0.0f64;
    if input.forward {
        f += 1.0;
    }
    if input.backward {
        f -= 1.0;
    }

    let len_sq = s * s + f * f;
    let (norm_s, norm_f) = if len_sq < COLLISION_EPSILON {
        (0.0, 0.0)
    } else if len_sq > 1.0 {
        let inv = 1.0 / len_sq.sqrt();
        (s * inv, f * inv)
    } else {
        (s, f)
    };

    let (yaw_sin, yaw_cos) = input.yaw.sin_cos();
    let move_dir =
        norm_s * DVec3::new(yaw_cos, 0.0, -yaw_sin) + norm_f * DVec3::new(yaw_sin, 0.0, yaw_cos);

    state.vel += move_dir * accel;

    // --- 4. move(): resolves AABB per-axis against solid blocks ---
    let current_box = state.aabb();
    let broadphase_box = current_box
        .expand_towards(state.vel)
        .expand_towards(DVec3::new(0.0, MAX_UP_STEP, 0.0));
    let colliders = collect_solid_colliders(world, &broadphase_box);

    let (resolved_delta, x_col, y_col, z_col) =
        move_and_collide(current_box, state.vel, was_on_ground, &colliders);

    state.pos += resolved_delta;

    // Ground latch recomputed: onGround = yCollision && requested dy < 0
    state.on_ground = y_col && state.vel.y < 0.0;

    // Zero colliding axes (player restitution is 0.0)
    if x_col {
        state.vel.x = 0.0;
    }
    if y_col {
        state.vel.y = 0.0;
    }
    if z_col {
        state.vel.z = 0.0;
    }

    // --- 5. Post-move (travelInAir): horizontal drag & gravity ---
    // Jump tick keeps ground accel + mu * 0.91 drag because was_on_ground was true at travel start
    let horiz_drag = if was_on_ground {
        mu * HORIZONTAL_DRAG_AIR
    } else {
        HORIZONTAL_DRAG_AIR
    };

    state.vel.x *= horiz_drag;
    state.vel.z *= horiz_drag;

    // Vertical: vy' = (vy - 0.08) * 0.98
    state.vel.y = (state.vel.y - GRAVITY) * VERTICAL_DRAG;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EmptyWorld;
    impl VoxelWorldCollision for EmptyWorld {
        fn is_solid_at(&self, _x: i32, _y: i32, _z: i32) -> bool {
            false
        }
    }

    struct FlatWorld {
        floor_y: i32,
        friction: f64,
    }
    impl FlatWorld {
        fn new(floor_y: i32) -> Self {
            Self {
                floor_y,
                friction: 0.6,
            }
        }
        fn with_friction(floor_y: i32, friction: f64) -> Self {
            Self { floor_y, friction }
        }
    }
    impl VoxelWorldCollision for FlatWorld {
        fn is_solid_at(&self, _x: i32, y: i32, _z: i32) -> bool {
            y <= self.floor_y
        }
        fn friction_at(&self, _x: i32, _y: i32, _z: i32) -> f64 {
            self.friction
        }
    }

    struct ObstacleWorld {
        solids: Vec<(i32, i32, i32)>,
    }
    impl VoxelWorldCollision for ObstacleWorld {
        fn is_solid_at(&self, x: i32, y: i32, z: i32) -> bool {
            self.solids.contains(&(x, y, z))
        }
    }

    #[test]
    fn test_player_dimensions_and_bounding_box() {
        assert_eq!(PLAYER_WIDTH, 0.6);
        assert_eq!(PLAYER_HEIGHT, 1.8);
        assert_eq!(EYE_HEIGHT, 1.62);
        assert_eq!(MAX_UP_STEP, 0.6);

        let pos = DVec3::new(10.0, 5.0, 20.0);
        let b = Aabb::player_box(pos);
        assert!((b.min_x - 9.7).abs() < 1e-9);
        assert!((b.max_x - 10.3).abs() < 1e-9);
        assert!((b.min_y - 5.0).abs() < 1e-9);
        assert!((b.max_y - 6.8).abs() < 1e-9);
        assert!((b.min_z - 19.7).abs() < 1e-9);
        assert!((b.max_z - 20.3).abs() < 1e-9);

        let state = PlayerPhysicsState::new(pos);
        assert_eq!(state.eye_pos(), DVec3::new(10.0, 6.62, 20.0));
    }

    #[test]
    fn test_input_vector_deadzone_and_normalize() {
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 10.0, 0.0));
        let world = EmptyWorld;

        // 1. Zero input
        let input_zero = PlayerInput::default();
        tick_player(&mut state, &input_zero, &world);
        assert_eq!(state.vel.x, 0.0);
        assert_eq!(state.vel.z, 0.0);

        // 2. Forward W only (phi = 0) -> +Z
        let mut state_w = PlayerPhysicsState::new(DVec3::new(0.0, 10.0, 0.0));
        let input_w = PlayerInput {
            forward: true,
            yaw: 0.0,
            ..Default::default()
        };
        tick_player(&mut state_w, &input_w, &world);
        // Air accel: 0.02, post-move air drag: 0.91 -> 0.02 * 0.91 = 0.0182
        assert!((state_w.vel.z - 0.0182).abs() < 1e-6);
        assert_eq!(state_w.vel.x, 0.0);

        // 3. Left A only (phi = 0) -> +X (A = +1-LEFT pin)
        let mut state_a = PlayerPhysicsState::new(DVec3::new(0.0, 10.0, 0.0));
        let input_a = PlayerInput {
            left: true,
            yaw: 0.0,
            ..Default::default()
        };
        tick_player(&mut state_a, &input_a, &world);
        assert!((state_a.vel.x - 0.0182).abs() < 1e-6);
        assert_eq!(state_a.vel.z, 0.0);

        // 4. Diagonal W + A: normalized length = 1.0
        let mut state_wa = PlayerPhysicsState::new(DVec3::new(0.0, 10.0, 0.0));
        let input_wa = PlayerInput {
            forward: true,
            left: true,
            yaw: 0.0,
            ..Default::default()
        };
        tick_player(&mut state_wa, &input_wa, &world);
        let horiz_speed =
            (state_wa.vel.x * state_wa.vel.x + state_wa.vel.z * state_wa.vel.z).sqrt();
        assert!((horiz_speed - 0.0182).abs() < 1e-6);
    }

    #[test]
    fn test_ground_acceleration_and_friction_math() {
        let world_default = FlatWorld::new(0);
        let world_ice = FlatWorld::with_friction(0, 0.98);

        // Standard terrain mu = 0.6: factor is 1.0 (exact float 0.21600002 / 0.6^3 = 1.0000001)
        let mut state_walk = PlayerPhysicsState::new(DVec3::new(0.0, 1.0, 0.0));
        state_walk.on_ground = true;
        let input_walk = PlayerInput {
            forward: true,
            yaw: 0.0,
            ..Default::default()
        };
        tick_player(&mut state_walk, &input_walk, &world_default);
        // Pre-move accel = 0.1 * 1.0 = 0.1; post-move drag = 0.6 * 0.91 = 0.546
        assert!((state_walk.vel.z - 0.1 * 0.546).abs() < 1e-6);

        // Sprinting on mu = 0.6: base 0.13
        let mut state_sprint = PlayerPhysicsState::new(DVec3::new(0.0, 1.0, 0.0));
        state_sprint.on_ground = true;
        let input_sprint = PlayerInput {
            forward: true,
            sprint: true,
            yaw: 0.0,
            ..Default::default()
        };
        tick_player(&mut state_sprint, &input_sprint, &world_default);
        assert!((state_sprint.vel.z - 0.13 * 0.546).abs() < 1e-6);

        // Ice mu = 0.98: accel factor = 0.21600002 / (0.98^3) = 0.2294975
        let mut state_ice = PlayerPhysicsState::new(DVec3::new(0.0, 1.0, 0.0));
        state_ice.on_ground = true;
        tick_player(&mut state_ice, &input_walk, &world_ice);
        let expected_ice_accel = 0.1 * (0.21600002 / (0.98 * 0.98 * 0.98));
        let expected_ice_drag = 0.98 * 0.91;
        assert!((state_ice.vel.z - expected_ice_accel * expected_ice_drag).abs() < 1e-6);
    }

    #[test]
    fn test_jump_tick_sequencing() {
        // Reviewer requirement: "t1 pos.y += 0.42 / vy = 0.3332"
        let world = FlatWorld::new(0);
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 1.0, 0.0));
        state.on_ground = true;

        let input_jump = PlayerInput {
            jump: true,
            ..Default::default()
        };

        tick_player(&mut state, &input_jump, &world);

        // In tick 1:
        // Jump impulse vy = 0.42
        // Position moves up by 0.42 -> pos.y = 1.42 (pos.y += 0.42)
        // Post-move vertical update: (0.42 - 0.08) * 0.98 = 0.34 * 0.98 = 0.3332
        assert!(
            (state.pos.y - 1.42).abs() < 1e-9,
            "t1 pos.y must increase by exactly 0.42 (got {})",
            state.pos.y
        );
        assert!(
            (state.vel.y - 0.3332).abs() < 1e-9,
            "t1 vy must be exactly 0.3332 (got {})",
            state.vel.y
        );
    }

    #[test]
    fn test_sprint_jump_boost() {
        let world = FlatWorld::new(0);
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 1.0, 0.0));
        state.on_ground = true;

        let input_sprint_jump = PlayerInput {
            forward: true,
            jump: true,
            sprint: true,
            yaw: 0.0, // forward is +Z
            ..Default::default()
        };

        tick_player(&mut state, &input_sprint_jump, &world);

        // vy = 0.3332
        assert!((state.vel.y - 0.3332).abs() < 1e-9);
        // Forward speed includes sprint accel (0.13) + sprint jump boost (+0.2)
        // Total before drag = 0.13 + 0.20 = 0.33
        // Ground drag on jump tick (started on ground) = 0.6 * 0.91 = 0.546
        // vz = 0.33 * 0.546 = 0.18018
        assert!((state.vel.z - 0.33 * 0.546).abs() < 1e-6);
    }

    #[test]
    fn test_freefall_sequencing() {
        // Reviewer requirement: "freefall t1 dy=0 → t2 dy=−0.0784"
        let world = EmptyWorld;
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 10.0, 0.0));
        state.on_ground = false;

        let input = PlayerInput::default();

        // Tick 1
        let pos_before_t1 = state.pos.y;
        tick_player(&mut state, &input, &world);
        let dy_t1 = state.pos.y - pos_before_t1;
        assert_eq!(dy_t1, 0.0, "t1 dy must be 0");
        assert!(
            (state.vel.y - (-0.0784)).abs() < 1e-9,
            "t1 vy must be -0.0784 (got {})",
            state.vel.y
        );

        // Tick 2
        let pos_before_t2 = state.pos.y;
        tick_player(&mut state, &input, &world);
        let dy_t2 = state.pos.y - pos_before_t2;
        assert!(
            (dy_t2 - (-0.0784)).abs() < 1e-9,
            "t2 dy must be -0.0784 (got {})",
            dy_t2
        );
        // t2 vy = (-0.0784 - 0.08) * 0.98 = -0.1584 * 0.98 = -0.155232
        assert!((state.vel.y - (-0.155232)).abs() < 1e-9);
    }

    #[test]
    fn test_terminal_fall_velocity() {
        let world = EmptyWorld;
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 10000.0, 0.0));

        let input = PlayerInput::default();
        // 600 ticks gives 0.98^600 < 5e-6, converging to terminal velocity -3.92
        for _ in 0..600 {
            tick_player(&mut state, &input, &world);
        }

        // Closed form: -0.08 * 0.98 / (1.0 - 0.98) = -3.92
        assert!(
            (state.vel.y - TERMINAL_VELOCITY).abs() < 1e-4,
            "vel.y must converge to -3.92 (got {})",
            state.vel.y
        );

        // Also verify that steady state -3.92 is an exact fixed point:
        let mut fixed_state = PlayerPhysicsState::new(DVec3::new(0.0, 1000.0, 0.0));
        fixed_state.vel.y = TERMINAL_VELOCITY;
        tick_player(&mut fixed_state, &input, &world);
        assert!(
            (fixed_state.vel.y - TERMINAL_VELOCITY).abs() < 1e-9,
            "terminal velocity must be an exact fixed point"
        );
    }

    #[test]
    fn test_dynamic_axis_sweep_order() {
        // Obstacle at X in [1.0, 2.0], Y in [0.0, 2.0], Z in [1.0, 2.0]
        let colliders = vec![Aabb::new(1.0, 0.0, 1.0, 2.0, 2.0, 2.0)];
        // AABB at X in [0.0, 0.8], Y in [0.0, 1.8], Z in [0.5, 0.9]
        // Initially, X and Z do not overlap the obstacle.
        let aabb = Aabb::new(0.0, 0.0, 0.5, 0.8, 1.8, 0.9);

        // 1. |dx| >= |dz|: YXZ order (X sweeps first, then Z)
        // dx=0.5 >= dz=0.3 -> X sweeps first (no Z overlap, so dx achieves full 0.5).
        // Then Z sweeps, now overlapping in X (X in [0.5, 1.3]), so Z hits obstacle at 1.0 (dz capped at 1.0 - 0.9 = 0.1).
        let mov_x_first = DVec3::new(0.5, 0.0, 0.3);
        let res_x_first = collide_with_shapes(mov_x_first, aabb, &colliders);
        assert!(
            (res_x_first.x - 0.5).abs() < COLLISION_EPSILON,
            "X sweeps first and achieves full dx=0.5"
        );
        assert!(
            (res_x_first.z - 0.1).abs() < COLLISION_EPSILON,
            "Z sweeps second and is capped at 0.1"
        );

        // 2. |dx| < |dz|: YZX order (Z sweeps first, then X)
        // dx=0.3 < dz=0.5 -> Z sweeps first (no X overlap, so dz achieves full 0.5).
        // Then X sweeps, now overlapping in Z (Z in [1.0, 1.4]), so X hits obstacle at 1.0 (dx capped at 1.0 - 0.8 = 0.2).
        let mov_z_first = DVec3::new(0.3, 0.0, 0.5);
        let res_z_first = collide_with_shapes(mov_z_first, aabb, &colliders);
        assert!(
            (res_z_first.z - 0.5).abs() < COLLISION_EPSILON,
            "Z sweeps first and achieves full dz=0.5"
        );
        assert!(
            (res_z_first.x - 0.2).abs() < COLLISION_EPSILON,
            "X sweeps second and is capped at 0.2"
        );
    }

    #[test]
    fn test_collision_restitution_and_on_ground() {
        let world = FlatWorld::new(0); // solid floor at y <= 0, top of floor is y = 1.0
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 1.5, 0.0));
        state.vel = DVec3::new(0.0, -0.6, 0.0);

        let input = PlayerInput::default();
        tick_player(&mut state, &input, &world);

        // Player stops on floor at y = 1.0
        assert!((state.pos.y - 1.0).abs() < 1e-6);
        assert!(state.on_ground, "on_ground must latch true after landing");
        // Colliding axis Y zeroed (restitution = 0.0), then post-move gravity applies: (0 - 0.08) * 0.98 = -0.0784
        assert!((state.vel.y - (-0.0784)).abs() < 1e-6);
    }

    #[test]
    fn test_wall_sliding() {
        // Wall along X=1
        let world = ObstacleWorld {
            solids: vec![(1, 0, 0), (1, 0, 1), (1, 0, 2)],
        };
        let mut state = PlayerPhysicsState::new(DVec3::new(0.5, 0.0, 0.0));
        state.on_ground = true;

        // Player moves diagonally into wall (+X, +Z)
        state.vel = DVec3::new(0.3, 0.0, 0.3);

        let input = PlayerInput::default();
        tick_player(&mut state, &input, &world);

        // X collision stopped by wall (player max_x reaches 1.0 -> pos.x reaches 0.7)
        assert!(state.pos.x <= 0.7 + 1e-6);
        // Z continues sliding freely
        assert!(state.pos.z > 0.1);
    }

    #[test]
    fn test_step_up_height() {
        // Slab obstacle of height 0.5 at (1, 0, 0): top is y = 0.5
        let world = ObstacleWorld {
            solids: vec![(1, 0, 0)], // 1.0 high block
        };

        let mut state = PlayerPhysicsState::new(DVec3::new(0.5, 0.0, 0.0));
        state.on_ground = true;
        state.vel = DVec3::new(0.3, 0.0, 0.0);

        let input = PlayerInput::default();
        tick_player(&mut state, &input, &world);

        // 1.0 block exceeds MAX_UP_STEP (0.6), so player is blocked horizontally
        assert!(state.pos.x <= 0.7 + 1e-6);
    }

    #[test]
    fn test_step_up_acceptance_climb() {
        // §1 ACCEPT path (review pass 5B-R2): a fractional-height collider whose top
        // sits inside the 0.6 step ceiling must be climbed, not blocked. tick_player
        // tests can only cover rejection (colliders snap to unit cubes), so this
        // drives move_and_collide directly with a half-height slab (top y = 0.5).
        let colliders = vec![Aabb::new(1.0, 0.0, 0.0, 2.0, 0.5, 1.0)];
        let aabb = Aabb::player_box(DVec3::new(0.5, 0.0, 0.0));
        let (final_mv, x_col, y_col, _z_col) =
            move_and_collide(aabb, DVec3::new(0.3, 0.0, 0.0), true, &colliders);

        // Non-step path caps x at 1.0 - 0.8 = 0.2 (slab overlaps at feet height); the
        // candidate replay climbs y = 0.5 (<= MAX_UP_STEP) where the slab no longer
        // overlaps, so the full dx = 0.3 strictly improves horizontal progress.
        assert!(
            (final_mv.x - 0.3).abs() < 1e-9,
            "must achieve full dx after stepping (got {})",
            final_mv.x
        );
        assert!(
            (final_mv.y - 0.5).abs() < 1e-9,
            "must climb exactly the candidate height 0.5 (got {})",
            final_mv.y
        );
        assert_eq!(final_mv.z, 0.0);
        // Accepted-step flags: horizontal freed, y counts as displaced (0 -> 0.5).
        assert!(!x_col, "stepping must not collide horizontally");
        assert!(y_col, "climb registers as a y displacement");

        // Settle next tick: gravity pulls the box back onto the slab top and latches.
        // (Vanilla quirk, pinned by design: onGround stays FALSE on the step tick
        // itself because requested dy == 0; it latches on the following settle.)
        let grounded_box = Aabb::player_box(DVec3::new(0.8, 0.5, 0.0));
        let (final_mv2, x_col2, y_col2, _z_col2) = move_and_collide(
            grounded_box,
            DVec3::new(0.0, (0.0 - GRAVITY) * VERTICAL_DRAG, 0.0),
            true,
            &colliders,
        );
        assert!(
            final_mv2.y.abs() < 1e-9,
            "must stay exactly on the slab top (got {})",
            final_mv2.y
        );
        assert!(y_col2, "landing must register as a y collision");
        assert!(!x_col2 && !_z_col2, "no horizontal movement while settling");
    }

    #[test]
    fn test_negligible_velocity_zeroing() {
        let world = EmptyWorld;
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 10.0, 0.0));

        // Horizontal velocity below sqrt(9.0E-6) = 0.003
        state.vel = DVec3::new(0.001, 0.002, 0.001);

        let input = PlayerInput::default();
        tick_player(&mut state, &input, &world);

        // vx and vz zeroed in prologue before post-move drag
        assert_eq!(state.vel.x, 0.0);
        assert_eq!(state.vel.z, 0.0);
    }

    #[test]
    fn test_prev_pos_and_interpolated_eye() {
        // [APPROX-VIEW-01]: prev_pos is captured at the first line of every tick and
        // the renderer interpolates eye = lerp(prev_pos, pos, alpha) + (0, 1.62, 0).
        let world = EmptyWorld;
        let mut state = PlayerPhysicsState::new(DVec3::new(0.0, 10.0, 0.0));
        let input = PlayerInput::default();

        tick_player(&mut state, &input, &world); // t1: dy = 0, vy = -0.0784
        tick_player(&mut state, &input, &world); // t2: dy = -0.0784

        // prev_pos must be the position at the START of t2 (== post-t1 pos).
        assert!(
            (state.prev_pos.y - 10.0).abs() < 1e-9,
            "prev_pos must capture the pre-tick foot position (got {})",
            state.prev_pos.y
        );
        assert!(
            (state.pos.y - (10.0 - 0.0784)).abs() < 1e-9,
            "pos must be post-move (got {})",
            state.pos.y
        );

        // Interpolated eye: alpha 0.5 is the exact midpoint + eye height.
        let mid = state.interpolated_eye_pos(0.5);
        let expected_y = 10.0 - 0.0784 * 0.5 + EYE_HEIGHT;
        assert!(
            (mid.y - expected_y).abs() < 1e-9,
            "alpha=0.5 must be the exact midpoint + eye (got {}, want {})",
            mid.y,
            expected_y
        );
        // Clamp semantics: alpha 0 -> previous boundary, oversized alpha -> current.
        assert_eq!(
            state.interpolated_eye_pos(0.0),
            DVec3::new(0.0, 10.0 + EYE_HEIGHT, 0.0)
        );
        assert!(
            (state.interpolated_eye_pos(1.5).y - (10.0 - 0.0784 + EYE_HEIGHT)).abs() < 1e-9,
            "alpha must clamp to 1.0"
        );
    }
}
