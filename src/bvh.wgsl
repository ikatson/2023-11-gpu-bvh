// Align: 16, sizeof: 32
struct AABB {
    min: vec3f,
    max: vec3f,
}

struct Sphere {
    center: vec3f,
    radius: f32,
}

const FLAG_IS_LEAF: u32 = 1u;
const FLAG_OVERLAPS: u32 = 2u;

struct BVHNode {
    min: vec3f,
    id1: u32,
    max: vec3f,
    id2: u32,
    flags: u32,

    // TODO: why the hell is this needed? Doesn't work without it.
    _pad_0: u32,
    _pad_1: u32,
    _pad_2: u32,
}

struct ComputePassUniforms {
    position: vec3<f32>,
    fov: f32,
    direction: vec3<f32>,
    aspect: f32,

    width: u32,
    height: u32,

    // This works without it, but putting here just in case.
    _pad_0: u32,
    _pad_1: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Intersection {
    coord: vec3<f32>,
    normal: vec3<f32>,
    is_hit: bool,
}

@group(0) @binding(0)
var<storage, read> bvh_objects: array<Sphere>;

@group(0) @binding(1)
var<storage, read> bvh_nodes: array<BVHNode>;

@group(1) @binding(0)
var output: texture_storage_2d<rgba32float, write>;

@group(1) @binding(1)
var<uniform> uniforms: ComputePassUniforms;

const FLAGS_EMPTY: u32 = 0u;
const FLAG_MERGE: u32 = 4u;
const FLAG_IGNORE_IF_SET: u32 = 4u;

struct StackItem {
    node_id: u32,
    flags: u32,
}

const MAX_ITER: u32 = 128u;
var<private> stack: array<StackItem, 32>;

fn aabb_tnear(node_id: u32, ray: Ray) -> f32 {
    // TODO: handle ray.direction == 0.
    let t1_tmp = (bvh_nodes[node_id].min - ray.origin) / ray.direction;
    let t2_tmp = (bvh_nodes[node_id].max - ray.origin) / ray.direction;

    let t1 = min(t1_tmp, t2_tmp);
    let t2 = max(t1_tmp, t2_tmp);

    let tnear = max(max(t1.x, t1.y), t1.z);
    let tfar = min(min(t2.x, t2.y), t2.z);

    if tnear < tfar {
        return tnear;
    } else {
        return 0.;
    }
}

fn bvh_color(ray: Ray) -> vec4f {
    let i = bvh_intersect(ray);
    if i.is_hit {
        return vec4(abs(i.normal), 1.);
    }
    return vec4(0.);
}

fn stack_push(current_len: u32, node_id: u32, op: u32) -> u32 {
    stack[current_len] = StackItem(node_id, op);
    return current_len + 1u;
}

fn sphere_ray_intersection(sphere: Sphere, ray: Ray) -> Intersection {
    let oc = vec3(
        ray.origin.x - sphere.center.x,
        ray.origin.y - sphere.center.y,
        ray.origin.z - sphere.center.z,
    );

    let a: f32 = dot(ray.direction, ray.direction);
    let b: f32 = 2.0 * dot(oc, ray.direction);
    let c: f32 = dot(oc, oc) - sphere.radius * sphere.radius;

    let discriminant: f32 = b * b - 4.0 * a * c;

    if discriminant < 0. {
        // No intersection
        return Intersection();
    }

    let sqrt_discriminant = sqrt(discriminant);
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);

    // Choose the smaller positive root
    var t = t2;
    if t1 >= 0.0 && t1 < t2 {
        t = t1;
    }

    if t < 0.0 {
        return Intersection();
    }
    // Intersection point
    let coord = vec3(
        ray.origin.x + t * ray.direction.x,
        ray.origin.y + t * ray.direction.y,
        ray.origin.z + t * ray.direction.z,
    );
    let normal = normalize(coord - sphere.center);
    return Intersection(coord, normal, true);
}

fn merge_intersections(ray: Ray, i1: Intersection, i2: Intersection) -> Intersection {
    if !i1.is_hit {
        return i2;
    }
    if !i2.is_hit {
        return i1;
    }
    let i1_d = dot(i1.coord - ray.origin, i1.coord - ray.origin);
    let i2_d = dot(i2.coord - ray.origin, i2.coord - ray.origin);
    if (i1_d < i2_d) {
        return i1;
    }
    return i2;
}

fn aabb_intersects_other(left_id: u32, right_id: u32) -> bool {
    // TODO
    return false;
}

fn bvh_intersect(ray: Ray) -> Intersection {
    var stack_len: u32 = 1u;
    var intersection = Intersection();

    let root_tnear = aabb_tnear(0u, ray);
    if root_tnear == 0. {
        return intersection;
    }
    stack[0] = StackItem(
        0u,
        FLAGS_EMPTY,
    );

    // Branching:
    // super fast: if only one of them present, don't merge.
    // - this can be done pre-recursion
    // fast (aabs don't intersect):
    // - check closest. If it hits, don't check the second.
    // - ONLY IF IT DOESN'T HIT, check the second.
    // slow (aabs intersect):
    // - check both. Then merge the intersections.

    var iterations = 0u;

    while stack_len > 0u && iterations < MAX_ITER {
        iterations += 1u;
        let idx = stack_len - 1u;
        stack_len -= 1u;
        let node_id = stack[idx].node_id;
        let op = stack[idx].flags;

        let is_leaf = (bvh_nodes[node_id].flags & FLAG_IS_LEAF) == FLAG_IS_LEAF;
        let overlaps = (bvh_nodes[node_id].flags & FLAG_OVERLAPS) == FLAG_OVERLAPS;

        if is_leaf {
            if intersection.is_hit && ((op & FLAG_IGNORE_IF_SET) == FLAG_IGNORE_IF_SET) {
                continue;
            }
            let i = sphere_ray_intersection(bvh_objects[bvh_nodes[node_id].id1], ray);
            if intersection.is_hit && ((op & FLAG_MERGE) == FLAG_MERGE) {
                intersection = merge_intersections(ray, intersection, i);
            } else {
                intersection = i;
            }
        } else {
            var right = bvh_nodes[node_id].id2;
            var left = bvh_nodes[node_id].id1;

            let left_tnear = aabb_tnear(left, ray);
            let right_tnear = aabb_tnear(right, ray);

            // If both are crossing:
            if left_tnear != 0. && right_tnear != 0. {
                if !overlaps {
                    // Encode "if the closest one hits, ignore the second"

                    // Swap left and right
                    if (right_tnear < left_tnear) {
                        let tmp = left;
                        left = right;
                        right = tmp;
                    }

                    // Now left is closer, more important. So push right first.
                    stack_len = stack_push(stack_len, right, op | FLAG_IGNORE_IF_SET);
                    stack_len = stack_push(stack_len, left, op);
                } else {
                    stack_len = stack_push(stack_len, right, op | FLAG_MERGE);
                    stack_len = stack_push(stack_len, left, op);
                }
            } else {
                // Basic (super fast, only one pushed)
                if left_tnear != 0. {
                    stack_len = stack_push(stack_len, left, op);
                }
                if right_tnear != 0. {
                    stack_len = stack_push(stack_len, right, op);
                }
            }
        }
    }
    return intersection;
}

@compute
@workgroup_size(8, 8, 1)
fn render_through_bvh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_abs: u32 = global_id.x;
    let y_abs: u32 = global_id.y;

    let PI: f32 = 3.1415926;
    let forward = uniforms.direction;
    let left = normalize(cross(uniforms.direction, vec3(0., 1., 0.)));
    let up = normalize(cross(left, forward));

    // fov/2 = hor / dirlen. As dirlen == 1, thus fov/2 = hor
    // as fov is in degrees, we need to convert to radians also, so
    // hor = fov/2 * PI / 180
    let hor = uniforms.fov / 360. * PI;
    let vert = hor / uniforms.aspect;
    let u = f32(x_abs) / f32(uniforms.width) - 0.5;
    let v = f32(y_abs) / f32(uniforms.height) - 0.5;
    let target_point = uniforms.position + uniforms.direction - left * hor * u + up * vert * v;
    let direction = normalize(target_point - uniforms.position);
    let ray = Ray(
        uniforms.position,
        direction,
    );

    let color = bvh_color(ray);

    textureStore(output, vec2(x_abs, y_abs), color);
}