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
    ids: u32,
    max: vec3f,
    flags: u32,
}

struct ComputePassUniforms {
    position: vec3<f32>,
    fov: f32,
    direction: vec3<f32>,
    aspect: f32,

    width: u32,
    height: u32,

    time: u32,

    // This works without it, but putting here just in case.
    _pad_0: u32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Intersection {
    coord: vec3<f32>,
    normal: vec3<f32>,
    is_hit: bool,
    distance_squared: f32,
    index: u32,
}

@group(0) @binding(0)
var<storage, read> bvh_objects: array<Sphere>;

@group(0) @binding(1)
var<storage, read> bvh_nodes: array<BVHNode>;

@group(1) @binding(0)
var output: texture_storage_2d<rgba16float, write>;

@group(1) @binding(1)
var<uniform> uniforms: ComputePassUniforms;

@group(1) @binding(2)
var<storage, read> random_colors: array<vec4f>;

@group(1) @binding(3)
var<storage, read> random_directions: array<vec3f>;

@group(1) @binding(4) var bgColorTexture : texture_2d<f32>;
@group(1) @binding(5) var bgIrradianceTexture : texture_2d<f32>;

const FLAG_EMPTY: u32 = 0u;
const FLAG_MERGE: u32 = 1u;

const PI: f32 = 3.1415926;

struct StackItem {
    node_id: u32,
    flags: u32,
}

const MAX_ITER: u32 = 256u;
const LIGHT_DIRECTION: vec3f = vec3(0.09901475, 0.09901475, -0.99014753);
var<private> stack: array<StackItem, 32>;

fn aabb_tnear(node_id: u32, ray: Ray) -> f32 {
    // TODO: handle ray.direction == 0.
    let t1_tmp = (bvh_nodes[node_id].min - ray.origin) / ray.direction;
    let t2_tmp = (bvh_nodes[node_id].max - ray.origin) / ray.direction;

    let t1 = min(t1_tmp, t2_tmp);
    let t2 = max(t1_tmp, t2_tmp);

    // The time it takes the ray to enter the box.
    let tnear = max(max(t1.x, t1.y), t1.z);
    // The time it takes the ray to exit the bos.
    let tfar = min(min(t2.x, t2.y), t2.z);

    // tnear < tfar means the ray enters and exits the box.
    // tfar > 0 is to ignore boxes that are behind the ray looking from origin.
    if tnear < tfar && tfar > 0. {
        return tnear;
    } else {
        return 0.;
    }
}

// Function to generate a random color based on a u32 seed
fn randomColor(seed: u32) -> vec3<f32> {
    return random_colors[seed].xyz;
}

fn get_sphere_color(id: u32) -> vec3f {
    return randomColor(id);
}

fn get_color(ray: Ray, i: Intersection) -> vec4f {
    let albedo = get_sphere_color(i.index);
    var irradiance = sample_irradiance(i.normal);
    let color = albedo * irradiance;
    return vec4(color, 1.);
}

fn direction_to_uv(direction: vec3<f32>) -> vec2<f32> {
    let longitude = atan2(direction.y, direction.x);
    let latitude  = asin(direction.z);
    let u = (longitude / (2.0 * PI)) + 0.5;
    let v = 0.5 - (latitude / PI);
    return vec2(u, v);
}

fn sample_irradiance(direction: vec3<f32>) -> vec3<f32> {
    let texSize = textureDimensions(bgIrradianceTexture);
    let texCoord = direction_to_uv(direction);
    let pixelCoord = vec2<i32>(texCoord * vec2<f32>(texSize));
    let texel = textureLoad(bgIrradianceTexture, pixelCoord, 0);
    return texel.rgb;
}

fn sample_background(ray: Ray) -> vec3<f32> {
    let texSize = textureDimensions(bgColorTexture);
    let texCoord = direction_to_uv(ray.direction);
    let pixelCoord = vec2<i32>(texCoord * vec2<f32>(texSize));
    let texel = textureLoad(bgColorTexture, pixelCoord, 0);
    return texel.rgb;
}

fn bvh_color(ray: Ray, pixel: vec2<u32>) -> vec4f {
    let i = bvh_intersect(ray);
    if i.is_hit {
        var color = get_color(ray, i);

        // Relfectivity.
        let new_direction = normalize(reflect(ray.direction, i.normal));
        let new_ray = Ray(i.coord + new_direction * 0.01, new_direction);
        let new_hit = bvh_intersect(new_ray);
        if new_hit.is_hit {
            color += get_color(new_ray, new_hit) * 0.1;
        } else {
            color += vec4(sample_background(new_ray), 1.) * 0.1;
        }

        return vec4(color);
    }

    return vec4(sample_background(ray), 1.);
}

fn stack_push(current_len: u32, node_id: u32, op: u32) -> u32 {
    stack[current_len] = StackItem(node_id, op);
    return current_len + 1u;
}

fn sphere_ray_intersection(id: u32, ray: Ray) -> Intersection {
    let sphere = bvh_objects[id];
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
    let dist = coord - ray.origin;
    return Intersection(coord, normal, true, dot(dist, dist), id);
}

fn merge_intersections(ray: Ray, i1: Intersection, i2: Intersection) -> Intersection {
    if !i1.is_hit {
        return i2;
    }
    if (i1.distance_squared < i2.distance_squared) {
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
        FLAG_EMPTY,
    );

    // Branching:
    // super fast: if only one of them present, don't merge.
    // - this can be done pre-recursion
    // fast (aabs don't intersect):
    // - check closest. If it hits, don't check the second.
    // - ONLY IF IT DOESN'T HIT, check the second.
    // slow (aabs intersect):
    // - check both. Then merge the intersections.

    // However, with a stack it's fucking complicated.
    // Goal: return as early as possible, stop traversal.
    // When can we NOT return:
    // - we are in ANY of the branches that where overlapping.
    // - BOTH need to be searched, result merged.
    // - the flags get only "worse" (poisoned) progressively.

    var iterations = 0u;

    while stack_len > 0u && iterations < MAX_ITER {
        iterations += 1u;
        let idx = stack_len - 1u;
        stack_len -= 1u;

        let op = stack[idx].flags;

        let flag_merge = (op & FLAG_MERGE) == FLAG_MERGE;

        if intersection.is_hit {
            if op == FLAG_EMPTY {
                return intersection;
            }
        }

        let node_id = stack[idx].node_id;
        let is_leaf = (bvh_nodes[node_id].flags & FLAG_IS_LEAF) == FLAG_IS_LEAF;
        let overlaps = (bvh_nodes[node_id].flags & FLAG_OVERLAPS) == FLAG_OVERLAPS;

        if is_leaf {
            let i = sphere_ray_intersection(bvh_nodes[node_id].ids, ray);
            if !i.is_hit {
                continue;
            }
            if op == FLAG_EMPTY {
                return i;
            }
            intersection = merge_intersections(ray, intersection, i);
        } else {
            var left = bvh_nodes[node_id].ids;
            var right = left + 1u;

            var left_flags = op;
            var right_flags = op;

            let left_tnear = aabb_tnear(left, ray);
            let right_tnear = aabb_tnear(right, ray);

            // If both are crossing:
            if left_tnear != 0. && right_tnear != 0. {
                if !overlaps {
                    // Encode "if the closest one hits, ignore the second"
                    if (right_tnear < left_tnear) {
                        let tmp = right;
                        right = left;
                        left = tmp;
                    }
                } else {
                    right_flags |= FLAG_MERGE;
                    left_flags |= FLAG_MERGE;
                }

                stack_len = stack_push(stack_len, right, right_flags);
                stack_len = stack_push(stack_len, left, left_flags);
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

    let forward = uniforms.direction;
    let abs_up = vec3f(0., 0., 1.);
    let left = -normalize(cross(uniforms.direction, abs_up));
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

    let color = bvh_color(ray, vec2(x_abs, y_abs));

    textureStore(output, vec2(x_abs, y_abs), color);
}
