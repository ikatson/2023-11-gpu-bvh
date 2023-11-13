// Align: 16, sizeof: 32
struct AABB {
    min: vec3f,
    max: vec3f,
}

struct Sphere {
    center: vec3f,
    radius: f32,
}

struct BVHNode {
    aabb: AABB,
    is_leaf: u32,
    // If leaf, this is sphere id.
    // If branch, id1 is left, id2 is right.
    id1: u32,
    id2: u32,
}

struct BVHMeta {
    root: u32,
}

struct ComputePassUniforms {
    // camera
    position: vec3<f32>,
    direction: vec3<f32>,
    fov: f32,
    aspect: f32,

    width: u32,
    height: u32,
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

@group(0) @binding(2)
var<uniform> bvh_meta: BVHMeta;

@group(1) @binding(0)
var output: texture_storage_2d<rgba32float, write>;

@group(1) @binding(1)
var<uniform> uniforms: ComputePassUniforms;

var<private> stack: array<u32, 32>;

fn aabb_tnear(aabb: AABB, ray: Ray) -> f32 {
    // TODO: handle ray.direction == 0.
    let t1_tmp = (aabb.min - ray.origin) / ray.direction;
    let t2_tmp = (aabb.max - ray.origin) / ray.direction;

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

fn stack_push(current_len: u32, node_id: u32) -> u32 {
    stack[current_len] = node_id;
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

fn bvh_intersect(ray: Ray) -> Intersection {
    var stack_len: u32 = 1u;
    var intersection = Intersection();

    stack[0] = bvh_meta.root;
    while stack_len > 0u {
        let node_id = stack[stack_len - 1u];
        stack_len -= 1u;
        let tnear = aabb_tnear(bvh_nodes[node_id].aabb, ray);
        if tnear == 0. {
            continue;
        }
        if bvh_nodes[node_id].is_leaf == 1u {
            let i = sphere_ray_intersection(bvh_objects[bvh_nodes[node_id].id1], ray);
            intersection = merge_intersections(ray, intersection, i);
        } else {
            stack_len = stack_push(stack_len, bvh_nodes[node_id].id2);
            stack_len = stack_push(stack_len, bvh_nodes[node_id].id1);
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