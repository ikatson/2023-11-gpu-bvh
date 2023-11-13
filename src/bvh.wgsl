// Align: 16, sizeof: 32
struct AABB {
    min: vec3f,
    max: vec3f,
}

struct Sphere {
    center: vec3f,
    radius: vec3f,
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
        return vec4(1.);
    }
    return vec4(0.);

    // return vec4(abs(bvh_nodes[bvh_meta.root].aabb.min), 1.);
}

fn bvh_intersect(ray: Ray) -> Intersection {
    let root = bvh_meta.root;
    let tnear = aabb_tnear(bvh_nodes[root].aabb, ray);
    if tnear == 0. {
        return Intersection(vec3(0.), false);
    }
    return Intersection(vec3(1.), true);
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