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
    width: u32,
    height: u32,
    camera: PerspectiveCamera,
}

struct PerspectiveCamera {
    position: vec3<f32>,
    direction: vec3<f32>,
    fov: f32,
    aspect: f32,
}

@group(0) @binding(0)
var<storage, read> bvh_nodes: array<BVHNode>;

@group(0) @binding(1)
var<storage, read> bvh_objects: array<Sphere>;

@group(0) @binding(2)
var<uniform> bvh_meta: BVHMeta;

@group(1) @binding(0)
var output: texture_storage_2d<rgba32float, write>;

@group(1) @binding(1)
var<uniform> u: ComputePassUniforms;

@compute
@workgroup_size(8, 8, 1)
fn render_through_bvh(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(output, vec2(5u, 5u), vec4(1., 0., 0., 1.));
}