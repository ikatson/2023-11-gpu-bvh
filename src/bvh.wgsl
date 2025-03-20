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
    distance_squared: f32,
    index: u32,
}

@group(0) @binding(0)
var<storage, read> bvh_objects: array<Sphere>;

@group(0) @binding(1)
var<storage, read> bvh_nodes: array<BVHNode>;

@group(1) @binding(0)
var output: texture_storage_2d<rgba32float, write>;

@group(1) @binding(1)
var<uniform> uniforms: ComputePassUniforms;

@group(1) @binding(2)
var<storage, read> random_colors: array<vec4f>;

@group(1) @binding(3)
var<storage, read> random_directions: array<vec3f>;

const FLAG_EMPTY: u32 = 0u;
const FLAG_MERGE: u32 = 1u;

const PI: f32 = 3.1415926;

struct StackItem {
    node_id: u32,
    flags: u32,
}

const MAX_ITER: u32 = 128u;
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

// UE4Falloff function
fn UE4Falloff(distance: f32, lightRadius: f32) -> f32 {
    let nominator = clamp(1.0 - pow(distance / lightRadius, 4.0), 0.0, 1.0);
    return (nominator * nominator) / (distance * distance + 1.0);
}

// UE4NDF function
fn UE4NDF(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH2 = NdotH * NdotH;

    let num = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

// GeometrySchlickGGX function
fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;

    let num = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

// GeometrySmith function
fn GeometrySmith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let ggx2 = GeometrySchlickGGX(NdotV, roughness);
    let ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// fresnelSchlick function with scalar input
fn fresnelSchlickScalar(HdotV: f32, F0: f32) -> f32 {
    return F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
}

// fresnelSchlick function with vector input
fn fresnelSchlick(albedo: vec3<f32>, metallic: f32, HdotV: f32) -> vec3<f32> {
    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);

    return vec3<f32>(fresnelSchlickScalar(HdotV, F0.r), fresnelSchlickScalar(HdotV, F0.g), fresnelSchlickScalar(HdotV, F0.b));
}

// CookTorranceBRDF function
fn CookTorranceBRDF(
    albedo: vec3<f32>, roughness: f32, metallic: f32,
    V: vec3<f32>, normal: vec3<f32>, L: vec3<f32>, radiance: vec3<f32>
) -> vec3<f32> {
    let H = normalize(V + L);
    let N = normal;

    let NdotL = max(dot(N, L), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    let NDF = UE4NDF(NdotH, roughness);

    let G = GeometrySmith(NdotV, NdotL, roughness);
    let F = fresnelSchlick(albedo, metallic, HdotV);

    var kD = vec3<f32>(1.0) - F;
    kD *= 1.0 - metallic;

    let numerator = NDF * G * F;
    let denominator = 4.0 * NdotV * NdotL;
    let specular = numerator / max(denominator, 0.001);

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// toneMap function
fn toneMap(color: vec3<f32>) -> vec3<f32> {
    var c = color / (color + vec3<f32>(1.0));
    c = pow(color, vec3<f32>(1.0 / 2.2));
    return c;
}

// Function to generate a random color based on a u32 seed
fn randomColor(seed: u32) -> vec3<f32> {
    return random_colors[seed].xyz;
}

fn lambert(light_direction: vec3f, normal: vec3f) -> f32 {
    return max(dot(-light_direction, normal), 0.);
}

fn hsv2rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;

    let c = v * s; // Chroma
    let h_prime = h * 6.0;
    let x = c * (1.0 - abs(fract(h_prime) * 2.0 - 1.0));

    var rgb = vec3<f32>(0.0, 0.0, 0.0);

    if (h_prime < 1.0) {
        rgb = vec3(c, x, 0.0);
    } else if (h_prime < 2.0) {
        rgb = vec3(x, c, 0.0);
    } else if (h_prime < 3.0) {
        rgb = vec3(0.0, c, x);
    } else if (h_prime < 4.0) {
        rgb = vec3(0.0, x, c);
    } else if (h_prime < 5.0) {
        rgb = vec3(x, 0.0, c);
    } else {
        rgb = vec3(c, 0.0, x);
    }

    let m = v - c;
    return rgb + vec3(m, m, m);
}


fn get_sphere_color(id: u32) -> vec3f {
    // let hue = randomColor(id).x;
    // let saturation = 1.;
    // let value =randomColor(id).z;

    // let color = hsv2rgb(vec3(hue, saturation, value));
    // return color;

    return randomColor(id);
}

fn get_color(ray: Ray, i: Intersection) -> vec4f {
    let albedo = get_sphere_color(i.index);

    // Point light
    let light_direction = LIGHT_DIRECTION;
    var light_intensity = 4.;

    // let new_ray = Ray(i.coord - light_direction * 0.1, -light_direction);
    // let new_hit = bvh_intersect(new_ray);
    // if new_hit.is_hit {
    //     // Hit some other object.
    //     // return vec4(0., 0., 0., 1.);
    //     light_intensity = 0.;
    // }

    // return vec4(albedo * light_intensity, 1.);

    // let roughness = 0.2;
    // let metallic = 0.;
    // let radiance = vec3(1.);

    // // let color = CookTorranceBRDF(albedo, roughness, metallic, ray.direction, i.normal, light_direction, radiance) * light_intensity;
    let color = albedo * lambert(light_direction, i.normal) * light_intensity;
    return vec4(color, 1.);
}

fn occlusion_bvh(i: Intersection, color: vec3f, pixel: vec2<u32>) -> vec3f {
    let id = (pixel.x + pixel.y * uniforms.height) % arrayLength(&random_colors);
    let random = normalize(random_colors[id].xyz);

    let tangent = cross(random, i.normal);
    let bitangent = cross(i.normal, tangent);

    let tangentToViewSpaceMatrix = mat3x3f(tangent, bitangent, i.normal);

    var totalIntensity = vec3(0.);
    let arrLen = arrayLength(&random_directions);

    for (var j = 0u; j < arrLen; j++) {
        let dir = tangentToViewSpaceMatrix * random_directions[j];
        let new_ray = Ray(
            i.coord + dir * 0.01,
            dir,
        );
        let new_i = bvh_intersect(new_ray);
        if new_i.is_hit {
            totalIntensity += get_color(new_ray, new_i).xyz;
        } else {
            totalIntensity += vec3(1.) * lambert(LIGHT_DIRECTION, i.normal);
        }
    }

    totalIntensity /= f32(arrLen);
    return color * totalIntensity;
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
            color += get_color(new_ray, new_hit);
        }

        // Shadow.
        let shadow_ray_direction = -LIGHT_DIRECTION;
        let shadow_ray = Ray(i.coord + shadow_ray_direction * 0.01, shadow_ray_direction);
        let shadow_hit = bvh_intersect(shadow_ray);
        if shadow_hit.is_hit {
            color *= 0.1;
        }

        return vec4(color);

        // let albedo = get_sphere_color(i.index);

        // return vec4(occlusion_bvh(i, albedo.xyz, pixel), 1.);
    }
    return vec4(0.);
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
