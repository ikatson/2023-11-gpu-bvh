struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) tx_coord: vec2<f32>,
}

@vertex
fn main_vs(
    @builtin(vertex_index) i: u32,
    @location(0) quad_vertex: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(quad_vertex.xy, 0., 1.0);
    out.tx_coord = out.pos.xy / 2. + 0.5;
    out.tx_coord.y *= -1.;
    return out;
}

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var input_texture_sampler: sampler;

fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (color * (a * color + b)) / (color * (c * color + d) + e);
}

fn tonemap_hdr(color: vec3<f32>) -> vec3<f32> {
    // Simple "film-like" curve that rolls off highlights but keeps >1.0 possible
    let shoulder_strength = 0.22;
    let linear_strength = 0.3;
    let linear_angle = 0.1;
    let toe_strength = 0.2;

    let rolled = ((color * (shoulder_strength * color + linear_strength)) /
                 (color * (shoulder_strength * color + linear_angle) + toe_strength));
    return rolled;
}

@fragment
fn main_fs(
    vo: VertexOutput,
) -> @location(0) vec4<f32> {
    let color: vec4<f32> = textureSample(input_texture, input_texture_sampler, vo.tx_coord);

    const DISPLAY_RANGE: f32 = 1.5;

    // return vec4(tonemap_reinhard(color.rgb), 1.) * DISPLAY_RANGE;
    // return vec4(tonemap_aces(color.rgb), 1.) * DISPLAY_RANGE;
    // return vec4(tonemap_hdr(color.rgb), 1.) * DISPLAY_RANGE;
    return color;
}
