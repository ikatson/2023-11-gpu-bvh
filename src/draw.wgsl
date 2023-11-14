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

@fragment
fn main_fs(
    vo: VertexOutput,
) -> @location(0) vec4<f32> {
    return textureSample(input_texture, input_texture_sampler, vo.tx_coord);
}