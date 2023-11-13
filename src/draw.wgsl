// Vec3::new(-1., 1., 0.),
// Vec3::new(-1., -1., 0.),
// Vec3::new(1., 1., 0.),
// Vec3::new(1., -1., 0.),

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
    return out;
    // if i == 0u {
    //     return vec4<f32>(-1., 1., 0., 1.);
    // }
    // if i == 1u {
    //     return vec4<f32>(-1., -1., 0., 1.);
    // }
    // if i == 2u {
    //     return vec4<f32>(1., 1., 0., 1.);
    // }
    // if i == 3u {
    //     return vec4<f32>(1., -1., 0., 1.);
    // }
    // return vec4<f32>(0.);
}

struct ScreenSize {
    width: u32,
    height: u32,
}

@group(0) @binding(0)
// var input_texture: texture_storage_2d<bgra8unorm, read>;
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var<uniform> screen_size: ScreenSize;

@group(0) @binding(2)
var input_texture_sampler: sampler;

@fragment
fn main_fs(
    vo: VertexOutput,
) -> @location(0) vec4<f32> {
//    let x = (pos.x / 2. + 0.5) * f32(screen_size.width);
//    let y = (pos.y / 2. + 0.5) * f32(screen_size.height);
    // let sample_pos = vec2(x, y);
//    return textureLoad(input_texture, vec2(u32(x), u32(y)));
    return textureSample(input_texture, input_texture_sampler, vo.tx_coord);
    // let idx = u32((pos.y / 2. + 0.5) * f32(screen_size.height) + (pos.x / 2. + 0.5) * f32(screen_size.width));
    // let bgra_pixel: u32 = input_texture[idx];
    // let a = bgra_pixel >> 24u;
    // let b = (bgra_pixel >> 16u) & 255u;
    // let g = (bgra_pixel >> 8u) & 255u;
    // let r = bgra_pixel & 255u;
    // return vec4<f32>(f32(r) / 255., f32(g) / 255., f32(b) / 255., 1.);
}