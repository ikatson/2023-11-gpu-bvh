@vertex
fn main_vs(
    @location(0) quad_vertex: vec3<f32>
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(quad_vertex.xy, 0., 1.0);
}

struct ScreenSize {
    width: u32,
    height: u32,
}

@group(0) @binding(0)
var<storage,read> input_texture: array<u32>;

@group(0) @binding(1)
var<uniform> screen_size: ScreenSize;

@fragment
fn main_fs(
    @builtin(position) pos: vec4<f32>
) -> @location(0) vec4<f32> {
    let idx = u32((pos.y / 2. + 0.5) * f32(screen_size.height) + (pos.x / 2. + 0.5) * f32(screen_size.width));
    let bgra_pixel: u32 = input_texture[idx];
    let a = bgra_pixel >> 24u;
    let b = (bgra_pixel >> 16u) & 255u;
    let g = (bgra_pixel >> 8u) & 255u;
    let r = bgra_pixel & 255u;
    return vec4<f32>(f32(r) / 255., f32(g) / 255., f32(b) / 255., 1.);
}