use std::{
    fs::File,
    io::BufWriter,
    path::Path,
    sync::atomic::{AtomicU8, Ordering},
};

use anyhow::Context;

use wgpu::{RenderPassColorAttachment, RenderPassDescriptor, TextureViewDescriptor};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder, WindowId},
};

use bvh::*;

struct Image {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; width * height * 3],
        }
    }
}

impl Image {
    fn write_ppm(&self, filename: impl AsRef<Path>) -> anyhow::Result<()> {
        let file = File::create(filename)?;
        let mut file = BufWriter::new(file);
        use std::io::Write;
        writeln!(file, "P6")?;
        writeln!(file, "{} {}", self.width, self.height)?;
        writeln!(file, "255")?;
        for row in self.pixels.chunks(self.height) {
            file.write_all(row)?;
        }
        Ok(())
    }
}

struct OrthoCamera {
    position: Vec3,
    direction: Vec3,
    width: f32,
    height: f32,
}

impl OrthoCamera {
    fn new_from_pos_and_target(position: Vec3, target: Vec3, width: f32, height: f32) -> Self {
        Self {
            position,
            direction: (target - position).normalize(),
            width,
            height,
        }
    }
}

struct PerspectiveCamera {
    position: Vec3,
    direction: Vec3,
    fov: f32,
    aspect: f32,
}

fn render_bvh_ortho(
    bvh: &BVH,
    camera: &OrthoCamera,
    output_width: usize,
    output_height: usize,
) -> Image {
    // render single-threaded first to ensure it works
    let mut image = Image::new(output_width, output_height);
    let forward = camera.direction;
    let left = camera.direction.cross(&Vec3::new(0., 1., 0.)).normalize();
    let up = left.cross(&forward).normalize();
    for x in 0..output_width {
        for y in 0..output_height {
            let u = ((x as f32) / (output_width as f32) - 0.5) * camera.width;
            let v = ((y as f32) / (output_height as f32) - 0.5) * camera.height;
            let origin = camera.position - left * u + up * v;
            let ray = Ray {
                origin,
                direction: camera.direction,
            };
            let i = bvh.intersection(&ray);
            if let Some(i) = i {
                image.pixels[y * output_width * 3 + x * 3] = (i.0.normal.x * 255.).abs() as u8;
                image.pixels[y * output_width * 3 + x * 3 + 1] = (i.0.normal.y * 255.).abs() as u8;
                image.pixels[y * output_width * 3 + x * 3 + 2] = (i.0.normal.z * 255.).abs() as u8;
            }
        }
    }
    image
}

fn render_bvh_perspective(
    bvh: &BVH,
    camera: &PerspectiveCamera,
    output_width: usize,
    output_height: usize,
) -> Image {
    use rayon::prelude::*;

    const PI: f32 = std::f32::consts::PI;
    let mut image = Image::new(output_width, output_height);
    let forward = camera.direction;
    let left = camera.direction.cross(&Vec3::new(0., 1., 0.)).normalize();
    let up = left.cross(&forward).normalize();

    // fov/2 = hor / dirlen. As dirlen == 1, thus fov/2 = hor
    // as fov is in degrees, we need to convert to radians also, so
    // hor = fov/2 * PI / 180
    let hor = camera.fov / 360. * PI;
    let vert = hor / camera.aspect;

    let pixels_view: &[AtomicU8] = unsafe {
        let ptr = image.pixels.as_ptr() as *const AtomicU8;
        core::slice::from_raw_parts(ptr, image.pixels.len())
    };

    let total_pixels = output_height * output_width;
    let pixels_per_block = 4;
    (0..(total_pixels / pixels_per_block))
        .into_par_iter()
        .for_each(|pixel_block| {
            for n in 0..pixels_per_block {
                let pixel = pixel_block * pixels_per_block + n;
                let x = pixel % output_width;
                let y = pixel / output_width;
                let u = (x as f32) / (output_width as f32) - 0.5;
                let v = (y as f32) / (output_height as f32) - 0.5;
                let target = camera.position + camera.direction - left * hor * u + up * vert * v;
                let direction = (target - camera.position).normalize();
                let ray = Ray {
                    origin: camera.position,
                    direction,
                };
                if let Some((i, _shape)) = bvh.intersection(&ray) {
                    let rgb = (i.normal * 255.).abs();
                    let r = rgb.x as u8;
                    let g = rgb.y as u8;
                    let b = rgb.z as u8;
                    let base_i = pixel * 3;
                    pixels_view[base_i].store(r, Ordering::Relaxed);
                    pixels_view[base_i + 1].store(g, Ordering::Relaxed);
                    pixels_view[base_i + 2].store(b, Ordering::Relaxed);
                }
            }
        });
    std::sync::atomic::fence(Ordering::SeqCst);
    image
}

macro_rules! timeit {
    ($name:expr, $expr:expr) => {{
        let t = std::time::Instant::now();
        let res = $expr;
        println!("{}: {:?}", $name, t.elapsed());
        res
    }};
}

struct App {
    bvh: BVH,
    camera: PerspectiveCamera,
}

impl App {
    fn render(
        &self,
        texture: &wgpu::SurfaceTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> anyhow::Result<()> {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let view = texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        let rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        todo!()
    }
}

async fn main_wgpu(bvh: BVH) -> anyhow::Result<()> {
    let el = EventLoop::new()?;
    const WIDTH: usize = 640;
    const HEIGHT: usize = 480;
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH as i32, HEIGHT as i32))
        .build(&el)?;

    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window) }?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .context("error requesting adapter")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::default(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .context("error requesting device")?;

    let position = Vec3::new(-10., -10., -10.);
    let target = Vec3::new(16., 16., 16.);
    const FOV: f32 = 90.;
    const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;
    let camera = PerspectiveCamera {
        position,
        direction: (target - position).normalize(),
        fov: FOV,
        aspect: ASPECT,
    };

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("destination"),
        size: wgpu::Extent3d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut app = App { bvh, camera };

    el.run(move |event, target| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter);
        match event {
            Event::WindowEvent { window_id, event } => match event {
                // WindowEvent::ActivationTokenDone { serial, token } => todo!(),
                // WindowEvent::Resized(_) => todo!(),
                // WindowEvent::Moved(_) => todo!(),
                // WindowEvent::CloseRequested => todo!(),
                // WindowEvent::Destroyed => todo!(),
                // WindowEvent::DroppedFile(_) => todo!(),
                // WindowEvent::HoveredFile(_) => todo!(),
                // WindowEvent::HoveredFileCancelled => todo!(),
                // WindowEvent::Focused(_) => todo!(),
                // WindowEvent::KeyboardInput { device_id, event, is_synthetic } => todo!(),
                // WindowEvent::ModifiersChanged(_) => todo!(),
                // WindowEvent::Ime(_) => todo!(),
                // WindowEvent::CursorMoved { device_id, position } => todo!(),
                // WindowEvent::CursorEntered { device_id } => todo!(),
                // WindowEvent::CursorLeft { device_id } => todo!(),
                // WindowEvent::MouseWheel { device_id, delta, phase } => todo!(),
                // WindowEvent::MouseInput { device_id, state, button } => todo!(),
                // WindowEvent::TouchpadMagnify { device_id, delta, phase } => todo!(),
                // WindowEvent::SmartMagnify { device_id } => todo!(),
                // WindowEvent::TouchpadRotate { device_id, delta, phase } => todo!(),
                // WindowEvent::TouchpadPressure { device_id, pressure, stage } => todo!(),
                // WindowEvent::AxisMotion { device_id, axis, value } => todo!(),
                // WindowEvent::Touch(_) => todo!(),
                // WindowEvent::ScaleFactorChanged { scale_factor, inner_size_writer } => todo!(),
                // WindowEvent::ThemeChanged(_) => todo!(),
                // WindowEvent::Occluded(_) => todo!(),
                WindowEvent::RedrawRequested => {
                    let txt = surface.get_current_texture().unwrap();
                    app.render(&txt, &device, &queue).unwrap();
                    txt.present();
                },
                _ => {}
            },
            _ => {}
            // Event::NewEvents(_) => todo!(),
            // Event::DeviceEvent { device_id, event } => todo!(),
            // Event::UserEvent(_) => todo!(),
            // Event::Suspended => todo!(),
            // Event::Resumed => todo!(),
            // Event::AboutToWait => todo!(),
            // Event::LoopExiting => todo!(),
            // Event::MemoryWarning => todo!(),
        }
    })
    .context("error running event loop")?;

    Ok(())
}

fn main() {
    const X: usize = 32;
    const Y: usize = 32;
    const Z: usize = 32;
    const RADIUS: f32 = 0.5;
    let make_shapes = || {
        let mut shapes = Vec::with_capacity(X * Y * Z);
        for x in 0..X {
            for y in 0..Y {
                for z in 0..Z {
                    let center = Vec3::new(x as f32, y as f32, z as f32);
                    shapes.push(Shape::Sphere(Sphere::new(center, RADIUS)));
                }
            }
        }
        shapes
    };
    let shapes = timeit!("make shapes", make_shapes());

    let bvh = timeit!("BVH::new", BVH::new(shapes));
    // let camera = OrthoCamera::new_from_pos_and_target(
    //     Vec3::new(4., 0., 0.),
    //     Vec3::new(0., 0., 4.),
    //     10.,
    //     10.,
    // );
    const WIDTH: usize = 640;
    const HEIGHT: usize = 480;
    const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;
    const FOV: f32 = 110.;
    let position = Vec3::new(-10., -10., -10.);
    let target = Vec3::new(16., 16., 16.);
    let camera = PerspectiveCamera {
        position,
        direction: (target - position).normalize(),
        fov: FOV,
        aspect: ASPECT,
    };
    let image = timeit!(
        "render_640_480",
        render_bvh_perspective(&bvh, &camera, WIDTH, HEIGHT)
    );
    timeit!("write PPM", image.write_ppm("/tmp/image.ppm").unwrap());

    pollster::block_on(main_wgpu(bvh)).unwrap();
}
