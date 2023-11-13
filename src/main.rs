use anyhow::Context;
use std::{
    borrow::Cow,
    collections::HashSet,
    fs::File,
    io::BufWriter,
    path::Path,
    sync::atomic::{AtomicU8, Ordering},
    time::{Duration, Instant},
};
use zerocopy::AsBytes;

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BufferUsages, ColorTargetState,
    ColorWrites, FragmentState, PipelineLayout, PipelineLayoutDescriptor, PrimitiveState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, SamplerDescriptor,
    ShaderModuleDescriptor, TextureUsages, TextureViewDescriptor, VertexAttribute,
    VertexBufferLayout, VertexState,
};
use winit::{
    event::{Event, StartCause, WindowEvent},
    event_loop::EventLoop,
    keyboard::KeyCode,
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
            // bgra
            pixels: vec![255; width * height * 4],
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
            for pixel in row.chunks(4) {
                let rgb = [pixel[2], pixel[1], pixel[0]];
                file.write_all(&rgb)?;
            }
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
                    let base_i = pixel * 4;
                    pixels_view[base_i].store(b, Ordering::Relaxed);
                    pixels_view[base_i + 1].store(g, Ordering::Relaxed);
                    pixels_view[base_i + 2].store(r, Ordering::Relaxed);
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
    render_pipeline: wgpu::RenderPipeline,
    render_pipeline_bgl: wgpu::BindGroupLayout,
    quad_buffer: wgpu::Buffer,
    screen_size_uniform: wgpu::Buffer,
    output_width: u32,
    output_height: u32,
    pressed_keys: HashSet<winit::keyboard::PhysicalKey>,
    time: Instant,
}

impl App {
    fn on_keyboard_event(&mut self, event: winit::event::KeyEvent) {
        let key = match event.physical_key {
            winit::keyboard::PhysicalKey::Code(KeyCode::KeyW)
            | winit::keyboard::PhysicalKey::Code(KeyCode::KeyA)
            | winit::keyboard::PhysicalKey::Code(KeyCode::KeyS)
            | winit::keyboard::PhysicalKey::Code(KeyCode::KeyD) => event.physical_key,
            _ => return,
        };
        match event.state {
            winit::event::ElementState::Pressed => {
                self.pressed_keys.insert(key);
            }
            winit::event::ElementState::Released => {
                self.pressed_keys.remove(&key);
            }
        }
    }

    fn dt(&mut self) -> Duration {
        let now = Instant::now();
        let dt = now - self.time;
        self.time = now;
        dt
    }

    fn update(&mut self, dt: &Duration) {
        let speed = 10.;
        let dt_secs = dt.as_secs_f32();
        let mut direction = Vec3::default();
        let forward = self.camera.direction;
        let left = self
            .camera
            .direction
            .cross(&Vec3::new(0., 1., 0.))
            .normalize();
        let up = left.cross(&forward).normalize();
        for key in self.pressed_keys.iter().copied() {
            match key {
                winit::keyboard::PhysicalKey::Code(KeyCode::KeyW) => {
                    direction = direction + forward * speed * dt_secs;
                }
                winit::keyboard::PhysicalKey::Code(KeyCode::KeyA) => {
                    direction = direction + left * speed * dt_secs
                }
                winit::keyboard::PhysicalKey::Code(KeyCode::KeyS) => {
                    direction = direction - forward * speed * dt_secs;
                }
                winit::keyboard::PhysicalKey::Code(KeyCode::KeyD) => {
                    direction = direction - left * speed * dt_secs;
                }
                _ => {}
            }
        }
        self.camera.position = self.camera.position + direction;
    }

    fn render(
        &self,
        texture: &wgpu::SurfaceTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> anyhow::Result<()> {
        let image = render_bvh_perspective(
            &self.bvh,
            &self.camera,
            self.output_width as usize,
            self.output_height as usize,
        );

        let image_texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: self.output_width,
                    height: self.output_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            image.pixels.as_bytes(),
        );
        let image_texture_sampler = device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.,
            lod_max_clamp: 32.,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        // let image_buf = device.create_buffer_init(&BufferInitDescriptor {
        //     label: None,
        //     contents: image.pixels.as_bytes(),
        //     usage: BufferUsages::STORAGE,
        // });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.render_pipeline_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: None,
                            dimension: None,
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: None,
                            base_array_layer: 0,
                            array_layer_count: None,
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        self.screen_size_uniform.as_entire_buffer_binding(),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&image_texture_sampler),
                },
            ],
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let view = texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_vertex_buffer(0, self.quad_buffer.slice(..));
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

async fn main_wgpu(bvh: BVH) -> anyhow::Result<()> {
    let mut el = EventLoop::new()?;
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);

    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 1024;
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT))
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

    let capabilities = surface.get_capabilities(&adapter);
    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: capabilities.formats[0],
            width: WIDTH,
            height: HEIGHT,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
        },
    );

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw.wgsl"))),
    });

    let render_pipeline_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // The input image
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Uniforms (screen size)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // ty: wgpu::BindingType::Texture {
                //     sample_type: wgpu::TextureSampleType::Float { filterable: false },
                //     view_dimension: wgpu::TextureViewDimension::D2,
                //     multisampled: false,
                // },
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });

    #[derive(zerocopy_derive::AsBytes)]
    #[repr(C)]
    struct ScreenSize {
        width: u32,
        height: u32,
    }

    let screen_size = ScreenSize {
        width: WIDTH,
        height: HEIGHT,
    };

    let screen_size_uniform = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: screen_size.as_bytes(),
        usage: BufferUsages::UNIFORM,
    });

    let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        // TBDs
        bind_group_layouts: &[&render_pipeline_bgl],
        push_constant_ranges: &[],
    });
    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: "main_vs",
            // Quad vertex buffer
            buffers: &[VertexBufferLayout {
                array_stride: core::mem::size_of::<Vec3>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
            //            buffers: &[],
        },
        primitive: PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "main_fs",
            targets: &[Some(capabilities.formats[0].into())],
        }),
        multiview: None,
    });
    let quad: [Vec3; 4] = [
        Vec3::new(-1., 1., 0.),
        Vec3::new(-1., -1., 0.),
        Vec3::new(1., 1., 0.),
        Vec3::new(1., -1., 0.),
    ];
    dbg!(core::mem::size_of_val(&quad));
    dbg!(core::mem::size_of::<Vec3>());
    let quad_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: quad.as_bytes(),
        usage: BufferUsages::VERTEX,
    });

    let mut app = App {
        bvh,
        camera,
        render_pipeline,
        render_pipeline_bgl,
        quad_buffer,
        screen_size_uniform,
        output_width: WIDTH,
        output_height: HEIGHT,
        pressed_keys: Default::default(),
        time: Instant::now(),
    };

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
                WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                    app.on_keyboard_event(event)
                },
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
                    let dt = app.dt();
                    app.update(&dt);
                    let txt = surface.get_current_texture().unwrap();
                    app.render(&txt, &device, &queue).unwrap();
                    txt.present();
                }
                we => {
                    // dbg!(we);
                }
            },
            Event::NewEvents(StartCause::Poll) => window.request_redraw(),
            e => {
//                dbg!(e);
            }
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
    // const X: usize = 32;
    // const Y: usize = 32;
    // const Z: usize = 32;
    const X: usize = 4;
    const Y: usize = 4;
    const Z: usize = 4;
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
