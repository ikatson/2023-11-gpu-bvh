use anyhow::Context;

use std::{
    borrow::Cow,
    collections::HashSet,
    fs::File,
    io::BufWriter,
    path::Path,
    sync::{
        atomic::{AtomicU8, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};
use zerocopy::AsBytes;

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    Extent3d, FragmentState, PipelineLayoutDescriptor, PrimitiveState, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderStages, TextureDescriptor, TextureUsages, TextureViewDescriptor, VertexAttribute,
    VertexBufferLayout, VertexState,
};
use winit::{
    event::{Event, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{
        KeyCode,
        PhysicalKey::{self, Code},
    },
    window::WindowBuilder,
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
                // Image is bgra8unorm, so need to swap pixels.
                let rgb = [pixel[2], pixel[1], pixel[0]];
                file.write_all(&rgb)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, zerocopy_derive::AsBytes, Clone, Copy, Default)]
#[repr(C)]
struct PerspectiveCamera {
    position: Vec3,
    _pad: [u8; 4],
    direction: Vec3,
    _pad_2: [u8; 4],
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
    let image = Image::new(output_width, output_height);
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

enum OtherEvent {
    MouseScroll(MouseScrollDelta),
    TouchPadMagnify(f64),
}

struct AppState {
    camera: PerspectiveCamera,
    pressed_keys: HashSet<winit::keyboard::PhysicalKey>,
    other_events: Vec<OtherEvent>,
    time: Instant,
    screen_width: u32,
    screen_height: u32,
    original_camera: PerspectiveCamera,
}

impl AppState {
    fn new(width: u32, height: u32, camera: PerspectiveCamera) -> Self {
        AppState {
            original_camera: camera,
            camera,
            pressed_keys: Default::default(),
            time: Instant::now(),
            screen_height: height,
            screen_width: width,
            other_events: Default::default(),
        }
    }
    fn on_keyboard_event(&mut self, event: winit::event::KeyEvent) {
        let key = match event.physical_key {
            Code(KeyCode::KeyW) | Code(KeyCode::KeyA) | Code(KeyCode::KeyS)
            | Code(KeyCode::KeyD) | Code(KeyCode::Space) | Code(KeyCode::KeyZ)
            | Code(KeyCode::KeyE) | Code(KeyCode::KeyQ) | Code(KeyCode::Enter) => {
                event.physical_key
            }
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

    fn on_mouse_scroll(&mut self, ev: MouseScrollDelta) {
        self.other_events.push(OtherEvent::MouseScroll(ev))
    }

    fn on_touchpad_magnify(&mut self, delta: f64) {
        self.other_events.push(OtherEvent::TouchPadMagnify(delta));
    }

    fn update(&mut self) {
        let dt = self.dt();
        let speed = 10.;
        let rotation_speed = 1.;
        let dt_secs = dt.as_secs_f32();
        let mut movement = Vec3::default();

        if self.pressed_keys.remove(&PhysicalKey::Code(KeyCode::Enter)) {
            self.camera = self.original_camera;
        }

        let mut new_direction = self.camera.direction;
        let forward = self.camera.direction;
        let left = self
            .camera
            .direction
            .cross(&Vec3::new(0., 1., 0.))
            .normalize();
        let up = forward.cross(&left).normalize();
        for key in self.pressed_keys.iter().copied() {
            match key {
                Code(KeyCode::KeyW) => {
                    movement = movement + forward * speed * dt_secs;
                }
                Code(KeyCode::KeyA) => movement = movement + left * speed * dt_secs,
                Code(KeyCode::KeyS) => {
                    movement = movement - forward * speed * dt_secs;
                }
                Code(KeyCode::KeyD) => {
                    movement = movement - left * speed * dt_secs;
                }
                Code(KeyCode::KeyZ) => {
                    movement = movement - up * speed * dt_secs;
                }
                Code(KeyCode::Space) => {
                    movement = movement + up * speed * dt_secs;
                }
                // TODO: this doesn't work
                Code(KeyCode::KeyE) => {
                    new_direction = self
                        .camera
                        .direction
                        .rotate_around_axis(&up, -rotation_speed * dt_secs)
                }
                Code(KeyCode::KeyQ) => {
                    new_direction = self
                        .camera
                        .direction
                        .rotate_around_axis(&up, rotation_speed * dt_secs)
                }
                _ => {}
            }
        }
        for event in self.other_events.drain(..) {
            match event {
                OtherEvent::MouseScroll(ev) => match ev {
                    MouseScrollDelta::PixelDelta(pos) => {
                        const MULT: f32 = 1.;
                        movement =
                            movement + left * MULT * (pos.x as f32) / (self.screen_width as f32);
                        movement =
                            movement + up * MULT * (pos.y as f32) / (self.screen_height as f32);
                    }
                    _ev => {}
                },
                OtherEvent::TouchPadMagnify(delta) => {
                    const MULT: f64 = 100.;
                    self.camera.fov -= (delta * MULT) as f32;
                }
            }
        }
        self.camera.position = self.camera.position + movement;
        self.camera.direction = new_direction;
    }
}

struct BlitTextureToTexturePipeline {
    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    quad_buffer: wgpu::Buffer,
}

impl BlitTextureToTexturePipeline {
    fn new(
        device: &wgpu::Device,
        capabilities: &wgpu::SurfaceCapabilities,
        input_texture: &wgpu::Texture,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw.wgsl"))),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // The input image
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
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
        let quad = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: quad.as_bytes(),
            usage: BufferUsages::VERTEX,
        });

        let input_sampler = device.create_sampler(&SamplerDescriptor {
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

        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &input_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&input_sampler),
                },
            ],
        });

        Self {
            bind_group: bg,
            render_pipeline: pipeline,
            quad_buffer: quad,
        }
    }

    fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, output_texture: &wgpu::Texture) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let view = output_texture.create_view(&TextureViewDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Default::default(),
                })],
                ..Default::default()
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_vertex_buffer(0, self.quad_buffer.slice(..));
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

struct BVHComputePipeline {
    gpu_bvh: GPUBVH,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    output_texture: wgpu::Texture,
    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl BVHComputePipeline {
    fn new(device: &wgpu::Device, bvh: &BVH, width: u32, height: u32) -> Self {
        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("bvh.wgsl"))),
        });

        let gpu_bvh = GPUBVH::new(bvh, device);

        let output = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Output texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Screen size
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&gpu_bvh.bgl, &bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &compute_shader,
            entry_point: "render_through_bvh",
        });

        Self {
            gpu_bvh,
            pipeline,
            bgl,
            output_texture: output,
            uniform_buffer: device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: ComputePipelineUniforms::default().as_bytes(),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }),
            width,
            height,
        }
    }

    fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, camera: &PerspectiveCamera) {
        let uniforms = ComputePipelineUniforms {
            position: camera.position,
            direction: camera.direction,
            fov: camera.fov,
            aspect: camera.aspect,
            width: self.width,
            height: self.height,
            ..Default::default()
        };

        queue.write_buffer(&self.uniform_buffer, 0, uniforms.as_bytes());

        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.output_texture.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.gpu_bvh.bind_group, &[]);
            cpass.set_bind_group(1, &bg, &[]);
            cpass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

struct Renderer {
    compute_pipeline: BVHComputePipeline,
    draw_pipeline: BlitTextureToTexturePipeline,
}

impl Renderer {
    fn new(
        device: &wgpu::Device,
        capabilities: wgpu::SurfaceCapabilities,
        bvh: BVH,
        width: u32,
        height: u32,
    ) -> Self {
        let compute = BVHComputePipeline::new(device, &bvh, width, height);
        let draw =
            BlitTextureToTexturePipeline::new(device, &capabilities, &compute.output_texture);
        Renderer {
            compute_pipeline: compute,
            draw_pipeline: draw,
        }
    }

    fn render(
        &self,
        texture: &wgpu::SurfaceTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &PerspectiveCamera,
    ) {
        self.compute_pipeline.render(device, queue, camera);
        self.draw_pipeline.render(device, queue, &texture.texture);
    }
}

#[derive(Clone, Copy, Default, zerocopy_derive::AsBytes)]
// WGSL: align=16, sizeof=48, stride=48
#[repr(C)]
struct ComputePipelineUniforms {
    position: Vec3,
    fov: f32,
    direction: Vec3,
    aspect: f32,

    width: u32,
    height: u32,
    _pad_struct: [u8; 8],
}

async fn main_wgpu(
    bvh: BVH,
    width: u32,
    height: u32,
    camera: &PerspectiveCamera,
) -> anyhow::Result<()> {
    let el = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
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

    let capabilities = surface.get_capabilities(&adapter);
    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: capabilities.formats[0],
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
        },
    );

    let app = Mutex::new(AppState::new(width, height, *camera));

    let renderer = Renderer::new(&device, capabilities, bvh, width, height);

    std::thread::scope(|s| {
        // "Game logic" thread
        s.spawn(|| loop {
            {
                app.lock().unwrap().update();
            }
            std::thread::sleep(Duration::from_millis(16));
        });

        // Render thread. It will spawn at most at 60 fps by itself.
        s.spawn(|| loop {
            let camera = { app.lock().unwrap().camera };
            timeit!("render", {
                let txt = surface.get_current_texture().unwrap();
                renderer.render(&txt, &device, &queue, &camera);
                txt.present();
            });
        });

        // Event loop.
        el.run(|event, _target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter);
            match event {
                Event::WindowEvent {
                    window_id: _,
                    event,
                } => match event {
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let Code(KeyCode::KeyX) = event.physical_key {
                            std::process::exit(0);
                        };
                        app.lock().unwrap().on_keyboard_event(event);
                    }
                    WindowEvent::MouseWheel {
                        device_id: _,
                        delta,
                        phase: _,
                    } => {
                        app.lock().unwrap().on_mouse_scroll(delta);
                    }
                    WindowEvent::TouchpadMagnify {
                        device_id: _,
                        delta,
                        phase: _,
                    } => {
                        app.lock().unwrap().on_touchpad_magnify(delta);
                    }
                    _we => {}
                },
                _e => {}
            }
        })
        .expect("error running event loop");
    });

    Ok(())
}

fn main() {
    const SPHERES: usize = 32 * 1024;
    let make_shapes = || {
        let mut shapes = Vec::with_capacity(SPHERES);
        for _ in 0..SPHERES {
            let r = |scale: f32, offset: f32| (rand::random::<f32>() + offset) * scale;
            let center = Vec3::new(r(40., -0.5), r(0.1, -0.5), r(40., -0.5));
            shapes.push(Shape::Sphere(Sphere::new(center, r(0.1, 0.05))));
        }
        shapes
    };
    let shapes = timeit!("make shapes", make_shapes());

    let bvh = timeit!("BVH::new", BVH::new(shapes));
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;
    const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;
    const FOV: f32 = 110.;
    let position = Vec3::new(-10., -10., -10.);
    let target = Vec3::new(16., 16., 16.);
    let camera = PerspectiveCamera {
        position,
        direction: (target - position).normalize(),
        fov: FOV,
        aspect: ASPECT,
        ..Default::default()
    };
    // let image = timeit!(
    //     "render_into_image_cpu",
    //     render_bvh_perspective(&bvh, &camera, WIDTH as usize, HEIGHT as usize)
    // );
    // timeit!("write PPM", image.write_ppm("/tmp/image.ppm").unwrap());

    pollster::block_on(main_wgpu(bvh, WIDTH, HEIGHT, &camera)).unwrap();
}
