use anyhow::Context;
use image::{codecs::hdr::HdrDecoder, DynamicImage, ImageDecoder, ImageReader};
use rand::{Rng, RngCore};

use std::{
    borrow::Cow,
    collections::HashSet,
    fs::File,
    io::BufWriter,
    path::Path,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use zerocopy::AsBytes;

use wgpu::{
    rwh::HasWindowHandle,
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Device, Extent3d, FragmentState, PipelineLayoutDescriptor,
    PrimitiveState, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderStages, Surface, Texture, TextureDescriptor, TextureFormat, TextureUsages,
    TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexState,
};
use winit::{
    event::{Event, MouseScrollDelta, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{
        KeyCode,
        PhysicalKey::{self, Code},
    },
    window::{Window, WindowAttributes},
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

impl PerspectiveCamera {
    fn new(position: Vec3, direction: Vec3, fov: f32, aspect: f32) -> Self {
        Self {
            position,
            direction,
            fov,
            aspect,
            ..Default::default()
        }
    }
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
    const UP: Vec3 = Vec3::new(0., 0., 1.);
    let left = camera.direction.cross(&UP).normalize() * -1.;
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

struct InitializedApp {
    camera: PerspectiveCamera,
    pressed_keys: HashSet<winit::keyboard::PhysicalKey>,
    other_events: Vec<OtherEvent>,
    time: Instant,
    screen_width: u32,
    screen_height: u32,
    original_camera: PerspectiveCamera,
    window: Arc<Window>,
    renderer: Renderer,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
}

impl InitializedApp {
    fn new(
        width: u32,
        height: u32,
        camera: PerspectiveCamera,
        window: Arc<Window>,
        renderer: Renderer,
        surface: Surface<'static>,
        device: Device,
        queue: Queue,
    ) -> Self {
        InitializedApp {
            original_camera: camera,
            camera,
            pressed_keys: Default::default(),
            time: Instant::now(),
            screen_height: height,
            screen_width: width,
            other_events: Default::default(),
            window,
            renderer,
            device,
            surface,
            queue,
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
        self.other_events.push(OtherEvent::MouseScroll(ev));
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
        const ABS_UP: Vec3 = Vec3::new(0., 0., 1.);
        let forward = self.camera.direction;
        let left = ABS_UP.cross(&self.camera.direction).normalize();
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
                        // Changed from movement to rotation
                        const ROTATION_SENSITIVITY: f32 = 0.4;

                        // Rotate around vertical axis (left/right)
                        new_direction = self.camera.direction.rotate_around_axis(
                            &up,
                            ROTATION_SENSITIVITY * (pos.x as f32) / (self.screen_width as f32),
                        );

                        // Rotate around horizontal axis (up/down)
                        new_direction = new_direction.rotate_around_axis(
                            &left,
                            -ROTATION_SENSITIVITY * (pos.y as f32) / (self.screen_height as f32),
                        );
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
        output_format: TextureFormat,
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
                entry_point: Some("main_vs"),
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
                compilation_options: Default::default(),
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
                entry_point: Some("main_fs"),
                targets: &[Some(output_format.into())],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: Default::default(),
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
    random_colors: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

fn load_texture(device: &wgpu::Device, queue: &wgpu::Queue, filename: &str) -> Texture {
    let img = ImageReader::open(filename)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba32f();
    let texture = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(filename),
            size: Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        },
        Default::default(),
        img.as_bytes(),
    );
    texture
}

impl BVHComputePipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, bvh: &BVH, width: u32, height: u32) -> Self {
        let bgtexture = load_texture(device, queue, "resources/background.hdr");
        let bgtexture_irradiance =
            load_texture(device, queue, "resources/background.hdr.irradiance.exr");

        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("bvh.wgsl"))),
        });

        let random_colors = {
            let mut random_colors = vec![0f32; bvh.objects().len() * 4];
            rand::thread_rng().try_fill(&mut random_colors[..]).unwrap();
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: random_colors.as_bytes(),
                usage: BufferUsages::STORAGE,
            })
        };

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
            format: wgpu::TextureFormat::Rgba16Float,
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
                        format: wgpu::TextureFormat::Rgba16Float,
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
                // Random colors
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Random directions on a hemisphere
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Background
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Background irradiance
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
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
            entry_point: Some("render_through_bvh"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: ComputePipelineUniforms::default().as_bytes(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let random_directions_buffer = {
            let mut random_directions = [0f32; 4 * 4];
            rand::thread_rng()
                .try_fill(&mut random_directions[..])
                .unwrap();
            random_directions.chunks_exact_mut(4).for_each(|chunk| {
                let normalized = Vec3::new(chunk[0], chunk[1], chunk[2].abs()).normalize();
                chunk[0] = normalized.x;
                chunk[1] = normalized.y;
                chunk[2] = normalized.z;
            });
            device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: random_directions.as_bytes(),
                usage: BufferUsages::STORAGE,
            })
        };

        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &output.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: random_colors.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: random_directions_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &bgtexture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(
                        &bgtexture_irradiance.create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        Self {
            gpu_bvh,
            pipeline,
            bgl,
            output_texture: output,
            uniform_buffer,
            random_colors,
            bg,
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

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.gpu_bvh.bind_group, &[]);
            cpass.set_bind_group(1, &self.bg, &[]);
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
        queue: &wgpu::Queue,
        output_format: TextureFormat,
        bvh: BVH,
        width: u32,
        height: u32,
    ) -> Self {
        let compute = BVHComputePipeline::new(device, queue, &bvh, width, height);
        let draw =
            BlitTextureToTexturePipeline::new(device, output_format, &compute.output_texture);

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

struct InitializingArgs {
    camera_position: Vec3,
    camera_target: Vec3,
    bvh: BVH,
}

#[derive(Default)]
enum WinitAppHandlerState {
    #[default]
    Unknown,
    Initializing(InitializingArgs),
    Initialized(Arc<Mutex<InitializedApp>>),
}

impl WinitAppHandlerState {
    fn initialize(&mut self, window: Window) {
        let args = match std::mem::take(self) {
            WinitAppHandlerState::Initializing(args) => args,
            _ => panic!("bad state"),
        };
        let window = Arc::new(window);
        let (width, height) = (window.inner_size().width, window.inner_size().height);
        let camera = PerspectiveCamera::new(
            args.camera_position,
            (args.camera_target - args.camera_position).normalize(),
            110.,
            width as f32 / height as f32,
        );

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                })
                .await
                .context("error requesting adapter")
                .unwrap();

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        ..Default::default()
                    },
                    None,
                )
                .await
                .context("error requesting device")
                .unwrap();

            (adapter, device, queue)
        });

        let capabilities = dbg!(surface.get_capabilities(&adapter));
        // let output_format = capabilities.formats[0];
        let output_format = TextureFormat::Rgba16Float;
        // let output_format = TextureFormat::Bgra8UnormSrgb;
        // let output_format = TextureFormat::Bgra8Unorm;
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                format: output_format,
                width,
                height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: capabilities.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let renderer = Renderer::new(&device, &queue, output_format, args.bvh, width, height);
        let app = InitializedApp::new(
            width,
            height,
            camera,
            window.clone(),
            renderer,
            surface,
            device,
            queue,
        );
        let app = Arc::new(Mutex::new(app));
        *self = WinitAppHandlerState::Initialized(app.clone());
    }
}

struct WinitAppHandler {
    state: WinitAppHandlerState,
}

impl WinitAppHandler {
    pub fn new(camera_position: Vec3, camera_target: Vec3, bvh: BVH) -> Self {
        Self {
            state: WinitAppHandlerState::Initializing(InitializingArgs {
                camera_position,
                camera_target,
                bvh,
            }),
        }
    }
}

impl winit::application::ApplicationHandler for WinitAppHandler {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(Default::default()).unwrap();
        self.state.initialize(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let app = match &self.state {
            WinitAppHandlerState::Unknown => panic!("unknown state"),
            WinitAppHandlerState::Initializing(_) => return,
            WinitAppHandlerState::Initialized(app) => app,
        };
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let mut app = app.lock().unwrap();
                app.update();
                let texture = app.surface.get_current_texture().unwrap();
                app.renderer
                    .render(&texture, &app.device, &app.queue, &app.camera);
                app.window.pre_present_notify();
                texture.present();

                // This is the main call that will loop
                app.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Code(KeyCode::KeyX) = event.physical_key {
                    std::process::exit(0);
                };
                let mut app = app.lock().unwrap();
                app.on_keyboard_event(event);
                app.window.request_redraw();
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta,
                phase: _,
            } => {
                let mut app = app.lock().unwrap();
                app.on_mouse_scroll(delta);
                app.window.request_redraw();
            }
            WindowEvent::PinchGesture {
                device_id: _,
                delta,
                phase: _,
            } => {
                let mut app = app.lock().unwrap();
                app.on_touchpad_magnify(delta);
                app.window.request_redraw();
            }
            _we => {}
        }
    }
}

fn main_wgpu(bvh: BVH, camera_position: Vec3, camera_target: Vec3) -> anyhow::Result<()> {
    let el = EventLoop::new()?;
    el.run_app(&mut WinitAppHandler::new(
        camera_position,
        camera_target,
        bvh,
    ))?;
    Ok(())
}

fn main() {
    const SPHERES: usize = 2048;
    // X is right
    // Y is forward
    // Z is up

    let make_shapes = || {
        let mut shapes = Vec::with_capacity(SPHERES);
        for _ in 0..SPHERES {
            let r = |scale: f32, offset: f32| (rand::random::<f32>() + offset) * scale;
            let center = Vec3::new(r(10., -0.5), r(10., -0.5), r(1., -0.5));
            shapes.push(Shape::Sphere(Sphere::new(center, r(0.2, 0.05))));
        }
        shapes
    };
    let shapes = timeit!("make shapes", make_shapes());
    let bvh = timeit!("BVH::new", BVH::new(shapes));
    let camera_position = Vec3::new(-7., 6., 5.);
    let camera_target = Vec3::new(0., 0., 0.);

    main_wgpu(bvh, camera_position, camera_target).unwrap();
}
