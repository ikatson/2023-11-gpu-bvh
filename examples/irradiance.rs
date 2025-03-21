use std::sync::mpsc;

use image::ImageBuffer;
use wgpu::{
    include_wgsl,
    util::{DeviceExt, TextureDataOrder},
    BindGroupEntry, BindGroupLayoutEntry, BufferAddress, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, DeviceDescriptor, Features, Limits, PipelineLayout,
    RequestAdapterOptions, ShaderModule, ShaderStages, TexelCopyBufferInfo, TexelCopyTextureInfo,
    TextureDescriptor, TextureUsages,
};
use zerocopy::AsBytes;

fn main() {
    let instance = wgpu::Instance::default();
    let (adapter, device, queue) = pollster::block_on(async {
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .unwrap();
        (adapter, device, queue)
    });

    let imgpath = std::env::args().nth(1).unwrap();

    let img = image::ImageReader::open(&imgpath)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba32f();
    let input_texture = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba32Float],
        },
        TextureDataOrder::default(),
        img.as_bytes(),
    );

    let output_texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[wgpu::TextureFormat::Rgba32Float],
    });

    let shader = include_wgsl!("../src/irradiance.wgsl");
    let shader = device.create_shader_module(shader);

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let playout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&playout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &input_texture.create_view(&Default::default()),
                ),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &output_texture.create_view(&Default::default()),
                ),
            },
        ],
    });

    let outbuf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 512 * 512 * 4 * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut cmd = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut cpass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        // cpass.dispatch_workgroups(512 / 8, 512 / 8, 1);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    cmd.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &outbuf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(512 * 4 * 4),
                rows_per_image: Some(512),
            },
        },
        wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
    );

    println!("submitting");
    queue.submit([cmd.finish()]);
    println!("submitted");

    let data = outbuf.slice(..);
    let (tx, rx) = mpsc::channel();
    println!("mapping");
    data.map_async(wgpu::MapMode::Read, move |v| {
        println!("mapped");
        tx.send(v).unwrap();
    });
    rx.recv().unwrap().unwrap();
    println!("mapped 2");

    let data = data.get_mapped_range();

    let data_f32: &[f32] =
        unsafe { core::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };

    let img = ImageBuffer::<image::Rgba<f32>, _>::from_raw(512, 512, data_f32).unwrap();
    img.save(format!("{imgpath}.irradiance.exr")).unwrap()
}
