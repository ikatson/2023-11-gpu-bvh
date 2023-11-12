use std::{fs::File, io::BufWriter, path::Path};

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

fn main() {
    let shapes: Vec<Shape> = vec![
        Shape::Sphere(Sphere {
            center: Vec3::new(0., 0., 0.),
            radius: 1.,
        }),
        Shape::Sphere(Sphere {
            center: Vec3::new(0., 0., 2.),
            radius: 1.,
        }),
        Shape::Sphere(Sphere {
            center: Vec3::new(0., 0., 4.),
            radius: 1.,
        }),
        Shape::Sphere(Sphere {
            center: Vec3::new(0., 0., 6.),
            radius: 1.,
        }),
        Shape::Sphere(Sphere {
            center: Vec3::new(0., 0., 8.),
            radius: 1.,
        }),
    ];
    let bvh = BVH::new(shapes);
    let camera =
        OrthoCamera::new_from_pos_and_target(Vec3::new(4., 0., 0.), Vec3::new(0., 0., 4.), 5., 5.);
    let image = render_bvh_ortho(&bvh, &camera, 640, 640);
    image.write_ppm("/tmp/image.ppm").unwrap();
}
