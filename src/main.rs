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

    for x in 0..output_width {
        for y in 0..output_height {
            let u = (x as f32) / (output_width as f32) - 0.5;
            let v = (y as f32) / (output_height as f32) - 0.5;
            let target = camera.position + camera.direction - left * hor * u + up * vert * v;
            let direction = (target - camera.position).normalize();
            let ray = Ray {
                origin: camera.position,
                direction,
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
    const X: usize = 32;
    const Y: usize = 32;
    const Z: usize = 32;
    const RADIUS: f32 = 0.5;
    let mut shapes = Vec::with_capacity(X * Y * Z);
    for x in 0..X {
        for y in 0..Y {
            for z in 0..Z {
                let center = Vec3::new(x as f32, y as f32, z as f32);
                shapes.push(Shape::Sphere(Sphere::new(center, RADIUS)));
            }
        }
    }
    let bvh = BVH::new(shapes);
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
    let image = render_bvh_perspective(&bvh, &camera, WIDTH, HEIGHT);
    image.write_ppm("/tmp/image.ppm").unwrap();
}
