#[derive(Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    pub fn normalize(&self) -> Self {
        let len = self.magnitude();
        self.scalar_div(len)
    }
    pub fn scalar_mult(&self, number: f32) -> Self {
        Self {
            x: self.x * number,
            y: self.y * number,
            z: self.z * number,
        }
    }
    pub fn scalar_div(&self, number: f32) -> Self {
        Self {
            x: self.x / number,
            y: self.y / number,
            z: self.z / number,
        }
    }
    pub fn magnitude(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}

#[derive(Clone, Copy)]
struct Sphere {
    center: Vec3,
    radius: f32,
}

#[derive(Clone, Copy)]
struct AxisAlignedBox {
    // I can represent multiple ways:
    // 2 opposite corners (6 * f32)
    // center, width, height, depth - same
    // one corner, width, height, depth - same
    // let's do the first one (2 opposite corners)
    min: Vec3,
    max: Vec3,
}

#[derive(Clone, Copy)]
struct Intersection {
    coord: Vec3,
    normal: Vec3,
}

#[derive(Clone, Copy)]
enum Shape {
    Sphere(Sphere),
    AAB(AxisAlignedBox),
}
