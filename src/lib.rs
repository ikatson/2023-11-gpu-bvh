#[derive(Clone, Copy, Default)]
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
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy)]
pub struct AxisAlignedBox {
    // I can represent multiple ways:
    // 2 opposite corners (6 * f32)
    // center, width, height, depth - same
    // one corner, width, height, depth - same
    // let's do the first one (2 opposite corners)
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Copy)]
pub struct Intersection {
    pub coord: Vec3,
    pub normal: Vec3,
}

#[derive(Clone, Copy)]
pub enum Shape {
    Sphere(Sphere),
    AAB(AxisAlignedBox),
}
