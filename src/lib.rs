#[derive(Clone, Copy, Default)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl From<f32> for Vec3 {
    fn from(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }
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
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    fn div(&self, other: &Vec3) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy)]
pub struct AxisAlignedQuad {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Copy)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
}

impl Plane {}

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

impl AxisAlignedBox {
    // Intersection using slab method
    // https://education.siggraph.org/static/HyperGraph/raytrace/rtinter3.htm
    pub fn intersects_slow_branching(&self, ray: &Ray) -> bool {
        let mut tnear = f32::NEG_INFINITY;
        let mut tfar = f32::INFINITY;
        for (min_x, max_x, ro_x, rd_x) in [
            (self.min.x, self.max.x, ray.origin.x, ray.direction.x),
            (self.min.y, self.max.y, ray.origin.y, ray.direction.y),
            (self.min.z, self.max.z, ray.origin.z, ray.direction.z),
        ] {
            // ray is parallel to the X plane,
            if rd_x == 0. {
                // And it's outside of the min and max planes.
                if ro_x < min_x || ro_x > max_x {
                    return false;
                }
                // X slab intersects, we're good.
                continue;
            }
            // "T" for time, how much "time" it takes the ray to hit the plane.
            let mut t1 = (min_x - ro_x) / rd_x;
            let mut t2 = (max_x - ro_x) / rd_x;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tnear = tnear.max(t1);
            tfar = tfar.min(t2);
            if tnear > tfar {
                return false;
            }
        }

        true
    }

    pub fn intersects_branchless(&self, ray: &Ray) -> bool {
        let mut tnear = f32::NEG_INFINITY;
        let mut tfar = f32::INFINITY;
        for (min_x, max_x, ro_x, rd_x) in [
            (self.min.x, self.max.x, ray.origin.x, ray.direction.x),
            (self.min.y, self.max.y, ray.origin.y, ray.direction.y),
            (self.min.z, self.max.z, ray.origin.z, ray.direction.z),
        ] {
            // "T" for time, how much "time" it takes the ray to hit the plane.
            // T1 is the first (nearest) hit, t2 is the furtherst hit.
            let t1 = (min_x - ro_x) / rd_x;
            let t2 = (max_x - ro_x) / rd_x;
            let (t1, t2) = (t1.min(t2), t1.max(t2));
            tnear = tnear.max(t1);
            tfar = tfar.min(t2);
        }

        tnear < tfar
    }

    pub fn intersects_vectors(&self, ray: &Ray) -> bool {
        let tnear = f32::NEG_INFINITY;
        let tfar = f32::INFINITY;
        let t1 = (self.min.sub(&ray.origin)).div(&ray.direction);
        let t2 = (self.max.sub(&ray.origin)).div(&ray.direction);
        let (t1, t2) = (t1.min(&t2), t1.max(&t2));

        let tnear = tnear.max(t1.x).max(t1.y).max(t1.z);
        let tfar = tfar.min(t2.x).min(t2.y).min(t2.z);

        tnear < tfar
    }
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
