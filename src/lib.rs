use std::ops::{Add, Div, Sub};

#[derive(Clone, Copy, Default, Debug)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug)]
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

    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Self { x, y, z }
    }
}

impl<'a> Add<&'a Vec3> for Vec3 {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Add<f32> for Vec3 {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        Vec3 {
            x: self.x + rhs,
            y: self.y + rhs,
            z: self.z + rhs,
        }
    }
}

impl Sub<f32> for Vec3 {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        Vec3 {
            x: self.x - rhs,
            y: self.y - rhs,
            z: self.z - rhs,
        }
    }
}

impl<'a> Sub<&'a Vec3> for Vec3 {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<'a> Div<&'a Vec3> for Vec3 {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        self / *rhs
    }
}

impl Div<Vec3> for Vec3 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}
impl Sphere {
    fn intersection(&self, ray: &Ray) -> Option<Intersection> {
        let sphere = *self;
        // sphere ray intersection
        // https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        let oc = Vec3 {
            x: ray.origin.x - sphere.center.x,
            y: ray.origin.y - sphere.center.y,
            z: ray.origin.z - sphere.center.z,
        };

        let a = ray.direction.dot(&ray.direction);
        let b = 2.0 * oc.dot(&ray.direction);
        let c = oc.dot(&oc) - sphere.radius * sphere.radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            // No intersection
            return None;
        }

        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);

        // Choose the smaller positive root
        let t = if t1 >= 0.0 && t1 < t2 { t1 } else { t2 };

        if t < 0.0 {
            return None;
        }
        // Intersection point
        let coord = Vec3 {
            x: ray.origin.x + t * ray.direction.x,
            y: ray.origin.y + t * ray.direction.y,
            z: ray.origin.z + t * ray.direction.z,
        };
        let normal = (coord - self.center).normalize();
        Some(Intersection { coord, normal })
    }
}

pub trait AABB {
    fn aabb(&self) -> AxisAlignedBox;
}

impl AABB for Sphere {
    fn aabb(&self) -> AxisAlignedBox {
        AxisAlignedBox {
            min: self.center - self.radius,
            max: self.center + self.radius,
        }
    }
}

impl AABB for AxisAlignedBox {
    fn aabb(&self) -> AxisAlignedBox {
        *self
    }
}

impl AABB for Shape {
    fn aabb(&self) -> AxisAlignedBox {
        match self {
            Shape::Sphere(s) => s.aabb(),
            Shape::AAB(a) => *a,
        }
    }
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

#[derive(Clone, Copy, Debug)]
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
        let t1 = (self.min - ray.origin) / ray.direction;
        let t2 = (self.max - ray.origin) / ray.direction;
        let (t1, t2) = (t1.min(&t2), t1.max(&t2));

        let tnear = tnear.max(t1.x).max(t1.y).max(t1.z);
        let tfar = tfar.min(t2.x).min(t2.y).min(t2.z);

        tnear < tfar
    }

    fn combine(&self, b: &AxisAlignedBox) -> AxisAlignedBox {
        AxisAlignedBox {
            min: self.min.min(&b.min),
            max: self.max.max(&b.max),
        }
    }

    fn center(&self) -> Vec3 {
        self.min + (self.max - self.min) / 2.
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Intersection {
    pub coord: Vec3,
    pub normal: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere(Sphere),
    AAB(AxisAlignedBox),
}

impl From<Sphere> for Shape {
    fn from(value: Sphere) -> Self {
        Shape::Sphere(value)
    }
}

impl Shape {
    fn intersection(&self, ray: &Ray) -> Option<Intersection> {
        match self {
            Shape::Sphere(s) => s.intersection(ray),
            Shape::AAB(a) => todo!(),
        }
    }
    fn center(&self) -> Vec3 {
        match self {
            Shape::Sphere(sphere) => sphere.center,
            Shape::AAB(aab) => aab.center(),
        }
    }
}

// Ok, so let's think how to code this guy.
//
// Querying:
// - there must be a tree which we walk. Every tree has self, optional (left, right).
// - if it's a leaf, then it also points to an object(id)
// -
// - left and right MIGHT be intersecting, and we must check both, and pick the earlier (in time) intersection.

mod bvh {
    use crate::{AxisAlignedBox, Intersection, Ray, Shape, AABB};

    #[derive(Clone, Copy, Debug, Default)]
    struct ShapeId(usize);

    #[derive(Clone, Copy, Debug, Default)]
    struct NodeId(usize);

    #[derive(Clone, Copy, Debug)]
    enum NodeKind {
        // Points to shape id
        Leaf(ShapeId),
        // Points to other nodes.
        Branch(NodeId, NodeId),
    }
    struct Node {
        aabb: AxisAlignedBox,
        kind: NodeKind,
    }

    pub struct BVH {
        objects: Vec<Shape>,
        nodes: Vec<Node>,
        root: NodeId,
    }

    impl core::fmt::Debug for BVH {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            struct NodeDebug<'a> {
                id: NodeId,
                bvh: &'a BVH,
            }

            impl<'a> core::fmt::Debug for NodeDebug<'a> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    let node = &self.bvh.nodes[self.id.0];
                    match node.kind {
                        NodeKind::Leaf(shape_id) => f
                            .debug_struct("Node")
                            .field("aabb", &node.aabb)
                            .field("shape", &self.bvh.objects[shape_id.0])
                            .finish(),
                        NodeKind::Branch(left, right) => f
                            .debug_struct("Node")
                            .field("aabb", &node.aabb)
                            .field(
                                "left",
                                &NodeDebug {
                                    bvh: self.bvh,
                                    id: left,
                                },
                            )
                            .field(
                                "right",
                                &NodeDebug {
                                    bvh: self.bvh,
                                    id: right,
                                },
                            )
                            .finish(),
                    }
                }
            }

            f.debug_struct("BVH")
                .field(
                    "nodes",
                    &NodeDebug {
                        id: self.root,
                        bvh: self,
                    },
                )
                .finish()
        }
    }

    impl BVH {
        pub fn new(objects: Vec<Shape>) -> Self {
            RecursiveBinarySplitBVHBuilder::build(objects)
        }

        pub fn intersection<'a>(&'a self, ray: &Ray) -> Option<(Intersection, &'a Shape)> {
            fn intersection<'a>(
                bvh: &'a BVH,
                ray: &Ray,
                node_id: NodeId,
            ) -> Option<(Intersection, &'a Shape)> {
                let node = bvh.nodes.get(node_id.0)?;
                if !node.aabb.intersects_branchless(ray) {
                    println!(
                        "no intersection of {:?} with {:?}, aabb: {:?}",
                        ray, node_id, node.aabb
                    );
                    return None;
                }
                fn filter_by_normal(ray: &Ray, intersection: Intersection) -> Option<Intersection> {
                    if ray.direction.dot(&intersection.normal) < 0. {
                        Some(intersection)
                    } else {
                        None
                    }
                }
                match node.kind {
                    NodeKind::Leaf(shape_id) => {
                        let shape = &bvh.objects[shape_id.0];
                        let i = shape
                            .intersection(ray)
                            .and_then(|i| filter_by_normal(ray, i))?;
                        Some((i, shape))
                    }
                    NodeKind::Branch(left, right) => {
                        let left = intersection(bvh, ray, left);
                        let right = intersection(bvh, ray, right);
                        match (left, right) {
                            (None, None) => None,
                            (None, Some(r)) | (Some(r), None) => Some(r),
                            (Some(l), Some(r)) => {
                                // pick the earlier intersection
                                //
                                // not great algo, but whatever
                                let lmag = (l.0.coord - ray.origin).magnitude();
                                let rmag = (r.0.coord - ray.origin).magnitude();
                                if lmag < rmag {
                                    Some(l)
                                } else {
                                    Some(r)
                                }
                            }
                        }
                    }
                }
            }
            intersection(self, ray, self.root)
        }
    }

    struct RecursiveBinarySplitBVHBuilder {}

    impl RecursiveBinarySplitBVHBuilder {
        fn build(objects: Vec<Shape>) -> BVH {
            let mut bvh = BVH {
                objects,
                nodes: Vec::new(),
                root: NodeId(0),
            };
            let object_ids: Vec<usize> = (0..(bvh.objects.len())).collect();

            fn aabb(bvh: &BVH, object_ids: &[usize]) -> AxisAlignedBox {
                object_ids
                    .iter()
                    .copied()
                    .map(|id| bvh.objects[id].aabb())
                    .reduce(|a, b| a.combine(&b))
                    .unwrap()
            }

            enum Axis {
                X,
                Y,
                Z,
            }

            fn find_longest_axis(aabb: &AxisAlignedBox) -> Axis {
                let x = aabb.max.x - aabb.min.x;
                let y = aabb.max.y - aabb.min.y;
                let z = aabb.max.z - aabb.min.z;
                if x > y && x > z {
                    Axis::X
                } else if y > z {
                    Axis::Y
                } else {
                    Axis::Z
                }
            }

            fn build_recursive(bvh: &mut BVH, mut object_ids: Vec<usize>) -> NodeId {
                if object_ids.len() == 1 {
                    let id = object_ids[0];
                    let node_id = NodeId(bvh.nodes.len());
                    bvh.nodes.push(Node {
                        aabb: bvh.objects[object_ids[0]].aabb(),
                        kind: NodeKind::Leaf(ShapeId(id)),
                    });
                    return node_id;
                }
                let bb = aabb(bvh, &object_ids);
                let longest_axis = find_longest_axis(&bb);

                let get_center = |object_id: usize| {
                    let center = bvh.objects[object_id].center();
                    match longest_axis {
                        Axis::X => center.x,
                        Axis::Y => center.y,
                        Axis::Z => center.z,
                    }
                };

                object_ids.sort_unstable_by(|left, right| {
                    get_center(*left).total_cmp(&get_center(*right))
                });

                let mid = object_ids.len() / 2;
                let left = build_recursive(bvh, object_ids[0..mid].to_vec());
                let right = build_recursive(bvh, object_ids[mid..].to_vec());
                let node_id = NodeId(bvh.nodes.len());
                bvh.nodes.push(Node {
                    aabb: bb,
                    kind: NodeKind::Branch(left, right),
                });
                node_id
            }

            bvh.root = build_recursive(&mut bvh, object_ids);
            bvh
        }
    }
}

pub use bvh::BVH;

#[cfg(test)]
mod tests {
    use crate::{Shape, Sphere, Vec3, BVH};

    #[test]
    fn test_bvh_1_object_works() {
        let sphere = Shape::Sphere(Sphere {
            center: Vec3::new(0., 0., 5.),
            radius: 1.,
        });
        let bvh = BVH::new(vec![sphere]);
        let ray = crate::Ray {
            origin: Vec3::new(0., 0., 0.),
            direction: Vec3::new(0., 0., 1.),
        };
        let i = bvh.intersection(&ray);
        assert!(i.is_some());
        dbg!(i);
    }

    #[test]
    fn test_bvh_1_object_from_inside_doesnt_work() {
        let sphere = Shape::Sphere(Sphere {
            center: Vec3::new(0., 0., 0.),
            radius: 1.,
        });
        let bvh = BVH::new(vec![sphere]);
        let ray = crate::Ray {
            origin: Vec3::new(0., 0., 0.),
            direction: Vec3::new(0., 0., 1.),
        };
        let i = bvh.intersection(&ray);
        assert!(i.is_none());
    }

    #[test]
    fn test_bvh_multiple_objects() {
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
        let ray = {
            let mut ray = crate::Ray {
                origin: Vec3::new(4., 0., 0.),
                direction: Vec3::default(),
            };
            ray.direction = (Vec3::new(0., 0., 8.) - ray.origin).normalize();
            ray
        };
        let i = bvh.intersection(&ray);
        assert!(i.is_some());
        dbg!(i);
    }
}
