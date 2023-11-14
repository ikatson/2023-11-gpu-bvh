use std::ops::{Add, Div, Mul, Sub};
use zerocopy_derive::AsBytes;

#[derive(Clone, Copy, Default, Debug, AsBytes)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn nearest_aabb_is_left(&self, left: &AxisAlignedBox, right: &AxisAlignedBox) -> bool {
        let tmin = (left.min - self.origin).dot(&self.direction);
        let tmax = (left.max - self.origin).dot(&self.direction);
        let tmin2 = (right.min - self.origin).dot(&self.direction);
        let tmax2 = (right.max - self.origin).dot(&self.direction);
        tmin > tmax2 && tmin2 > tmax
    }
}

impl From<f32> for Vec3 {
    fn from(value: f32) -> Self {
        Self::new(value, value, value)
    }
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Self { x, y, z }
    }

    pub fn normalize(&self) -> Self {
        let len = self.magnitude();
        self.scalar_div(len)
    }
    pub fn scalar_mult(&self, number: f32) -> Self {
        Self::new(self.x * number, self.y * number, self.z * number)
    }
    pub fn scalar_div(&self, number: f32) -> Self {
        Self::new(self.x / number, self.y / number, self.z / number)
    }
    pub fn squared_magnitude(&self) -> f32 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }
    pub fn magnitude(&self) -> f32 {
        self.squared_magnitude().sqrt()
    }
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn min(&self, other: &Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    pub fn max(&self, other: &Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    pub fn abs(&self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub fn rotate_around_axis(&self, axis: &Vec3, angle_rad: f32) -> Vec3 {
        let cos_theta = angle_rad.cos();
        let sin_theta = angle_rad.sin();
        let one_minus_cos_theta = 1.0 - cos_theta;

        // Rodrigues' rotation formula
        let new_x = self.x * (cos_theta + axis.x * axis.x * one_minus_cos_theta)
            + self.y * (axis.x * axis.y * one_minus_cos_theta - axis.z * sin_theta)
            + self.z * (axis.x * axis.z * one_minus_cos_theta + axis.y * sin_theta);

        let new_y = self.x * (axis.y * axis.x * one_minus_cos_theta + axis.z * sin_theta)
            + self.y * (cos_theta + axis.y * axis.y * one_minus_cos_theta)
            + self.z * (axis.y * axis.z * one_minus_cos_theta - axis.x * sin_theta);

        let new_z = self.x * (axis.z * axis.x * one_minus_cos_theta - axis.y * sin_theta)
            + self.y * (axis.z * axis.y * one_minus_cos_theta + axis.x * sin_theta)
            + self.z * (cos_theta + axis.z * axis.z * one_minus_cos_theta);

        Vec3 {
            x: new_x,
            y: new_y,
            z: new_z,
        }
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
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<f32> for Vec3 {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        Vec3::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl Sub<f32> for Vec3 {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        Vec3::new(self.x - rhs, self.y - rhs, self.z - rhs)
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
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
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
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

#[derive(Clone, Copy, Debug, Default, zerocopy_derive::AsBytes)]
// WGPU: align=16, size=16
#[repr(C)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}
impl Sphere {
    pub fn new(center: Vec3, radius: f32) -> Sphere {
        Sphere { center, radius }
    }
    fn intersection(&self, ray: &Ray) -> Option<Intersection> {
        let sphere = *self;
        // sphere ray intersection
        // https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        let oc = Vec3::new(
            ray.origin.x - sphere.center.x,
            ray.origin.y - sphere.center.y,
            ray.origin.z - sphere.center.z,
        );

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
        let coord = Vec3::new(
            ray.origin.x + t * ray.direction.x,
            ray.origin.y + t * ray.direction.y,
            ray.origin.z + t * ray.direction.z,
        );
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

#[derive(Clone, Copy, Debug, Default)]
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
        self.tnear(ray).is_some()
    }

    pub fn tnear(&self, ray: &Ray) -> Option<f32> {
        let tnear = f32::NEG_INFINITY;
        let tfar = f32::INFINITY;
        let t1 = (self.min - ray.origin) / ray.direction;
        let t2 = (self.max - ray.origin) / ray.direction;
        let (t1, t2) = (t1.min(&t2), t1.max(&t2));

        let tnear = tnear.max(t1.x).max(t1.y).max(t1.z);
        let tfar = tfar.min(t2.x).min(t2.y).min(t2.z);

        if tnear < tfar {
            Some(tnear)
        } else {
            None
        }
    }

    pub fn intersects_other_aabb(&self, other: &Self) -> bool {
        fn axis_intersects(a_min: f32, a_max: f32, b_min: f32, b_max: f32) -> bool {
            a_min.max(b_min) <= a_max.min(b_max)
        }
        macro_rules! axis {
            ($axis:tt) => {
                axis_intersects(
                    self.min.$axis,
                    self.max.$axis,
                    other.min.$axis,
                    other.max.$axis,
                )
            };
        }
        axis!(x) && axis!(y) && axis!(z)
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
            Shape::AAB(_a) => todo!(),
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
    use wgpu::{util::DeviceExt, BindGroupEntry, BindGroupLayoutEntry, BufferUsages, ShaderStages};

    use crate::{AxisAlignedBox, Intersection, Ray, Shape, Vec3, AABB};

    #[derive(Clone, Copy, Debug, Default)]
    struct ShapeId(usize);

    #[derive(Clone, Copy, Debug, Default)]
    struct NodeId(usize);

    #[derive(Clone, Copy, Debug)]
    enum NodeKind {
        // Points to shape id
        Leaf(ShapeId),
        // Points to other nodes.
        Branch {
            left: NodeId,
            right: NodeId,
            overlaps: bool,
        },
    }

    impl Default for NodeKind {
        fn default() -> Self {
            NodeKind::Leaf(ShapeId(0))
        }
    }

    #[derive(Default)]
    struct Node {
        aabb: AxisAlignedBox,
        kind: NodeKind,
    }

    #[derive(Default)]
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
                        NodeKind::Branch {
                            left,
                            right,
                            overlaps,
                        } => f
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
                            .field("overlaps", &overlaps)
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

        pub fn root_aabb(&self) -> AxisAlignedBox {
            self.get(self.root).unwrap().aabb
        }

        pub fn intersection_brute_force<'a>(
            &'a self,
            ray: &Ray,
        ) -> Option<(Intersection, &'a Shape)> {
            self.objects
                .iter()
                .map(|o| {
                    // Uncomment if you want to test AABB first.
                    //
                    // if o.aabb().intersects_vectors(ray) {
                    o.intersection(ray).map(|i| (i, o))
                    // } else {
                    //     None
                    // }
                })
                .reduce(|l, r| {
                    match (l, r) {
                        (None, None) => None,
                        (None, Some(r)) | (Some(r), None) => Some(r),
                        (Some(l), Some(r)) => {
                            // pick the earlier intersection
                            //
                            // not great algo, but whatever
                            let lmag = (l.0.coord - ray.origin).squared_magnitude();
                            let rmag = (r.0.coord - ray.origin).squared_magnitude();
                            if lmag < rmag {
                                Some(l)
                            } else {
                                Some(r)
                            }
                        }
                    }
                })
                .unwrap()
        }

        fn get(&self, id: NodeId) -> Option<&Node> {
            self.nodes.get(id.0)
        }

        pub fn intersection<'a>(&'a self, ray: &Ray) -> Option<(Intersection, &'a Shape)> {
            fn get_intersecting_node<'a>(ray: &Ray, node: &'a Node) -> Option<(&'a Node, f32)> {
                node.aabb.tnear(ray).map(|tnear| (node, tnear))
            }

            fn filter_by_normal(ray: &Ray, intersection: Intersection) -> Option<Intersection> {
                if ray.direction.dot(&intersection.normal) < 0. {
                    Some(intersection)
                } else {
                    None
                }
            }

            fn traverse<'a>(
                bvh: &'a BVH,
                ray: &Ray,
                intersecting_node: &Node,
            ) -> Option<(Intersection, &'a Shape)> {
                let node = intersecting_node;
                match node.kind {
                    NodeKind::Leaf(shape_id) => {
                        let shape = &bvh.objects[shape_id.0];
                        let i = shape
                            .intersection(ray)
                            .and_then(|i| filter_by_normal(ray, i))?;
                        Some((i, shape))
                    }
                    NodeKind::Branch {
                        left,
                        right,
                        overlaps,
                    } => {
                        let trav = |n| traverse(bvh, ray, n);
                        let get = |id| get_intersecting_node(ray, bvh.get(id)?);

                        // Fast path - only one of left or right (or none) matched.
                        let ((left, lnear), (right, rnear)) = match (get(left), get(right)) {
                            (None, n) | (n, None) => return n.and_then(|(n, _)| trav(n)),
                            (Some(l), Some(r)) => (l, r),
                        };

                        // Fast path, left and right AABBs don't intersect.
                        // Traverse nearest first, first hit returns.
                        if !overlaps {
                            let (near, far) = if lnear < rnear {
                                (left, right)
                            } else {
                                (right, left)
                            };
                            return trav(near).or_else(|| trav(far));
                        }

                        // Slow path - traverse both sides. If both hit, return nearest intersection.
                        trav(left).into_iter().chain(trav(right)).reduce(|l, r| {
                            let lmag = (l.0.coord - ray.origin).squared_magnitude();
                            let rmag = (r.0.coord - ray.origin).squared_magnitude();
                            if lmag < rmag {
                                l
                            } else {
                                r
                            }
                        })
                    }
                }
            }
            let (root, _) = get_intersecting_node(ray, self.get(self.root)?)?;
            traverse(self, ray, root)
        }
    }

    struct RecursiveBinarySplitBVHBuilder {}

    impl RecursiveBinarySplitBVHBuilder {
        fn build(objects: Vec<Shape>) -> BVH {
            let mut bvh = BVH {
                objects,
                ..Default::default()
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

            #[derive(Debug)]
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

            fn alloc(bvh: &mut BVH) -> NodeId {
                let id = NodeId(bvh.nodes.len());
                bvh.nodes.push(Node::default());
                id
            }

            fn build_recursive(bvh: &mut BVH, current_node_id: NodeId, mut object_ids: Vec<usize>) {
                if object_ids.len() == 1 {
                    let id = object_ids[0];
                    bvh.nodes[current_node_id.0] = Node {
                        aabb: bvh.objects[object_ids[0]].aabb(),
                        kind: NodeKind::Leaf(ShapeId(id)),
                    };
                    return;
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
                let (left, right) = (alloc(bvh), alloc(bvh));
                build_recursive(bvh, left, object_ids[0..mid].to_vec());
                build_recursive(bvh, right, object_ids[mid..].to_vec());
                let overlaps = bvh
                    .get(left)
                    .unwrap()
                    .aabb
                    .intersects_other_aabb(&bvh.get(right).unwrap().aabb);
                bvh.nodes[current_node_id.0] = Node {
                    aabb: bb,
                    kind: NodeKind::Branch {
                        left,
                        right,
                        overlaps,
                    },
                };
            }

            let root = alloc(&mut bvh);
            build_recursive(&mut bvh, root, object_ids);
            bvh
        }
    }

    // struct BVHOptimizer {}

    // impl BVHOptimizer {
    //     pub fn optimize(bvh: &BVH) -> BVH {
    //         // Traverse the bvh in DFS
    //     }
    // }

    const FLAG_IS_LEAF: u32 = 1;
    const FLAG_OVERLAPS: u32 = 2;

    #[derive(Debug, Default, zerocopy_derive::AsBytes)]
    // WGSL:
    #[repr(C)]
    struct GPUBVHNode {
        // aabb min
        min: Vec3,
        // If branch, id1 is left, id2 is right.
        id1: u32,
        // aabb max
        max: Vec3,
        id2: u32,

        flags: u32,
        _pad_struct: [u8; 12],
    }

    pub struct GPUBVH {
        // objects: Vec<Sphere>,
        // nodes: Vec<GPUBVHNode>,
        pub objects_buf: wgpu::Buffer,
        pub nodes_buf: wgpu::Buffer,
        pub meta_buf: wgpu::Buffer,
        pub bgl: wgpu::BindGroupLayout,
        pub bind_group: wgpu::BindGroup,
    }

    impl GPUBVH {
        pub fn new(bvh: &BVH, device: &wgpu::Device) -> Self {
            use zerocopy::AsBytes;

            let objects = bvh
                .objects
                .iter()
                .map(|s| match s {
                    Shape::Sphere(s) => *s,
                    _ => unimplemented!(),
                })
                .collect::<Vec<_>>();
            let nodes = bvh
                .nodes
                .iter()
                .map(|n| {
                    let mut result = GPUBVHNode {
                        min: n.aabb.min,
                        max: n.aabb.max,
                        ..Default::default()
                    };
                    match n.kind {
                        NodeKind::Leaf(shape_id) => {
                            result.flags |= FLAG_IS_LEAF;
                            result.id1 = shape_id.0 as u32;
                        }
                        NodeKind::Branch {
                            left,
                            right,
                            overlaps,
                        } => {
                            result.flags |= if overlaps { FLAG_OVERLAPS } else { 0 };
                            result.id1 = left.0 as u32;
                            result.id2 = right.0 as u32;
                        }
                    }
                    result
                })
                .collect::<Vec<_>>();
            let objects = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: objects.as_bytes(),
                usage: BufferUsages::STORAGE,
            });
            let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: nodes.as_bytes(),
                usage: BufferUsages::STORAGE,
            });

            #[derive(zerocopy_derive::AsBytes)]
            #[repr(C)]
            struct BVHMeta {
                root_id: u32,
            }
            let meta = BVHMeta {
                root_id: bvh.root.0 as u32,
            };
            let meta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: meta.as_bytes(),
                usage: BufferUsages::UNIFORM,
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
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

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: objects.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: nodes.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: meta.as_entire_binding(),
                    },
                ],
            });

            Self {
                objects_buf: objects,
                nodes_buf: nodes,
                meta_buf: meta,
                bgl,
                bind_group,
            }
        }
    }
}

pub use bvh::{BVH, GPUBVH};

#[cfg(test)]
mod tests {
    use crate::{Shape, Sphere, Vec3, BVH};

    #[test]
    fn test_bvh_1_object_works() {
        let sphere = Shape::Sphere(Sphere::new(Vec3::new(0., 0., 5.), 1.));
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
        let sphere = Shape::Sphere(Sphere::new(Vec3::new(0., 0., 0.), 1.));
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
            Shape::Sphere(Sphere::new(Vec3::new(0., 0., 0.), 1.)),
            Shape::Sphere(Sphere::new(Vec3::new(0., 0., 2.), 1.)),
            Shape::Sphere(Sphere::new(Vec3::new(0., 0., 4.), 1.)),
            Shape::Sphere(Sphere::new(Vec3::new(0., 0., 6.), 1.)),
            Shape::Sphere(Sphere::new(Vec3::new(0., 0., 8.), 1.)),
        ];
        let bvh = BVH::new(shapes);
        let ray = {
            let mut ray = crate::Ray {
                // starts "inside" the first sphere. Should hit it
                origin: Vec3::new(0.5, 0., 0.),
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
