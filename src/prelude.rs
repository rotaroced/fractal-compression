use core::ops::{Add, Div};
use ndarray::*;
use num_traits::{FromPrimitive, Zero};
use std::collections::HashMap;

pub type Arr<A> = Array2<A>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default, PartialOrd, Ord)]
pub enum Rotation {
    #[default]
    Zero,
    Quarter,
    Half,
    ThreeQuarter,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct DomainBlockLocation {
    pub pos: (usize, usize),
    pub rotation: Rotation,
    pub flipped: bool,
    pub size: (usize, usize),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct RangeBlockLocation {
    pub pos: (usize, usize),
    pub size: (usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainBlock<'a> {
    pub location: DomainBlockLocation,
    pub arr: ArrayView2<'a, f32>,
}

pub type Transformation = (DomainBlockLocation, f32, f32);

pub type Mappings = HashMap<RangeBlockLocation, Transformation>;

pub fn scale_down<T>(img: &Arr<T>, target_size: (usize, usize)) -> Arr<T>
where
    T: Clone + FromPrimitive + Add<Output = T> + Div<Output = T> + Zero,
{
    // println!("{:?} -> {:?}", img.dim(), target_size);
    let mut a = Arr::<T>::zeros(target_size);
    for i in 0..target_size.0 {
        for j in 0..target_size.1 {
            let (x, y) = (
                img.dim().0 * i / target_size.0,
                img.dim().1 * j / target_size.1,
            );
            let (ex, ey) = (
                img.dim().0 * (i + 1) / target_size.0,
                img.dim().1 * (j + 1) / target_size.1,
            );
            a[(i, j)] = img.slice(s![x..ex, y..ey]).mean().unwrap();
        }
    }

    a
}

/// computes the distance between two images
pub fn distance(img1: ArrayView2<f32>, img2: ArrayView2<f32>) -> f32 {
    debug_assert!(img1.dim() == img2.dim());
    let size: f32 = (img1.dim().0 * img1.dim().1) as f32;
    img1.iter()
        .zip(img2.iter())
        .map(|(a, b)| (a - b) * (a - b) / size)
        .fold(0., |a, b| a + b)
        .sqrt()
}

pub fn get_rangeblock(img: &Arr<f32>, range_block: RangeBlockLocation) -> ArrayView2<f32> {
    // println!(
    //     "{}..{} , {}..{}",
    //     block.pos.0,
    //     block.pos.0 + block.size.0,
    //     block.pos.1,
    //     block.pos.1 + block.size.1
    // );
    // println!("{:?}", img.dim());
    img.slice(s![
        range_block.pos.0..(range_block.size.0 + range_block.pos.0),
        (range_block.pos.1)..(range_block.size.1 + range_block.pos.1)
    ])
}

#[inline]
pub fn get_domainblock(img: &Arr<f32>, domain_block: DomainBlockLocation) -> ArrayView2<f32> {
    get_rangeblock(
        img,
        RangeBlockLocation {
            pos: domain_block.pos,
            size: domain_block.size,
        },
    )
}

pub fn rotate_block(img: ArrayView2<f32>, rot: Rotation) -> Arr<f32> {
    let (h, w) = img.dim();

    match rot {
        Rotation::Zero => img.to_owned(),
        Rotation::Quarter => Arr::<f32>::from_shape_fn((w, h), |(i, j)| img[(h - j - 1, i)]),
        Rotation::ThreeQuarter => Arr::<f32>::from_shape_fn((w, h), |(i, j)| img[(j, w - i - 1)]),
        Rotation::Half => Arr::<f32>::from_shape_fn((h, w), |(i, j)| img[(h - i - 1, w - j - 1)]),
    }
}
