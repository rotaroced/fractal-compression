use core::f32;
use std::collections::HashMap;

use indicatif::{ParallelProgressIterator, ProgressFinish, ProgressStyle};
use ndarray::*;
use rayon::{
    self,
    iter::{IntoParallelRefIterator, ParallelIterator},
};

use crate::quadtree::variance;
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

/// computes the distance between two images
pub fn distance(img1: ArrayView2<f32>, img2: ArrayView2<f32>) -> f32 {
    debug_assert!(img1.dim() == img2.dim());
    img1.iter()
        .zip(img2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .fold(0., |a, b| a + b)
        .sqrt()
}

pub fn scale_down(img: &Arr<f32>, target_size: (usize, usize)) -> Arr<f32> {
    // println!("{:?} -> {:?}", img.dim(), target_size);
    let mut a = Arr::<f32>::zeros(target_size);
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

/// calcul de la luminosité et du contraste optimaux (avec la méthode des moindres carrés)
pub fn find_brightness_and_contrast(
    range_block: ArrayView2<f32>,
    domain_block: &Arr<f32>,
) -> (f32, f32) {
    debug_assert_eq!(range_block.dim(), domain_block.dim());
    let er = range_block.mean().unwrap();
    let dr = domain_block.mean().unwrap();

    let cov = range_block
        .iter()
        .zip(domain_block.iter())
        .map(|(&a, &b)| (a - er) * (b - dr))
        .fold(0., |a, b| a + b)
        / (range_block.dim().0 * range_block.dim().1) as f32;

    let var = variance(range_block);

    let contrast = if var.abs() < 1e-15 { 0. } else { cov / var };
    let brightness = domain_block.mean().unwrap() - contrast * range_block.mean().unwrap();

    (contrast, brightness)
}

fn get_rangeblock(img: &Arr<f32>, range_block: RangeBlockLocation) -> ArrayView2<f32> {
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
fn get_domainblock(img: &Arr<f32>, domain_block: DomainBlockLocation) -> ArrayView2<f32> {
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

fn find_best_domain_block(
    img: &Arr<f32>,
    rb: RangeBlockLocation,
    dbs: &[DomainBlock],
) -> (DomainBlockLocation, f32, f32) {
    let mut best_block = (DomainBlockLocation::default(), 0., 0.);
    let mut best_dist = f32::INFINITY;
    let range_block = get_rangeblock(img, rb);

    for &db in dbs {
        // println!("{:?} {:?}", db.arr, rb);
        if db.arr.dim().0 > rb.size.0
            && db.arr.dim().1 > rb.size.1
            && db.arr.dim().0 <= 4 * rb.size.0
            && db.arr.dim().1 <= 4 * rb.size.1
        {
            let domain_block = scale_down(&db.arr.to_owned(), rb.size);
            let (c, b) = find_brightness_and_contrast(range_block, &domain_block);
            let d = distance(range_block, (domain_block * c + b).view());

            if d < best_dist {
                best_block = (db.location, c, b);
                best_dist = d;
            }
        }
    }

    best_block
}
#[inline]
pub fn find_mappings(
    img: &Arr<f32>,
    range_blocks: &[RangeBlockLocation],
    domain_blocks: &[DomainBlock],
) -> Mappings {
    range_blocks
        .par_iter()
        .progress_count(range_blocks.len() as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template("{bar:150} {pos:>7}/{len:<7} = {percent_precise}% ({eta} remaining, {duration} in total)")
                .unwrap(),
        )
        .with_finish(ProgressFinish::AndLeave)
        .map(|&rb| (rb, find_best_domain_block(img, rb, domain_blocks)))
        .collect::<Mappings>()
}

pub fn reconstruct(mappings: Mappings, mut img: Arr<f32>, n: usize) -> Arr<f32> {
    let mut new_img = img.clone();
    for _ in 0..n {
        for (&rb, &(db, contrast, brightness)) in mappings.iter() {
            // println!("{rb:?}, {db:?}");
            let transformed_block = scale_down(
                &rotate_block(get_domainblock(&img, db), db.rotation),
                rb.size,
            ) * contrast
                + brightness;

            for i in 0..rb.size.0 {
                for j in 0..rb.size.1 {
                    // println!("{:?}", (rb.pos.0 + i, rb.pos.1 + j));
                    new_img[(rb.pos.0 + i, rb.pos.1 + j)] = transformed_block[(i, j)];
                }
            }
        }
        img = new_img.clone();
    }

    img
}

impl PartialOrd for DomainBlockLocation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            self.pos
                .partial_cmp(&other.pos)
                .unwrap()
                .then(self.rotation.cmp(&other.rotation))
                .then(self.flipped.cmp(&other.flipped)),
        )
    }
}
