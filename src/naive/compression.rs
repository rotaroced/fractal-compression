use core::f32;

use crate::prelude::*;
use ndarray::*;
use rayon::{
    self,
    iter::{IntoParallelRefIterator, ParallelIterator},
};

use crate::quadtree::variance;
pub type Arr<A> = Array2<A>;

/// calcul de la luminosité et du contraste optimaux (régression linéaire pour la norme 2)
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

// Trouve la meilleure transformation associée à un bloc destination
fn find_best_domain_block(
    img: &Arr<f32>,
    rb: RangeBlockLocation,
    dbs: &[DomainBlock],
) -> (DomainBlockLocation, f32, f32) {
    let mut best_block = (DomainBlockLocation::default(), 0., 0.);
    let mut best_dist = f32::INFINITY;
    let range_block = get_rangeblock(img, rb);

    for &db in dbs {
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
        .iter()
        .map(|&rb| (rb, find_best_domain_block(img, rb, domain_blocks)))
        .collect::<Mappings>()
}

#[inline]
pub fn find_mappings_parallel(
    img: &Arr<f32>,
    range_blocks: &[RangeBlockLocation],
    domain_blocks: &[DomainBlock],
) -> Mappings {
    range_blocks
        .par_iter()
        .map(|&rb| (rb, find_best_domain_block(img, rb, domain_blocks)))
        .collect::<Mappings>()
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
