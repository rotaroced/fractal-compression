use super::variance;
use crate::{
    prelude::*,
    quadtree::{QuadtreeSettings, quadtree::Quadtree},
};
use indicatif::{ParallelProgressIterator, ProgressFinish, ProgressStyle};
use ndarray::prelude::*;
use rayon::prelude::*;

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

    let contrast = if var.abs() < 1e-15 {
        0.
    } else {
        (cov / var).clamp(-0.99, 0.99)
    };
    let brightness = domain_block.mean().unwrap() - contrast * range_block.mean().unwrap();

    (contrast, brightness)
}
fn find_best_domain_block(img: &Arr<f32>, rb: RangeBlockLocation) -> (Transformation, f32) {
    let mut best_block = (DomainBlockLocation::default(), 0., 0.);
    let mut best_dist = f32::INFINITY;
    let range_block = get_rangeblock(img, rb);

    for i in 0..(img.nrows() / rb.size.0 / 2) {
        for j in 0..(img.ncols() / rb.size.1 / 2) {
            let db = DomainBlockLocation {
                flipped: false,
                rotation: Rotation::Zero,
                size: (rb.size.0 * 2, rb.size.1 * 2),
                pos: (i * rb.size.0 * 2, j * rb.size.1 * 2),
            };
            // println!("{:?}, {i}, {j}", db);
            let arr = get_domainblock(img, db);
            // println!("{:?} {:?}", db.arr, rb);
            if arr.dim().0 > rb.size.0
                && arr.dim().1 > rb.size.1
                && arr.dim().0 <= 2 * rb.size.0
                && arr.dim().1 <= 2 * rb.size.1
            {
                let domain_block = scale_down(&arr.to_owned(), rb.size);
                let (c, b) = find_brightness_and_contrast(range_block, &domain_block);
                let d = distance(range_block, (domain_block * c + b).view());

                if d < best_dist {
                    best_block = (db, c, b);
                    best_dist = d;
                }
            }
        }
    }

    (best_block, best_dist)
}

pub fn tl(b: RangeBlockLocation) -> RangeBlockLocation {
    RangeBlockLocation {
        size: (b.size.0 / 2, b.size.1 / 2),
        pos: b.pos,
    }
}
pub fn bl(b: RangeBlockLocation) -> RangeBlockLocation {
    RangeBlockLocation {
        size: (b.size.0 / 2, b.size.1 / 2),
        pos: (b.pos.0, b.pos.1 + b.size.1 / 2),
    }
}
pub fn tr(b: RangeBlockLocation) -> RangeBlockLocation {
    RangeBlockLocation {
        size: (b.size.0 / 2, b.size.1 / 2),
        pos: (b.pos.0 + b.size.0 / 2, b.pos.1),
    }
}
pub fn br(b: RangeBlockLocation) -> RangeBlockLocation {
    RangeBlockLocation {
        size: (b.size.0 / 2, b.size.1 / 2),
        pos: (b.pos.0 + b.size.0 / 2, b.pos.1 + b.size.1 / 2),
    }
}

#[inline]
pub fn make_quadtree(
    img: &Arr<f32>,
    range_block: RangeBlockLocation,
    s: QuadtreeSettings,
    level: usize,
) -> Quadtree<Transformation> {
    debug_assert!(img.nrows() % (1 << s.maximum_range_splits) == 0);
    debug_assert!(img.ncols() % (1 << s.maximum_range_splits) == 0);

    if level < s.minimum_range_splits {
        return Quadtree::node(
            make_quadtree(img, tl(range_block), s, level + 1),
            make_quadtree(img, tr(range_block), s, level + 1),
            make_quadtree(img, bl(range_block), s, level + 1),
            make_quadtree(img, br(range_block), s, level + 1),
        );
    }

    let (t, dist) = find_best_domain_block(img, range_block);

    if dist >= s.max_distance && level < s.maximum_range_splits {
        Quadtree::node(
            make_quadtree(img, tl(range_block), s, level + 1),
            make_quadtree(img, tr(range_block), s, level + 1),
            make_quadtree(img, bl(range_block), s, level + 1),
            make_quadtree(img, br(range_block), s, level + 1),
        )
    } else {
        println!("{:?}", range_block);
        Quadtree::leaf(t)
    }
}
