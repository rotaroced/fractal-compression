use super::variance;
use crate::{
    MAX_COEF, N_DIMS_SQRT,
    prelude::*,
    quadtree::{QuadtreeSettings, quadtree::Quadtree},
};
use core::num;
use indicatif::{ParallelProgressIterator, ProgressFinish, ProgressStyle};
use ndarray::prelude::*;
use ordered_float::NotNan;
use r_tree::RTree;
use rayon::prelude::*;
use std::{f32::INFINITY, ops::Deref, rc::Rc};

/// calcul de la luminosité et du contraste optimaux (avec la méthode des moindres carrés)
pub fn find_brightness_and_contrast(
    range_block: ArrayView2<f32>,
    domain_block: &ArrayView2<f32>,
) -> (f32, f32) {
    debug_assert_eq!(range_block.dim(), domain_block.dim());
    let er = range_block.mean().unwrap();
    let dr = domain_block.mean().unwrap();

    let cov = range_block
        .iter()
        .zip(domain_block.iter())
        .map(|(&a, &b)| (a - er) * (b - dr))
        .fold(0., |a, b| a + b);

    let var = domain_block
        .iter()
        .map(|x| (x - dr) * (x - dr))
        .fold(0., |a, b| a + b);

    // TODO: rendre ça mieux
    let contrast = if var.abs() < 1e-15 {
        0.
    } else {
        (cov / var).clamp(-MAX_COEF, MAX_COEF)
    };
    let brightness = (er - contrast * dr).clamp(-MAX_COEF, MAX_COEF);

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
            let arr = get_domainblock(img, db);
            if arr.dim().0 > rb.size.0
                && arr.dim().1 > rb.size.1
                && arr.dim().0 <= 2 * rb.size.0
                && arr.dim().1 <= 2 * rb.size.1
            {
                let domain_block = scale_down(&arr.to_owned(), rb.size);
                let (c, b) = find_brightness_and_contrast(range_block, &domain_block.view());
                let d = distance(range_block, (domain_block.clone() * c + b).view());

                // println!(
                //     "{:?}, {i}, {j}, {}, c={}, b={}, \n{}\n{}\n{}\n\n",
                //     db,
                //     d,
                //     c,
                //     b,
                //     range_block,
                //     domain_block.clone(),
                //     domain_block * c + b
                // );
                //
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

pub fn key(img: Array2<f64>) -> Vec<f64> {
    assert_eq!(img.dim().0, img.dim().1);
    if img.dim().0 >= N_DIMS_SQRT {
        scale_down(&img, (N_DIMS_SQRT, N_DIMS_SQRT)).into_raw_vec()
    } else {
        let mut v = img.into_raw_vec();

        while v.len() < N_DIMS_SQRT * N_DIMS_SQRT {
            v.push(0.0);
        }

        v
    }
}

#[inline]
pub fn make_quadtree(
    img: &Arr<f32>,
    range_block: RangeBlockLocation,
    s: QuadtreeSettings,
    level: usize,
    domainblocks_rtree: &Vec<RTree<usize, 16>>,
) -> Quadtree<Transformation> {
    println!("{:?}", range_block);
    debug_assert_eq!(img.nrows() % (1 << s.maximum_range_splits), 0);
    debug_assert_eq!(img.ncols() % (1 << s.maximum_range_splits), 0);

    if level <= s.minimum_range_splits {
        return Quadtree::node(
            make_quadtree(img, tl(range_block), s, level + 1, domainblocks_rtree),
            make_quadtree(img, tr(range_block), s, level + 1, domainblocks_rtree),
            make_quadtree(img, bl(range_block), s, level + 1, domainblocks_rtree),
            make_quadtree(img, br(range_block), s, level + 1, domainblocks_rtree),
        );
    }

    let rb_vec = get_rangeblock(img, range_block).iter().collect::<Vec<_>>();

    let normalized = normalize(get_rangeblock(img, range_block));

    // TODO : make it so there is no panic if `domainblocks_rtree` is empty.
    let closest = if let Some(n) = &normalized {
        domainblocks_rtree[level - s.minimum_range_splits - 1].k_closest(n, s.max_neighbors)
    } else {
        vec![]
    };
    let dist_const = dist_from_constant(get_rangeblock(img, range_block));

    // variables needed to reconstruct the domain blocks.
    let number = 1 << (level - 1);
    let size = (img.dim().0 / number, img.dim().1 / number);

    let (t, dist): (Transformation, NotNan<f32>) = closest
        .iter()
        .map(|(_, _, index)| {
            let db = DomainBlockLocation {
                pos: (
                    size.0 * (index.deref() / number),
                    size.1 * (index.deref() % number),
                ),
                rotation: Rotation::Zero,
                flipped: false,
                size,
            };
            let domain_block = scale_down(&get_domainblock(img, db).to_owned(), range_block.size);
            let (c, b) = find_brightness_and_contrast(
                get_rangeblock(img, range_block),
                &domain_block.view(),
            );
            let d = NotNan::try_from(distance(
                get_rangeblock(img, range_block),
                (domain_block * c + b).view(),
            ))
            .unwrap();
            ((db, c, b), d)
        })
        .min_by_key(|(_, d)| *d)
        .unwrap_or((Default::default(), NotNan::try_from(f32::INFINITY).unwrap()));

    // println!(
    //     "{dist:?} {dist_const:?}, actual best dist = {}",
    //     find_best_domain_block(img, range_block).1
    // );

    if dist.into_inner() > s.max_distance
        && dist_const > s.max_distance
        && level - 1 < s.maximum_range_splits
    {
        Quadtree::node(
            make_quadtree(img, tl(range_block), s, level + 1, domainblocks_rtree),
            make_quadtree(img, tr(range_block), s, level + 1, domainblocks_rtree),
            make_quadtree(img, bl(range_block), s, level + 1, domainblocks_rtree),
            make_quadtree(img, br(range_block), s, level + 1, domainblocks_rtree),
        )
    } else if dist_const > dist.into_inner() {
        // println!("choosing other");
        Quadtree::leaf(t)
    } else {
        // println!("choosing constant");
        Quadtree::leaf((
            t.0,
            0.,
            get_rangeblock(img, range_block).sum()
                / (range_block.size.0 * range_block.size.1) as f32,
        ))
    }
}

fn normalize(block: ArrayView2<f32>) -> Option<Vec<f64>> {
    let b = block;
    let avg = b.sum() / (b.dim().0 * b.dim().1) as f32;
    let sign: f32 = b.iter().fold(None, |o, &x| {
        o.or(if x > avg {
            Some(1.)
        } else if x < avg {
            Some(-1.)
        } else {
            None
        })
    })?;

    let v =
        ((b.to_owned() - avg) * (b.to_owned() - avg)).sum() / (b.dim().0 * b.dim().1) as f32 * 500.;
    let normalized = b.map(|x| (sign * (x - avg) / v.sqrt()) as f64);

    if normalized.iter().any(|x| x.is_nan()) {
        None
    } else {
        Some(key(normalized))
    }
}

pub(super) fn create_rtree(img: ArrayView2<f32>, level: usize) -> RTree<usize, 16> {
    println!(
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB {:?}",
        level
    );
    assert_eq!(img.dim().0 % (1 << (level + 1)), 0);
    assert_eq!(img.dim().1 % (1 << (level + 1)), 0);

    let number = 1 << level;
    let n = img.dim().0 / number;
    let m = img.dim().1 / number;

    let mut t = RTree::<usize, 16>::new(n * m);

    for i in 0..number {
        for j in 0..number {
            println!("{:?}", (i, j));
            let domain_block = img.slice(s![(i * n)..((i + 1) * n), (j * m)..((j + 1) * m)]);
            println!("{:?}", (i, j));
            // println!(
            //     "{:?}\n",
            //     normalize(scale_down(&domain_block.to_owned(), (n / 2, m / 2)).view())
            // );
            if let Some(p) = &normalize(scale_down(&domain_block.to_owned(), (n / 2, m / 2)).view())
            {
                println!("{:?}", (i, j));
                t.insert(p, i * number + j);
                println!("{:?}", (i, j));
            }
        }
    }

    t
}

fn dist_from_constant(img: ArrayView2<f32>) -> f32 {
    let avg = img.sum() / (img.dim().0 * img.dim().1) as f32;

    (((img.to_owned() - avg) * (img.to_owned() - avg)).sum() / (img.dim().0 * img.dim().1) as f32)
        .sqrt()
}
