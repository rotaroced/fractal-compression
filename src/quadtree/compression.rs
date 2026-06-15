use crate::{
    MAX_COEF, N_DIMS, N_DIMS_SQRT,
    prelude::*,
    quadtree::{QuadtreeSettings, quadtree::Quadtree},
};
use ndarray::prelude::*;
use ordered_float::NotNan;
use r_tree::RTree;
use std::ops::Deref;

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

pub fn key(img: Array2<f32>) -> Option<[f64; N_DIMS]> {
    assert_eq!(img.dim().0, img.dim().1);
    if img.dim().0 >= N_DIMS_SQRT {
        normalize(scale_down(&img, (N_DIMS_SQRT, N_DIMS_SQRT)).view())
    } else {
        let mut v = [0.; N_DIMS];
        let _ = img.iter().enumerate().map(|(i, &x)| v[i] = x as f64);

        Some(v)
    }
}

#[inline]
pub fn make_quadtree(
    img: &Arr<f32>,
    range_block: RangeBlockLocation,
    s: QuadtreeSettings,
    level: usize,
    domainblocks_rtree: &Vec<RTree<usize, 16, 4>>,
) -> Quadtree<Transformation> {
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

    let _rb_vec = get_rangeblock(img, range_block).iter().collect::<Vec<_>>();

    let normalized = key(get_rangeblock(img, range_block).to_owned());

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
        .unwrap_or((
            (
                DomainBlockLocation {
                    size: (2 * range_block.size.0, 2 * range_block.size.1),
                    ..Default::default()
                },
                0.,
                0.,
            ),
            NotNan::try_from(f32::INFINITY).unwrap(),
        ));

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
        Quadtree::leaf(t)
    } else {
        Quadtree::leaf((
            t.0,
            0.,
            get_rangeblock(img, range_block).sum()
                / (range_block.size.0 * range_block.size.1) as f32,
        ))
    }
}

fn normalize(b: ArrayView2<f32>) -> Option<[f64; N_DIMS]> {
    assert_eq!(b.nrows() * b.ncols(), N_DIMS);
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
    let mut normalized = [0.; N_DIMS];

    let _ = b
        .iter()
        .enumerate()
        .map(|(i, x)| normalized[i] = (sign * (x - avg) / v.sqrt()) as f64);
    if normalized.iter().any(|x| x.is_nan()) {
        None
    } else {
        Some(normalized)
    }
}

// Crée l'arbre r* qui contient les blocs sources normalisés
pub(super) fn create_rtree(img: ArrayView2<f32>, level: usize) -> RTree<usize, 16, N_DIMS> {
    assert_eq!(img.dim().0 % (1 << (level + 1)), 0);
    assert_eq!(img.dim().1 % (1 << (level + 1)), 0);

    let number = 1 << level;
    let n = img.dim().0 / number;
    let m = img.dim().1 / number;

    let mut t = RTree::<usize, 16, N_DIMS>::default();

    for i in 0..number {
        for j in 0..number {
            let domain_block = img.slice(s![(i * n)..((i + 1) * n), (j * m)..((j + 1) * m)]);
            if let Some(p) = &key(domain_block.to_owned()) {
                t.insert(p, i * number + j);
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
