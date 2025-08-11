pub mod io;
pub mod quadtree;

use super::compression::*;
use super::naive::{create_images, domain_blocks_from_locations};
use ndarray::{Array2, ArrayView2, s as sl};
use ndarray_image::save_gray_image;
use quadtree::*;

pub fn variance(img: ArrayView2<f32>) -> f32 {
    let m = img.mean().unwrap();
    img.iter()
        .map(|x| (x - m) * (x - m))
        .fold(0., <f32 as std::ops::Add<f32>>::add)
        / ((img.dim().0 * img.dim().1) as f32)
}

pub fn generate_range_blocks(
    img: ArrayView2<f32>,
    s: QuadtreeSettings,
    pos: (usize, usize),
    has_to_split: usize,
) -> Quadtree<RangeBlockLocation> {
    let v = variance(img);
    // println!("{v:?}");
    if img.dim().0 / 2 < s.min_range_block_size
        || img.dim().1 / 2 < s.min_range_block_size
        || (v <= s.min_range_variance && has_to_split == 0)
    {
        Quadtree::leaf(RangeBlockLocation {
            pos,
            size: img.dim(),
        })
    } else {
        let h = img.dim().0;
        let w = img.dim().1;

        let tl = generate_range_blocks(
            img.slice(sl![0..(h / 2), 0..(w / 2)]),
            s,
            pos,
            has_to_split.saturating_sub(1),
        );
        let tr = generate_range_blocks(
            img.slice(sl![0..(h / 2), (w / 2)..w]),
            s,
            (pos.0, pos.1 + w / 2),
            has_to_split.saturating_sub(1),
        );
        let bl = generate_range_blocks(
            img.slice(sl![(h / 2)..h, 0..(w / 2)]),
            s,
            (pos.0 + h / 2, pos.1),
            has_to_split.saturating_sub(1),
        );
        let br = generate_range_blocks(
            img.slice(sl![(h / 2)..h, (w / 2)..w]),
            s,
            (pos.0 + h / 2, pos.1 + w / 2),
            has_to_split.saturating_sub(1),
        );

        Quadtree::node(
            RangeBlockLocation {
                pos,
                size: img.dim(),
            },
            tl,
            tr,
            bl,
            br,
        )
    }
}

pub fn generate_domain_blocks(
    img: ArrayView2<f32>,
    s: QuadtreeSettings,
    pos: (usize, usize),
) -> Quadtree<RangeBlockLocation> {
    generate_range_blocks(
        img,
        QuadtreeSettings {
            min_range_block_size: s.min_domain_block_size,
            min_range_variance: s.min_domain_variance,
            ..s
        },
        pos,
        0,
    )
}

pub fn compress(
    img: &Array2<f32>,
    s: QuadtreeSettings,
) -> (
    Mappings,
    Quadtree<RangeBlockLocation>,
    Quadtree<RangeBlockLocation>,
) {
    let rt = generate_range_blocks(img.view(), s, (0, 0), s.minimum_range_splits);
    let dt = generate_domain_blocks(img.view(), s, (0, 0));

    let mut range_blocks = rt.prefix_leaves();
    range_blocks.sort_by_key(|&b| b.pos);

    let non_transformed_domain_blocks = if s.only_leaves {
        dt.prefix_leaves()
    } else {
        dt.prefix_traversal()
    };

    let mut domain_blocks_locations = Vec::with_capacity(8 * non_transformed_domain_blocks.len());

    for RangeBlockLocation { pos, size } in non_transformed_domain_blocks {
        for rotation in [
            Rotation::Zero,
            Rotation::Quarter,
            Rotation::Half,
            Rotation::ThreeQuarter,
        ] {
            // for flipped in [false, true] {
            domain_blocks_locations.push(DomainBlockLocation {
                pos,
                size,
                rotation,
                flipped: false,
            });
            // }
        }
    }

    let imgs = create_images(img);
    let views = [
        imgs[0].view(),
        imgs[1].view(),
        imgs[2].view(),
        imgs[3].view(),
    ];
    let domain_blocks = domain_blocks_from_locations(&views, &domain_blocks_locations);

    (find_mappings(img, &range_blocks, &domain_blocks), rt, dt)
}

pub fn show_ranbeblocks(img: &Arr<f32>, t: Quadtree<RangeBlockLocation>, file: String) {
    let mut m = Array2::from_shape_fn(img.dim(), |x| (img[x] * 255.).clamp(0., 255.) as i32 as u8);

    for rb in t.prefix_leaves() {
        for i in 0..rb.size.0 {
            m[(rb.pos.0 + i, rb.pos.1)] = 255;
            m[(rb.pos.0 + i, rb.pos.1 + rb.size.1 - 1)] = 255;
        }
        for j in 0..rb.size.1 {
            m[(rb.pos.0, rb.pos.1 + j)] = 255;
            m[(rb.pos.0 + rb.size.0 - 1, rb.pos.1 + j)] = 255;
        }
    }

    save_gray_image(file, m.view()).unwrap();
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct QuadtreeSettings {
    pub min_domain_variance: f32,
    pub min_range_variance: f32,
    pub min_domain_block_size: usize,
    pub min_range_block_size: usize,
    pub only_leaves: bool,
    pub minimum_range_splits: usize,
}
