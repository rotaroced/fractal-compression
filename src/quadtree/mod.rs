mod compression;
pub mod io;
pub mod quadtree;

use std::collections::HashMap;

use super::prelude::*;
use compression::create_rtree;
use ndarray::{Array2, ArrayView2};
use ndarray_image::save_gray_image;
use quadtree::*;
use rayon::prelude::*;

pub fn variance(img: ArrayView2<f32>) -> f32 {
    let m = img.mean().unwrap();
    img.iter()
        .map(|x| (x - m) * (x - m))
        .fold(0., <f32 as std::ops::Add<f32>>::add)
        / ((img.dim().0 * img.dim().1) as f32)
}

pub fn compress(
    img: &Array2<f32>,
    s: QuadtreeSettings,
) -> (Mappings, Quadtree<RangeBlockLocation>) {
    let dbs_rtrees = (s.minimum_range_splits..=s.maximum_range_splits)
        .map(|level| create_rtree(img.view(), level))
        .collect::<Vec<_>>();

    let qt = compression::make_quadtree(
        img,
        RangeBlockLocation {
            pos: (0, 0),
            size: img.dim(),
        },
        s,
        0,
        &dbs_rtrees,
    );

    let mut m = HashMap::new();
    let rbs = qt.mapi(
        &mut |t, rb| {
            m.insert(rb, t);
            rb
        },
        RangeBlockLocation {
            pos: (0, 0),
            size: img.dim(),
        },
    );

    (m, rbs)
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
    pub max_distance: f32,
    pub minimum_range_splits: usize,
    pub maximum_range_splits: usize,
    pub max_neighbors: usize,
}
