mod color;
mod decompression;
mod naive;
pub mod prelude;
mod quadtree;
mod smoothen;
mod worst_case;

use crate::decompression::reconstruct;
use crate::decompression::reconstruct_smart;
use crate::quadtree::compress;
use crate::smoothen::blur_bottom;
use crate::smoothen::blur_side;
use color::*;
use ndarray::*;
use ndarray_image::*;
pub use prelude::*;
use std::fs::File;
use std::time::{Duration, Instant};
use worst_case::*;

use crate::quadtree::QuadtreeSettings;
use crate::quadtree::io::*;

const MAX_COEF: f32 = 3.99;
const N_DIMS_SQRT: usize = 6;

/*
use crate::naive::NaiveCompressionSettings;
use crate::quadtree::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum CompressionMethod {
    Naive(NaiveCompressionSettings),
    Quadtree(QuadtreeSettings),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DecompressionMethod {
    Naive,
    Quadtree,
}

fn compress_grey(
    img: Arr<f32>,
    method: CompressionMethod,
    out_file: String,
) -> Result<usize, std::io::Error> {
    match method {
        CompressionMethod::Naive(s) => {
            let mappings = crate::naive::compress(img, s);
            crate::naive::io::save_mappings(&mappings, out_file, s)
        }
        CompressionMethod::Quadtree(s) => {
            let m = crate::quadtree::compress(&img, s);
            crate::quadtree::io::save_mappings(out_file, &m, s)
        }
    }
}

fn decompress_grey(
    mappings_file: String,
    method: DecompressionMethod,
    decompression_passes: usize,
) -> Result<Arr<f32>, std::io::Error> {
    match method {
        DecompressionMethod::Naive => {
            let (_, size, mappings) = crate::naive::io::load_mappings(mappings_file)?;
            Ok(reconstruct(
                mappings,
                Arr::<f32>::zeros(size),
                decompression_passes,
            ))
        }
        DecompressionMethod::Quadtree => {
            let (_, size, (mappings, _, _)) = crate::quadtree::io::load_mappings(mappings_file)?;
            Ok(reconstruct(
                mappings,
                Arr::<f32>::zeros(size),
                decompression_passes,
            ))
        }
    }
}
*/

fn main() -> Result<(), std::io::Error> {
    let s = QuadtreeSettings {
        max_distance: 0.005,
        minimum_range_splits: 3,
        maximum_range_splits: 6,
        max_neighbors: 30,
    };

    let img_u8 = open_gray_image("baboon.png").unwrap();
    let img = &scale_down(
        &Array2::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f32 / 255.),
        (224 * 4, 224 * 4),
    );

    let size = img.dim();

    let (m, q) = compress(img, s);

    let f = File::create("baboon.map")?;
    let compressed_size = quadtree::io::save_mappings(f, &(m.clone(), q), s)?;

    println!("compressed size: {compressed_size}");

    let f = File::open("baboon.map")?;
    let (_, _, mappings, _) = quadtree::io::load_mappings(f)?;

    let i = reconstruct(mappings, Array2::zeros(size), 15);

    let iu8 = Array2::from_shape_fn(size, |t| (i[t] * 255.) as u8);

    save_gray_image("baboon_decomp.png", iu8.view()).unwrap();

    println!("bmp size: {}", i.dim().0 * i.dim().1);
    println!(
        "bmp compression ratio: {}",
        1. / (compressed_size as f64) * (i.dim().0 * i.dim().1) as f64
    );
    println!("PSNR {}", -20. * (distance(i.view(), img.view())).log10());

    Ok(())
}
