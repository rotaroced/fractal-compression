mod color;
mod decompression;
mod naive;
pub mod prelude;
mod quadtree;
mod smoothen;
mod worst_case;

use crate::decompression::reconstruct;
use crate::quadtree::compress;
use crate::smoothen::blur_bottom;
use crate::smoothen::blur_side;
use color::*;
use ndarray::*;
use ndarray_image::*;
pub use prelude::*;
use std::time::{Duration, Instant};

use crate::quadtree::QuadtreeSettings;
use crate::quadtree::io::*;
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

#[cfg(not(find_worst))]
fn main() -> Result<(), std::io::Error> {
    let s = QuadtreeSettings {
        max_distance: 0.01,
        maximum_range_splits: 7,
        minimum_range_splits: 1,
    };

    let img_u8 = open_gray_image("capybara_orange.png").unwrap();
    let img = Array2::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f32 / 255.);

    let size = img.dim();

    let (m, q) = compress(&img, s);

    println!("{:?}", m);

    save_mappings("capybara.map".into(), &(m.clone(), q), s).unwrap();

    let mut r = decompression::reconstruct(m.clone(), img.clone(), 45);
    println!("rms sharp: {}", distance(r.view(), img.view()));
    let r_u81 = Array2::from_shape_fn(r.dim(), |t| (r[t] * 255.) as u8);
    save_gray_image("lena_decompressed_qt_sharp.png", r_u81.view()).unwrap();
    blur_bottom(&mut r, m.keys().into_iter().map(|&x| x));
    blur_side(&mut r, m.keys().into_iter().map(|&x| x));
    let r_u82 = Array2::from_shape_fn(r.dim(), |t| (r[t] * 255.) as u8);

    println!("rms smooth: {}", distance(r.view(), img.view()));
    save_gray_image("lena_decompressed_qt.png", r_u82.view()).unwrap();

    /*let s_lumi = QuadtreeSettings {
            min_domain_variance: 0.002,
            min_range_variance: 0.000001,
            min_domain_block_size: 9,
            min_range_block_size: 4,
            only_leaves: true,
            minimum_range_splits: 5,
        };
        let s_chro = QuadtreeSettings {
            min_domain_variance: (0.008f32).powi(2),
            min_range_variance: (0.006f32).powi(2),
            min_domain_block_size: 9,
            min_range_block_size: 4,
            only_leaves: false,
            minimum_range_splits: 3,
        };

        let s = NaiveCompressionSettings {
            range_block_size: 4,
            domain_block_size: 8,
            domain_block_stepx: 8,
            domain_block_stepy: 8,
            ..Default::default()
        };

        let img_u8 = open_gray_image("joli.bmp").unwrap();
        let img = Array2::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f32 / 255.);

        let (h, w) = img.clone().dim();
        let size = (h, w);

        // let (mut y, mut cr, mut cb) = rgb_to_y_cb_cr(img.clone());

        // y = scale_down(&y, size);
        // cr = scale_down(&cr, size);
        // cb = scale_down(&cb, size);

        // compress_grey(y, CompressionMethod::Naive(s), "y.map".into()).unwrap();
        // compress_grey(cr, CompressionMethod::Naive(s), "cr.map".into()).unwrap();
        // compress_grey(cb, CompressionMethod::Naive(s), "cb.map".into()).unwrap();
        compress_grey(img, CompressionMethod::Naive(s), "compressed.map".into()).unwrap();
        println!("compression ok");

        // let comp_y = decompress_grey("y.map".into(), DecompressionMethod::Naive, 30).unwrap();
        // let comp_cr = decompress_grey("cr.map".into(), DecompressionMethod::Naive, 30).unwrap();
        // let comp_cb = decompress_grey("cb.map".into(), DecompressionMethod::Naive, 30).unwrap();
        let dc = decompress_grey("compressed.map".into(), DecompressionMethod::Naive, 30).unwrap();

        println!("decompression ok");

        // let comp_img = y_cb_cr_to_rgb(comp_y, comp_cb, comp_cr);
        let comp_img_u8 = Array2::from_shape_fn((size.0, size.1), |t| {
            (dc[t] * 255.).clamp(0., 255.) as i32 as u8
        });
        save_gray_image("joli_decomp.png", comp_img_u8.view()).unwrap();
    */
    Ok(())
}

#[cfg(find_worst)]
pub fn main() -> () {
    use core::time::Duration;

    for i in 3..8 {
        for divisions in 2..i {
            let width = 1 << i;
            println!("starting width = {width}, divisions = {divisions}");
            let mut worst = (
                Arr::<f32>::zeros((width, width)),
                Arr::<f32>::zeros((width, width)),
                0.,
            );

            worst = worst_case::rand_gen_worst(
                width,
                divisions,
                Instant::now() + Duration::from_secs(60),
                1.,
                worst,
            );

            worst = worst_case::rand_gen_worst(
                width,
                divisions,
                Instant::now() + Duration::from_secs(60),
                0.5,
                worst,
            );

            worst = worst_case::rand_gen_worst(
                width,
                divisions,
                Instant::now() + Duration::from_secs(60),
                0.1,
                worst,
            );
            println!("{:?}", worst.0);
            println!("{:?}", worst.1);
            println!("({i}, {divisions}): {}", worst.2);
        }
    }
}
