mod color;
mod compression;
mod naive;
mod quadtree;

use color::*;
use ndarray::*;
use ndarray_image::*;

use crate::compression::*;
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

fn main() -> Result<(), std::io::Error> {
    let s_lumi = QuadtreeSettings {
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

    let img_u8 = open_image("joli.bmp", Colors::Rgb).unwrap();
    let img = Array3::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f32 / 255.);

    let (h, w, _) = img.clone().dim();
    let size = (h / 3, w / 3);

    let (mut y, mut cr, mut cb) = rgb_to_y_cb_cr(img.clone());

    y = scale_down(&y, size);
    cr = scale_down(&cr, size);
    cb = scale_down(&cb, size);

    compress_grey(y, CompressionMethod::Quadtree(s_lumi), "y.map".into()).unwrap();
    compress_grey(cr, CompressionMethod::Quadtree(s_chro), "cr.map".into()).unwrap();
    compress_grey(cb, CompressionMethod::Quadtree(s_chro), "cb.map".into()).unwrap();

    let comp_y = decompress_grey("y.map".into(), DecompressionMethod::Quadtree, 30).unwrap();
    let comp_cr = decompress_grey("cr.map".into(), DecompressionMethod::Quadtree, 30).unwrap();
    let comp_cb = decompress_grey("cb.map".into(), DecompressionMethod::Quadtree, 30).unwrap();

    let comp_img = y_cb_cr_to_rgb(comp_y, comp_cb, comp_cr);
    let comp_img_u8 = Array3::from_shape_fn((size.0, size.1, 3), |t| {
        (comp_img[t] * 255.).clamp(0., 255.) as i32 as u8
    });
    save_image("joli_col_decomp.png", comp_img_u8.view(), Colors::Rgb).unwrap();

    Ok(())
}
