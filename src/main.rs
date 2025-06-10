mod compression;

use std::error::Error;

use ndarray::*;
use ndarray_image::*;

use crate::compression::*;

fn main() {
    let img_u8 = open_gray_image("lena.jpg").unwrap();
    let img = Array2::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f64 / 255.);

    let mappings = compress(img.clone());
    let dimg = reconstruct(mappings, Array2::<f64>::zeros(img.dim()), 30);
    let dimg_u8 = Array2::from_shape_fn(dimg.dim(), |t| (dimg[t].clamp(0., 1.) * 255.) as u8);

    save_gray_image("res.png", dimg_u8.view()).unwrap();
}
