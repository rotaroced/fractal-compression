use fractal_compression::naive::NaiveCompressionSettings;
use fractal_compression::*;

use ndarray::Array2;
use ndarray_image::open_gray_image;
use ndarray_image::save_gray_image;
use quadtree::QuadtreeSettings;
use std::fs::File;
use std::time::Duration;
use std::time::Instant;

// const NAIVE: bool = false;

fn main() -> Result<(), std::io::Error> {
    let s = QuadtreeSettings {
        max_distance: 0.025,
        minimum_range_splits: 0,
        maximum_range_splits: 6,
        max_neighbors: 30,
    };
    let naive_s = NaiveCompressionSettings {
        range_block_size: 8,
        domain_block_size: 16,
        domain_block_stepx: 16,
        domain_block_stepy: 16,
        ..Default::default()
    };

    // let
    let img_u8 = open_gray_image("capybara_orange.png").unwrap();
    let img = &scale_down(
        &Array2::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f32 / 255.),
        (512, 512),
    );
    let size = img.dim();

    let t = Instant::now();
    let (m, q) = quadtree::compress(img, s);
    // let m = naive::compress(img, naive_s);
    let d = Instant::now() - t;
    println!("time: {:?}", d);

    let f = File::create("capybara_orange.map")?;
    let compressed_size = quadtree::io::save_mappings(f, &(m.clone(), q), s)?;
    // let compressed_size = naive::io::save_mappings(&m, "capybara.map".into(), naive_s)?;

    // println!("compressed size: {compressed_size}");

    let f = File::open("capybara_orange.map")?;
    let (_, _, mappings, _) = quadtree::io::load_mappings(f)?;
    // let (_, _, mappings) = naive::io::load_mappings("capybara.map".into())?;

    let i = reconstruct(m, Array2::zeros(size), 1);
    // let i = reconstruct_smart(&mappings, Array2::zeros(size), 0.0001);

    let iu8 = Array2::from_shape_fn(size, |t| (i[t] * 255.) as u8);

    save_gray_image("arch_quick.png", iu8.view()).unwrap();
    let png_size = File::open("arch_quick.png")?.metadata().unwrap().len();

    // println!("bmp size: {}", i.dim().0 * i.dim().1);
    // println!("png size: {}", png_size);
    // println!(
    //     "bmp compression ratio: {}",
    //     1. / (compressed_size as f64) * (i.dim().0 * i.dim().1) as f64
    // );
    // println!(
    //     "png compression ratio: {}",
    //     (png_size as f64) / (compressed_size as f64)
    // );
    // println!("PSNR {}", -20. * (distance(i.view(), img.view())).log10());
    //
    Ok(())
}
