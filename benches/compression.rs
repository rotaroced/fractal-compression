use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fractal_compression::*;
use ndarray_image::open_gray_image;
use std::{fs::File, hint::black_box};

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("img_test");
    g.sample_size(20);

    let img_u8 = open_gray_image("img_test.png").unwrap();
    let lena256 = scale_down(
        &ndarray::Array2::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f32 / 255.),
        (256, 256),
    );
    let lena512 = ndarray::Array2::from_shape_fn(img_u8.dim(), |t| img_u8[t] as f32 / 255.);

    let s = quadtree::QuadtreeSettings {
        max_distance: 0.01,
        minimum_range_splits: 3,
        maximum_range_splits: 6,
        max_neighbors: 10,
    };
    let s2 = quadtree::QuadtreeSettings {
        max_distance: 0.01,
        minimum_range_splits: 5,
        maximum_range_splits: 5,
        max_neighbors: 10,
    };

    let naive_params = naive::NaiveCompressionSettings {
        range_block_size: 8,
        domain_block_size: 16,
        domain_block_stepx: 16,
        domain_block_stepy: 16,
        coord_bits: 8,
    };

    let s512 = quadtree::QuadtreeSettings {
        max_distance: 0.01,
        minimum_range_splits: 4,
        maximum_range_splits: 7,
        max_neighbors: 10,
    };
    let s2_512 = quadtree::QuadtreeSettings {
        max_distance: 0.01,
        minimum_range_splits: 6,
        maximum_range_splits: 6,
        max_neighbors: 10,
    };

    g.bench_with_input(
        BenchmarkId::new(
            format!(
                "quadtree + rtree + maximum_range_splits = {}",
                s.maximum_range_splits
            ),
            "lena256",
        ),
        &lena256,
        |b, i| b.iter(|| quadtree::compress(i, s)),
    );

    g.bench_with_input(
        BenchmarkId::new(
            format!(
                "quadtree + rtree + always {:?} splits",
                s2.maximum_range_splits
            ),
            "lena256",
        ),
        &lena256,
        |b, i| b.iter(|| quadtree::compress(i, s2)),
    );

    g.bench_with_input(
        BenchmarkId::new("naive (8x8 range blocks)", "lena256"),
        &lena256,
        |b, i| b.iter(|| naive::compress(i, naive_params)),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
