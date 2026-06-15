use std::time::Instant;

use super::naive::compression::find_mappings_parallel;
use crate::{decompression::reconstruct, prelude::*};
use ndarray::prelude::*;
use rand::{
    distr::{StandardUniform, uniform::UniformSampler},
    prelude::*,
    rng,
};

/// L'objectif est de répondre à la question "Quelle est la distance avec l'image de départ dans le
/// pire des cas ?"

fn random_matrix(shape: (usize, usize)) -> Arr<f32> {
    Arr::from_shape_simple_fn(shape, || {
        rng().sample::<f32, StandardUniform>(StandardUniform) * 2. - 1.
    })
}

pub fn construct(image: &Arr<f32>, rb: &[RangeBlockLocation], db: &[DomainBlock]) -> Arr<f32> {
    let m = find_mappings_parallel(image, rb, db);
    reconstruct(m, Arr::<f32>::zeros(image.dim()), 50)
}

fn clamp(img: Arr<f32>) -> Arr<f32> {
    img.map(|&x| {
        if x <= 1. && x >= 0. {
            x
        } else {
            rng().sample(StandardUniform)
        }
    })
}

pub fn rand_gen_worst(
    width: usize,
    divisions: usize,
    max_time: Instant,
    perturbation: f32,
    mut worst: (Arr<f32>, Arr<f32>, f32),
) -> (Arr<f32>, Arr<f32>, f32) {
    let db_size = width >> (divisions - 1);
    let rb_size = width >> divisions;
    assert_eq!(width % db_size, 0);

    // generate range & domain blocks (copied from naive::mod)
    let mut range_blocks = vec![];

    for i in 0..(width / rb_size) {
        for j in 0..(width / rb_size) {
            range_blocks.push(RangeBlockLocation {
                pos: (i * rb_size, j * rb_size),
                size: (rb_size, rb_size),
            });
        }
    }

    let mut domain_blocks_locations = Vec::with_capacity((width / db_size) * (width / db_size));
    for i in (0..(width - db_size)).step_by(db_size) {
        for j in (0..(width - db_size)).step_by(db_size) {
            domain_blocks_locations.push(DomainBlockLocation {
                pos: (i, j),
                rotation: Rotation::Zero,
                flipped: false,
                size: (db_size, db_size),
            });
        }
    }

    let mut candidate = Arr::<f32>::from_shape_fn((width, width), |(i, j)| ((i + j) & 1) as f32);

    while Instant::now() <= max_time {
        let view = candidate.view();
        let domain_blocks = domain_blocks_locations
            .iter()
            .map(|&db| DomainBlock {
                location: db,
                arr: view.slice(s![
                    db.pos.0..(db.pos.0 + db.size.0),
                    db.pos.1..(db.pos.1 + db.size.1)
                ]),
            })
            .collect::<Vec<_>>();
        let best = construct(&candidate, &range_blocks, &domain_blocks);

        let d = distance(candidate.view(), best.view());
        if d > worst.2 {
            worst = (candidate, best, d);
        }
        candidate = clamp(worst.0.clone() + perturbation * random_matrix((width, width)));
    }

    worst
}
