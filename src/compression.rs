use core::f64;
use std::collections::HashMap;

use ndarray::*;
use rayon::{
    self,
    iter::{IntoParallelRefIterator, ParallelIterator},
};
pub const DOMAIN_BLOCK_SIZE: usize = 8;
pub const RANGE_BLOCK_SIZE: usize = 4;

pub type Arr<A> = Array2<A>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Block {
    pub pos: (usize, usize),
}

pub type RangeBlock = Block;

pub type Transformation = (Block, f64, f64);

pub type Mappings = HashMap<RangeBlock, Transformation>;

pub fn square(img: &Arr<f64>) -> Arr<f64> {
    img * img
}

/// computes the distance between two images
pub fn distance(img1: &Arr<f64>, img2: &Arr<f64>) -> f64 {
    square(&(img1 - img2)).sum().sqrt()
}

pub fn scale_down(img: &Arr<f64>, target_size: (usize, usize)) -> Arr<f64> {
    let mut a = Arr::<f64>::zeros(target_size);
    for i in 0..target_size.0 {
        for j in 0..target_size.1 {
            let (x, y) = (
                img.dim().0 * i / target_size.0,
                img.dim().1 * j / target_size.1,
            );
            let (ex, ey) = (
                img.dim().0 * (i + 1) / target_size.0,
                img.dim().1 * (j + 1) / target_size.1,
            );
            a[(i, j)] = img.slice(s![x..ex, y..ey]).mean().unwrap();
        }
    }

    a
}

/// calcul de la luminosité et du contraste optimaux (avec la méthode des moindres carrés)
pub fn find_brightness_and_contrast(range_block: &Arr<f64>, domain_block: &Arr<f64>) -> (f64, f64) {
    debug_assert_eq!(range_block.dim(), domain_block.dim());
    let er = range_block.mean().unwrap();
    let dr = domain_block.mean().unwrap();

    let cov = ((range_block - er) * (domain_block - dr)).mean().unwrap();
    let var = square(&(range_block - er)).mean().unwrap();

    let contrast = cov / var;
    let brightness = (domain_block - &(contrast * range_block)).mean().unwrap();

    (contrast, brightness)
}

fn get_block(img: &Arr<f64>, block: Block, size: usize) -> Arr<f64> {
    let a = img.slice(s![
        block.pos.0..(size + block.pos.0),
        (block.pos.1)..(size + block.pos.1)
    ]);
    a.into_owned()
}

fn find_best_domain_block(img: &Arr<f64>, rb: Block, dbs: &Vec<Block>) -> Block {
    *dbs.par_iter()
        .min_by(|&&db1, &&db2| {
            distance(
                &get_block(img, rb, RANGE_BLOCK_SIZE),
                &scale_down(
                    &get_block(img, db1, DOMAIN_BLOCK_SIZE).into_owned(),
                    (RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE),
                ),
            )
            .partial_cmp(&distance(
                &get_block(img, rb, RANGE_BLOCK_SIZE),
                &scale_down(
                    &get_block(img, db2, DOMAIN_BLOCK_SIZE).into_owned(),
                    (RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE),
                ),
            ))
            .unwrap()
        })
        .unwrap()
}

pub fn find_mappings(
    img: &Arr<f64>,
    range_blocks: &Vec<RangeBlock>,
    domain_blocks: &Vec<Block>,
) -> Mappings {
    let mut mappings = HashMap::<RangeBlock, Transformation>::new();

    for &rb in range_blocks {
        println!("{:?}", rb);
        let db = find_best_domain_block(img, rb, domain_blocks);

        let (contrast, brightness) = find_brightness_and_contrast(
            &get_block(img, rb, RANGE_BLOCK_SIZE),
            &scale_down(
                &get_block(img, db, DOMAIN_BLOCK_SIZE),
                (RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE),
            ),
        );

        mappings.insert(rb, (db, contrast, brightness));
    }

    mappings
}

pub fn reconstruct(mappings: Mappings, mut img: Arr<f64>, n: usize) -> Arr<f64> {
    let mut new_img = img.clone();
    for _ in 0..n {
        for (&rb, &(db, contrast, brightness)) in mappings.iter() {
            let transformed_block = scale_down(
                &get_block(&img, db, DOMAIN_BLOCK_SIZE),
                (RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE),
            ) * contrast
                + brightness;

            for i in 0..RANGE_BLOCK_SIZE {
                for j in 0..RANGE_BLOCK_SIZE {
                    new_img[(rb.pos.0 + i, rb.pos.1 + j)] = transformed_block[(i, j)];
                }
            }
        }
        img = new_img.clone();
    }

    img
}

pub fn compress(img: Arr<f64>) -> Mappings {
    assert!(img.dim().0 % DOMAIN_BLOCK_SIZE == 0);
    assert!(img.dim().1 % DOMAIN_BLOCK_SIZE == 0);

    let mut range_blocks = vec![];

    for i in 0..(img.dim().0 / RANGE_BLOCK_SIZE) {
        for j in 0..(img.dim().0 / RANGE_BLOCK_SIZE) {
            range_blocks.push(RangeBlock {
                pos: (i * RANGE_BLOCK_SIZE, j * RANGE_BLOCK_SIZE),
            });
        }
    }

    let mut domain_blocks = vec![];
    for i in (0..(img.dim().0 - DOMAIN_BLOCK_SIZE)).step_by(5) {
        for j in (0..(img.dim().0 - DOMAIN_BLOCK_SIZE)).step_by(5) {
            domain_blocks.push(RangeBlock { pos: (i, j) });
        }
    }

    find_mappings(&img, &range_blocks, &domain_blocks)
}
