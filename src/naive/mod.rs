pub mod io;

pub mod compression;

use crate::prelude::*;
use compression::find_mappings;
use ndarray::prelude::*;

pub fn compress(img: Arr<f32>, params: NaiveCompressionSettings) -> Mappings {
    assert!(params.domain_block_size > params.range_block_size && params.range_block_size > 0);
    assert_eq!(img.dim().0 % params.range_block_size, 0);
    assert_eq!(img.dim().1 % params.range_block_size, 0);

    let mut range_blocks = vec![];

    for i in 0..(img.dim().0 / params.range_block_size) {
        for j in 0..(img.dim().1 / params.range_block_size) {
            range_blocks.push(RangeBlockLocation {
                pos: (i * params.range_block_size, j * params.range_block_size),
                size: (params.range_block_size, params.range_block_size),
            });
        }
    }

    let mut domain_blocks_locations = Vec::with_capacity(
        (img.dim().0 - params.domain_block_size) * (img.dim().1 - params.domain_block_size)
            / params.domain_block_stepy
            / params.domain_block_stepx,
    );
    for i in (0..(img.dim().0 - params.domain_block_size)).step_by(params.domain_block_stepy) {
        for j in (0..(img.dim().1 - params.domain_block_size)).step_by(params.domain_block_stepx) {
            for rot in [
                Rotation::Zero,
                Rotation::Quarter,
                Rotation::Half,
                Rotation::ThreeQuarter,
            ] {
                //for f in [false, true] {
                domain_blocks_locations.push(DomainBlockLocation {
                    pos: (i, j),
                    rotation: rot,
                    flipped: false,
                    size: (params.domain_block_size, params.domain_block_size),
                });
                // }
            }
        }
    }
    let imgs = create_images(&img);
    let views = [
        imgs[0].view(),
        imgs[1].view(),
        imgs[2].view(),
        imgs[3].view(),
    ];

    let domain_blocks = domain_blocks_from_locations(&views, &domain_blocks_locations);

    find_mappings(&img, &range_blocks, &domain_blocks)
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, Default)]
pub struct NaiveCompressionSettings {
    pub range_block_size: usize,
    pub domain_block_size: usize,
    pub domain_block_stepx: usize,
    pub domain_block_stepy: usize,

    pub coord_bits: usize,
}

pub(crate) fn domain_blocks_from_locations<'a>(
    images: &'a [ArrayView2<'a, f32>; 4],
    domain_blocks_locations: &[DomainBlockLocation],
) -> Vec<DomainBlock<'a>> {
    use Rotation::*;
    let (h, w) = images[0].dim();

    domain_blocks_locations
        .iter()
        .map(|&db| match db.rotation {
            Zero => DomainBlock {
                location: db,
                arr: images[0].slice(s![
                    db.pos.0..(db.pos.0 + db.size.0),
                    db.pos.1..(db.pos.1 + db.size.1)
                ]),
            },
            Quarter => DomainBlock {
                location: db,
                arr: images[1].slice(s![
                    db.pos.1..(db.pos.1 + db.size.1),
                    (h - db.pos.0 - db.size.0)..(h - db.pos.0)
                ]),
            },
            Half => DomainBlock {
                location: db,
                arr: images[2].slice(s![
                    (h - db.pos.0 - db.size.0)..(h - db.pos.0),
                    (w - db.pos.1 - db.size.1)..(w - db.pos.1)
                ]),
            },
            ThreeQuarter => DomainBlock {
                location: db,
                arr: images[3].slice(s![
                    (w - db.pos.1 - db.size.1)..(w - db.pos.1),
                    db.pos.0..(db.pos.0 + db.size.0)
                ]),
            },
        })
        .collect()
}

pub(crate) fn create_images(img: &Arr<f32>) -> [Arr<f32>; 4] {
    [
        img.clone(),
        rotate_block(img.view(), Rotation::Quarter),
        rotate_block(img.view(), Rotation::Half),
        rotate_block(img.view(), Rotation::ThreeQuarter),
    ]
}
