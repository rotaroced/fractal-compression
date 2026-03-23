use core::f32;

use super::prelude::*;

// reconstruction naïve
pub fn reconstruct(mappings: Mappings, mut img: Arr<f32>, n: usize) -> Arr<f32> {
    let mut new_img = img.clone();
    for _ in 0..n {
        for (&rb, &(db, contrast, brightness)) in mappings.iter() {
            // println!("{rb:?}, {db:?}");
            // println!("{:?} {:?}", db, rb.size);
            let transformed_block = scale_down(
                &rotate_block(get_domainblock(&img, db), db.rotation),
                rb.size,
            ) * contrast
                + brightness;

            for i in 0..rb.size.0 {
                for j in 0..rb.size.1 {
                    // println!("{:?}", (rb.pos.0 + i, rb.pos.1 + j));
                    new_img[(rb.pos.0 + i, rb.pos.1 + j)] = transformed_block[(i, j)];
                }
            }
        }
        img = new_img.clone();
    }

    img
}

/// decompresses the image without cloning the image at each iteration
/// TODO : Construct Mappings in a smarter way (quadtree ?) so there is no need to create the rbs
/// vec
pub fn reconstruct_smart(mappings: &Mappings, mut img: Arr<f32>, delta: f32) -> Arr<f32> {
    let mut dist = f32::INFINITY;

    let dflt: (RangeBlockLocation, &Transformation) =
        (Default::default(), &(Default::default(), 0., 0.));
    let mut rbs: Vec<(RangeBlockLocation, &Transformation)> =
        Vec::with_capacity(img.dim().0 * img.dim().1);
    rbs.resize(img.dim().0 * img.dim().1, dflt);

    for (&rb, m) in mappings {
        for i in 0..rb.size.0 {
            for j in 0..rb.size.1 {
                rbs[(i + rb.pos.0) * img.dim().1 + j + rb.pos.1] = (rb, m);
            }
        }
    }

    while dist > delta {
        dist = 0.;
        for (u, (db, m)) in rbs.iter().enumerate() {
            let (i, j) = (u / img.dim().1, u % img.dim().1);
            let old_val = img[(i, j)];

            let (x, y) = (i - db.pos.0, j - db.pos.1);
            let (xrb, yrb) = (2 * x + m.0.pos.0, 2 * y + m.0.pos.1);
            img[(i, j)] = (img[(xrb, yrb)]
                + img[(xrb + 1, yrb)]
                + img[(xrb, yrb + 1)]
                + img[(xrb + 1, yrb + 1)])
                / 4.
                * m.1
                + m.2;

            dist += (old_val - img[(i, j)]) * (old_val - img[(i, j)]);
        }

        dist = dist.sqrt();
    }

    img
}
