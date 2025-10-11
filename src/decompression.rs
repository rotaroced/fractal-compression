use super::prelude::*;

// reconstruction naïve
pub fn reconstruct(mappings: Mappings, mut img: Arr<f32>, n: usize) -> Arr<f32> {
    let mut new_img = img.clone();
    for _ in 0..n {
        for (&rb, &(db, contrast, brightness)) in mappings.iter() {
            // println!("{rb:?}, {db:?}");
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
