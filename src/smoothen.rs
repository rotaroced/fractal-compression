use crate::RangeBlockLocation;
use ndarray::Array2;
use std::iter::IntoIterator;

pub fn blur_bottom<I>(img: &mut Array2<f32>, range_blocks: I)
where
    I: IntoIterator<Item = RangeBlockLocation>,
{
    for rb in range_blocks {
        let (x, y) = (rb.pos.0, rb.pos.1);
        let (sx, sy) = rb.size;

        if y + sy == img.dim().1 || x + sx == img.dim().0 || sx <= 5 || sy <= 5 {
            continue;
        }

        if y + sy < 2 && y + sy + 1 >= img.dim().1 {
            continue;
        }

        let j = y + sy;

        for i in x..(x + sx) {
            let a = img[(i, j - 2)];
            let b = img[(i, j - 1)];
            let c = img[(i, j)];
            let d = img[(i, j + 1)];

            img[(i, j - 2)] = (3. * a + 2. * b + c) / 6.;
            img[(i, j - 1)] = (2. * b + c) / 3.;
            img[(i, j)] = (b + 2. * c) / 3.;
            img[(i, j + 1)] = (b + 2. * c + 3. * d) / 6.;
        }
    }
}

pub fn blur_side<I>(img: &mut Array2<f32>, range_blocks: I)
where
    I: IntoIterator<Item = RangeBlockLocation>,
{
    for rb in range_blocks {
        let (x, y) = (rb.pos.0, rb.pos.1);
        let (sx, sy) = rb.size;

        if y + sy == img.dim().1 || x + sx == img.dim().0 || sx <= 5 || sy <= 5 {
            continue;
        }

        if x + sx < 2 && x + sx + 1 >= img.dim().1 {
            continue;
        }

        let i = x + sx;

        for j in y..(y + sy) {
            let a = img[(i - 2, j)];
            let b = img[(i - 1, j)];
            let c = img[(i, j)];
            let d = img[(i + 1, j)];

            img[(i - 2, j)] = (3. * a + 2. * b + c) / 6.;
            img[(i - 1, j)] = (2. * b + c) / 3.;
            img[(i, j)] = (b + 2. * c) / 3.;
            img[(i + 1, j)] = (b + 2. * c + 3. * d) / 6.;
        }
    }
}
