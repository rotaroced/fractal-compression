use ndarray::{Array2, Array3, Axis, stack};

/// converts an RGB image to YCbCr
pub fn rgb_to_y_cb_cr(img: Array3<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    assert_eq!(img.dim().2, 3);

    let r = img.index_axis(Axis(2), 0);
    let g = img.index_axis(Axis(2), 1);
    let b = img.index_axis(Axis(2), 2);

    let y =
        Array2::<f32>::from_shape_fn(r.dim(), |t| 0.2627 * r[t] + 0.6780 * g[t] + 0.0593 * b[t]);
    let cr = Array2::<f32>::from_shape_fn(r.dim(), |t| (b[t] - y[t]) / 1.8814);
    let cb = Array2::<f32>::from_shape_fn(r.dim(), |t| (r[t] - y[t]) / 1.4746);

    (y, cb, cr)
}

pub fn y_cb_cr_to_rgb(y: Array2<f32>, cb: Array2<f32>, cr: Array2<f32>) -> Array3<f32> {
    assert_eq!(y.shape(), cb.shape());
    assert_eq!(y.shape(), cr.shape());

    let r = Array2::<f32>::from_shape_fn(y.dim(), |t| y[t] + 1.4746 * cr[t]);
    let g = Array2::<f32>::from_shape_fn(y.dim(), |t| y[t] - 0.1646 * cb[t] - 0.5714 * cr[t]);
    let b = Array2::<f32>::from_shape_fn(y.dim(), |t| y[t] + 1.8814 * cb[t]);

    stack(Axis(2), &[r.view(), g.view(), b.view()]).unwrap()
}
