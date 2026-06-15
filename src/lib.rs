pub mod color;
pub mod decompression;
pub mod naive;
pub mod prelude;
pub mod quadtree;
pub mod smoothen;
mod worst_case;

pub use crate::decompression::reconstruct;
pub use crate::decompression::reconstruct_smart;
pub use crate::quadtree::QuadtreeSettings;
pub use crate::{DomainBlock, DomainBlockLocation, Mappings, RangeBlockLocation, Rotation};
pub use prelude::*;

const MAX_COEF: f32 = 0.999;
const N_DIMS_SQRT: usize = 2;
const N_DIMS: usize = N_DIMS_SQRT * N_DIMS_SQRT;
