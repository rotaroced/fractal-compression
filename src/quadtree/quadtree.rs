use super::compression::{bl, br, tl, tr};
use crate::prelude::RangeBlockLocation;
use either::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quadtree<T: Copy> {
    pub(super) children: Either<Box<[Quadtree<T>; 4]>, T>,
}

impl<T: Copy> Quadtree<T> {
    #[inline]
    pub fn leaf(label: T) -> Self {
        Self {
            children: Right(label),
        }
    }

    #[inline]
    pub fn node(c1: Quadtree<T>, c2: Quadtree<T>, c3: Quadtree<T>, c4: Quadtree<T>) -> Self {
        Self {
            children: Left(Box::new([c1, c2, c3, c4])),
        }
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_right()
    }

    #[inline]
    pub fn is_node(&self) -> bool {
        self.children.is_left()
    }

    // traverses the quadtree's leaves
    fn prefix_leaves_aux(&self, v: &mut Vec<T>) {
        match self.children.as_ref() {
            Left(b) => {
                b[0].prefix_leaves_aux(v);
                b[1].prefix_leaves_aux(v);
                b[2].prefix_leaves_aux(v);
                b[3].prefix_leaves_aux(v);
            }
            Right(&label) => v.push(label),
        }
    }

    pub fn prefix_leaves(&self) -> Vec<T> {
        let mut v = vec![];
        self.prefix_leaves_aux(&mut v);
        v
    }

    pub fn mapi<Q: Copy, F: FnMut(T, RangeBlockLocation) -> Q>(
        &self,
        f: &mut F,
        rb: RangeBlockLocation,
    ) -> Quadtree<Q> {
        match self.children.as_ref() {
            Left(b) => Quadtree::node(
                b[0].mapi(f, tl(rb)),
                b[1].mapi(f, tr(rb)),
                b[2].mapi(f, bl(rb)),
                b[3].mapi(f, br(rb)),
            ),
            Right(&label) => Quadtree::leaf(f(label, rb)),
        }
    }
}

impl Quadtree<RangeBlockLocation> {
    pub fn image_size(&self) -> (usize, usize) {
        match self.children.as_ref() {
            Left(b) => {
                let (h, w) = b[0].image_size();
                (2 * h, 2 * w)
            }
            Right(block) => block.size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn traversal() {
        let t1 = Quadtree::leaf(3);
        let t2 = Quadtree::leaf(69);
        let t3 = Quadtree::node(t1.clone(), t2.clone(), Quadtree::leaf(17), t1.clone());
        let t4 = Quadtree::leaf(861);

        let t = Quadtree::node(t1, t2, t3, t4);

        assert_eq!(t.prefix_leaves(), vec![3, 69, 3, 69, 17, 3, 861]);
    }
}
