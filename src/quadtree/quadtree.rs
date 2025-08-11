#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quadtree<T: Copy> {
    pub(super) children: Option<Box<[Quadtree<T>; 4]>>,

    pub(super) label: T,
}

impl<T: Copy> Quadtree<T> {
    #[inline]
    pub fn leaf(label: T) -> Self {
        Self {
            label,
            children: None,
        }
    }

    #[inline]
    pub fn node(
        label: T,
        c1: Quadtree<T>,
        c2: Quadtree<T>,
        c3: Quadtree<T>,
        c4: Quadtree<T>,
    ) -> Self {
        Self {
            children: Some(Box::new([c1, c2, c3, c4])),
            label,
        }
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    #[inline]
    pub fn is_node(&self) -> bool {
        self.children.is_some()
    }

    // constructs the prefix traversal of the quadtree
    fn prefix(&self, v: &mut Vec<T>) {
        v.push(self.label);

        if let Some(b) = self.children.as_ref() {
            b[0].prefix(v);
            b[1].prefix(v);
            b[2].prefix(v);
            b[3].prefix(v);
        }
    }

    pub fn prefix_traversal(&self) -> Vec<T> {
        let mut v = vec![];
        self.prefix(&mut v);
        v
    }

    // traverses the quadtree's leaves
    fn prefix_leaves_aux(&self, v: &mut Vec<T>) {
        if let Some(b) = self.children.as_ref() {
            b[0].prefix_leaves_aux(v);
            b[1].prefix_leaves_aux(v);
            b[2].prefix_leaves_aux(v);
            b[3].prefix_leaves_aux(v);
        } else {
            v.push(self.label)
        }
    }

    pub fn prefix_leaves(&self) -> Vec<T> {
        let mut v = vec![];
        self.prefix_leaves_aux(&mut v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn traversal() {
        let t1 = Quadtree::leaf(3);
        let t2 = Quadtree::leaf(69);
        let t3 = Quadtree::node(-7, t1.clone(), t2.clone(), Quadtree::leaf(17), t1.clone());
        let t4 = Quadtree::leaf(861);

        let t = Quadtree::node(47, t1, t2, t3, t4);

        assert_eq!(t.prefix_traversal(), vec![47, 3, 69, -7, 3, 69, 17, 3, 861]);
        assert_eq!(t.prefix_leaves(), vec![3, 69, 3, 69, 17, 3, 861]);
    }
}
