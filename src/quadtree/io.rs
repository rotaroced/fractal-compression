use std::{
    collections::HashMap,
    io::{Read, Write},
};

use super::quadtree::*;
use crate::{
    naive::io::{BinBufReader, BinBufWriter},
    prelude::*,
    quadtree::QuadtreeSettings,
};
use either::*;

// TODO: ne pas stocker les branchements nécessaires (par rapport à `minimum_range_splits` et
// `maximum_range_splits`)
pub fn write_quadtree<T: Write>(
    w: &mut BinBufWriter<T>,
    t: &Quadtree<RangeBlockLocation>,
    s: QuadtreeSettings,
    depth: usize,
) -> Result<(), std::io::Error> {
    if let Left(b) = t.children.as_ref() {
        if depth > s.minimum_range_splits {
            // println!("{depth}");
            w.add_bit(false)?;
        } else {
            // println!("skipping start");
        }
        write_quadtree(w, &b[0], s, depth + 1)?;
        write_quadtree(w, &b[1], s, depth + 1)?;
        write_quadtree(w, &b[2], s, depth + 1)?;
        write_quadtree(w, &b[3], s, depth + 1)
    } else {
        // println!("{depth}");
        if depth <= s.maximum_range_splits {
            w.add_bit(true)
        } else {
            // println!("skipping leaf");
            Ok(())
        }
    }
}

pub fn read_quadtree<T: Read>(
    r: &mut BinBufReader<T>,
    size: (usize, usize),
    offset: (usize, usize),
    s: QuadtreeSettings,
    depth: usize,
) -> Result<Quadtree<RangeBlockLocation>, std::io::Error> {
    // no read if obvious result
    if depth > s.maximum_range_splits {
        return Ok(Quadtree::leaf(RangeBlockLocation { size, pos: offset }));
    } else if depth <= s.minimum_range_splits {
        let c1 = read_quadtree(r, (size.0 / 2, size.1 / 2), offset, s, depth + 1)?;
        let c2 = read_quadtree(
            r,
            (size.0 - size.0 / 2, size.1 / 2),
            (offset.0 + size.0 / 2, offset.1),
            s,
            depth + 1,
        )?;
        let c3 = read_quadtree(
            r,
            (size.0 / 2, size.1 - size.1 / 2),
            (offset.0, offset.1 + size.1 / 2),
            s,
            depth + 1,
        )?;
        let c4 = read_quadtree(
            r,
            (size.0 - size.0 / 2, size.1 - size.1 / 2),
            (offset.0 + size.0 / 2, offset.1 + size.1 / 2),
            s,
            depth + 1,
        )?;

        return Ok(Quadtree::node(c1, c2, c3, c4));
    }

    let b = r.read_bit()?;

    if b {
        Ok(Quadtree::leaf(RangeBlockLocation { size, pos: offset }))
    } else {
        let c1 = read_quadtree(r, (size.0 / 2, size.1 / 2), offset, s, depth + 1)?;
        let c2 = read_quadtree(
            r,
            (size.0 - size.0 / 2, size.1 / 2),
            (offset.0 + size.0 / 2, offset.1),
            s,
            depth + 1,
        )?;
        let c3 = read_quadtree(
            r,
            (size.0 / 2, size.1 - size.1 / 2),
            (offset.0, offset.1 + size.1 / 2),
            s,
            depth + 1,
        )?;
        let c4 = read_quadtree(
            r,
            (size.0 - size.0 / 2, size.1 - size.1 / 2),
            (offset.0 + size.0 / 2, offset.1 + size.1 / 2),
            s,
            depth + 1,
        )?;

        Ok(Quadtree::node(c1, c2, c3, c4))
    }
}

pub fn save_mappings<T: Write>(
    file: T,

    mappings: &(Mappings, Quadtree<RangeBlockLocation>),
    s: QuadtreeSettings,
) -> Result<usize, std::io::Error> {
    let mut writer = BinBufWriter {
        x: 0,
        n: 0,
        size_written: 0,
        buf: file,
    };

    // saves the settings
    writer.write_int(s.minimum_range_splits, 6)?;
    writer.write_int(s.maximum_range_splits, 6)?;
    writer.write_float(s.max_distance)?;

    // saves the image size
    let (h, w) = mappings.1.image_size();
    writer.write_int(h, 32)?;
    writer.write_int(w, 32)?;

    let a = writer.size_written;
    // saves the quadtrees
    write_quadtree(&mut writer, &mappings.1, s, 0)?;
    println!("quadtree size: {}", writer.size_written - a);

    // saves transformations. domain blocks are identified by their index in the prefix traversal
    // of the quadtree
    for rb in mappings.1.prefix_leaves() {
        let &(db, c, b) = mappings.0.get(&rb).unwrap();

        // println!("wrote {:?}, {:?}", rb, (db, c, b));

        // TODO : optimiser le nombre de bits utilisés pour stocker les transformations
        writer.write_int(db.pos.0 / 8, 7)?;
        writer.write_int(db.pos.1 / 8, 7)?;
        // writer.write_rotation(db.rotation)?;
        // writer.add_bit(db.flipped)?;
        writer.write_float(c)?;
        writer.write_float(b)?;
    }
    Ok(writer.size_written.div_ceil(8))
}

pub fn load_mappings<T: Read>(
    file: T,
) -> Result<
    (
        QuadtreeSettings,
        (usize, usize),
        Mappings,
        Quadtree<RangeBlockLocation>,
    ),
    std::io::Error,
> {
    let mut reader = BinBufReader::new(file);

    let mut s = QuadtreeSettings::default();

    s.minimum_range_splits = reader.read_int(6)?;
    s.maximum_range_splits = reader.read_int(6)?;

    s.max_distance = reader.read_float()?;

    let (h, w) = (reader.read_int(32)?, reader.read_int(32)?);

    let range_blocks_qt = read_quadtree(&mut reader, (h, w), (0, 0), s, 0)?;

    let mut mappings = HashMap::new();

    for rb in range_blocks_qt.prefix_leaves() {
        let dbx = reader.read_int(7)? * 8;
        let dby = reader.read_int(7)? * 8;
        // let rotation = reader.read_rotation()?;
        // let flipped = reader.read_bit()?;
        let rotation = Rotation::Zero;
        let flipped = false;
        let c = reader.read_float()?;
        let b = reader.read_float()?;
        // println!(
        //     "read {:?}, {:?}, {:?}",
        //     rb,
        //     (
        //         DomainBlockLocation {
        //             pos: (dbx, dby),
        //             size: (2 * rb.size.0, 2 * rb.size.1),
        //             rotation,
        //             flipped,
        //         },
        //         c,
        //         b,
        //     ),
        //     dbx
        // );

        mappings.insert(
            rb,
            (
                DomainBlockLocation {
                    pos: (dbx, dby),
                    size: (2 * rb.size.0, 2 * rb.size.1),
                    rotation,
                    flipped,
                },
                c,
                b,
            ),
        );
    }

    Ok((s, (h, w), mappings, range_blocks_qt))
}

#[cfg(test)]
mod tests {
    use crate::quadtree::{QuadtreeSettings, compress};

    use super::*;
    #[test]
    fn quadtree_saving() {
        let mut buf = [0; 10000];

        let imgs = (0..5)
            .map(|_| ndarray::Array2::from_shape_fn((64, 64), |_| rand::random()))
            .collect::<Vec<_>>();

        let s = QuadtreeSettings {
            max_distance: 0.0861,
            minimum_range_splits: 0,
            maximum_range_splits: 4,
            max_neighbors: 3,
        };

        let mappings = imgs.iter().map(|img| compress(img, s)).collect::<Vec<_>>();

        for (i, t) in mappings.iter().enumerate() {
            save_mappings(buf.as_mut_slice(), t, s).expect("aaaaa");

            println!("{:?}", &buf);

            let read = load_mappings(buf.as_slice()).unwrap().2;

            let _ = read.keys().map(|u| println!("{:?}", u)).collect::<Vec<_>>();
            println!("\n");
            let _ = mappings[i]
                .0
                .keys()
                .map(|u| println!("{:?}", u))
                .collect::<Vec<_>>();

            assert!(read.keys().all(|k| mappings[i].0.contains_key(k)));
            assert!(mappings[i].0.keys().all(|k| read.contains_key(k)));

            for k in read.keys() {
                let read_val = read.get(k).unwrap();
                let actual_val = mappings[i].0.get(k).unwrap();

                println!("{:?}", k);
                println!("{:?} {:?}", read_val, actual_val);

                assert_eq!(read_val.0, actual_val.0);
                assert!(
                    (read_val.1 - actual_val.1).abs()
                        <= 4. / 2u32.pow(crate::naive::io::F32_BITS as u32) as f32
                );
                assert!(
                    (read_val.2 - actual_val.2).abs()
                        <= 4. / 2u32.pow(crate::naive::io::F32_BITS as u32) as f32
                );
            }
        }
    }
}
