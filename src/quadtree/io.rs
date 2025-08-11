use std::{
    collections::HashMap,
    io::{Read, Write},
};

use super::quadtree::*;
use crate::{
    compression::{DomainBlockLocation, Mappings, RangeBlockLocation},
    naive::io::{BinBufReader, BinBufWriter},
    quadtree::QuadtreeSettings,
};

pub fn write_quadtree<T: Write>(
    w: &mut BinBufWriter<T>,
    t: &Quadtree<RangeBlockLocation>,
) -> Result<(), std::io::Error> {
    if let Some(b) = t.children.as_ref() {
        w.add_bit(false)?;
        write_quadtree(w, &b[0])?;
        write_quadtree(w, &b[1])?;
        write_quadtree(w, &b[2])?;
        write_quadtree(w, &b[3])
    } else {
        w.add_bit(true)
    }
}

pub fn read_quadtree<T: Read>(
    r: &mut BinBufReader<T>,
    size: (usize, usize),
    offset: (usize, usize),
) -> Result<Quadtree<RangeBlockLocation>, std::io::Error> {
    let b = r.read_bit()?;

    if b {
        Ok(Quadtree::leaf(RangeBlockLocation { size, pos: offset }))
    } else {
        let c1 = read_quadtree(r, (size.0 / 2, size.1 / 2), offset)?;
        let c2 = read_quadtree(
            r,
            (size.0 / 2, size.1 - size.1 / 2),
            (offset.0, offset.1 + size.1 / 2),
        )?;
        let c3 = read_quadtree(
            r,
            (size.0 - size.0 / 2, size.1 / 2),
            (offset.0 + size.0 / 2, offset.1),
        )?;
        let c4 = read_quadtree(
            r,
            (size.0 - size.0 / 2, size.1 - size.1 / 2),
            (offset.0 + size.0 / 2, offset.1 + size.1 / 2),
        )?;

        Ok(Quadtree::node(
            RangeBlockLocation { pos: offset, size },
            c1,
            c2,
            c3,
            c4,
        ))
    }
}

pub fn save_mappings(
    file: String,
    mappings: &(
        Mappings,
        Quadtree<RangeBlockLocation>,
        Quadtree<RangeBlockLocation>,
    ),
    s: QuadtreeSettings,
) -> Result<usize, std::io::Error> {
    let mut writer = BinBufWriter {
        x: 0,
        n: 0,
        size_written: 0,
        buf: std::fs::File::create(file)?,
    };

    // saves the settings
    writer.write_float(s.min_domain_variance)?;
    writer.write_float(s.min_range_variance)?;
    writer.add_bit(s.only_leaves)?;
    writer.write_int(s.min_domain_block_size, 8)?;
    writer.write_int(s.min_range_block_size, 8)?;
    writer.write_int(s.minimum_range_splits, 4)?;

    // saves the image size
    let (h, w) = mappings.1.label.size;
    writer.write_int(h, 32)?;
    writer.write_int(w, 32)?;

    // saves the quadtrees
    write_quadtree(&mut writer, &mappings.1)?;
    write_quadtree(&mut writer, &mappings.2)?;

    // saves transformations. domain blocks are identified by their index in the prefix traversal
    // of the quadtree
    let domain_blocks = if s.only_leaves {
        mappings.2.prefix_leaves()
    } else {
        mappings.2.prefix_traversal()
    };
    let domain_blocks_indices: HashMap<RangeBlockLocation, usize> = domain_blocks
        .iter()
        .enumerate()
        .map(|(i, &db)| (db, i))
        .collect();
    let bits_domain_blocks = domain_blocks_indices.len().ilog2() as usize + 1;

    for rb in mappings.1.prefix_leaves() {
        let &(db, c, b) = mappings.0.get(&rb).unwrap();

        let db_index = *domain_blocks_indices
            .get(&RangeBlockLocation {
                pos: db.pos,
                size: db.size,
            })
            .unwrap();

        writer.write_int(db_index, bits_domain_blocks)?;
        writer.write_rotation(db.rotation)?;
        writer.add_bit(db.flipped)?;
        writer.write_float(c)?;
        writer.write_float(b)?;
    }
    Ok(writer.size_written.div_ceil(8))
}

pub fn load_mappings(
    file: String,
) -> Result<
    (
        QuadtreeSettings,
        (usize, usize),
        (
            Mappings,
            Quadtree<RangeBlockLocation>,
            Quadtree<RangeBlockLocation>,
        ),
    ),
    std::io::Error,
> {
    let mut reader = BinBufReader::new(std::fs::File::open(file)?);

    let mut s = QuadtreeSettings::default();

    s.min_domain_variance = reader.read_float()?;
    s.min_range_variance = reader.read_float()?;
    s.only_leaves = reader.read_bit()?;
    s.min_domain_block_size = reader.read_int(8)?;
    s.min_range_block_size = reader.read_int(8)?;
    s.minimum_range_splits = reader.read_int(4)?;

    let (h, w) = (reader.read_int(32)?, reader.read_int(32)?);

    let range_blocks_qt = read_quadtree(&mut reader, (h, w), (0, 0))?;
    let domain_blocks_qt = read_quadtree(&mut reader, (h, w), (0, 0))?;

    //
    let domain_blocks = if s.only_leaves {
        domain_blocks_qt.prefix_leaves()
    } else {
        domain_blocks_qt.prefix_traversal()
    };

    let bits_domain_blocks = domain_blocks.len().ilog2() as usize + 1;

    let mut mappings = HashMap::new();

    for rb in range_blocks_qt.prefix_leaves() {
        let db_index = reader.read_int(bits_domain_blocks)?;
        let rotation = reader.read_rotation()?;
        let flipped = reader.read_bit()?;
        let c = reader.read_float()?;
        let b = reader.read_float()?;

        let RangeBlockLocation { pos, size } = domain_blocks[db_index];

        mappings.insert(
            rb,
            (
                DomainBlockLocation {
                    pos,
                    size,
                    rotation,
                    flipped,
                },
                c,
                b,
            ),
        );
    }

    Ok((s, (h, w), (mappings, range_blocks_qt, domain_blocks_qt)))
}

#[cfg(test)]
mod tests {
    use crate::quadtree::{QuadtreeSettings, generate_range_blocks};

    use super::*;
    #[test]
    fn quadtree_saving() {
        let mut buf = [0; 4000];

        let mut w = BinBufWriter::new(buf.as_mut_slice());

        let imgs = (0..100)
            .map(|_| ndarray::Array2::from_shape_fn((11, 10), |_| rand::random()))
            .collect::<Vec<_>>();

        let ts = imgs
            .iter()
            .map(|img| {
                generate_range_blocks(
                    img.view(),
                    QuadtreeSettings {
                        min_domain_variance: 0.,
                        min_range_variance: 0.1,
                        min_domain_block_size: 0,
                        min_range_block_size: 1,
                        only_leaves: false,
                        minimum_range_splits: 0,
                    },
                    (0, 0),
                    0,
                )
            })
            .collect::<Vec<_>>();
        let _ = ts
            .iter()
            .map(|t| write_quadtree(&mut w, t).unwrap())
            .collect::<Vec<_>>();
        drop(w);

        let mut r = BinBufReader {
            x: [0],
            n: 0,
            buf: buf.as_slice(),
        };

        let read_ts = (0..100)
            .map(|_| read_quadtree(&mut r, (11, 10), (0, 0)).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(ts, read_ts);
    }
}
