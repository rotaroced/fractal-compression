#![allow(unused)]

use crate::CompressionMethod;
use crate::compression::*;
use crate::naive::NaiveCompressionSettings;
use std::collections::VecDeque;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;
use std::ptr::read;

pub struct BinBufWriter<T: Write> {
    pub(crate) x: u64,
    pub(crate) n: usize,
    pub(crate) buf: T,
    pub(crate) size_written: usize,
}

pub struct BinBufReader<T: Read> {
    pub(crate) x: [u8; 1],
    pub(crate) n: usize,
    pub(crate) buf: T,
}

pub fn save_mappings(
    m: &Mappings,
    file: String,
    mut s: NaiveCompressionSettings,
) -> Result<usize, std::io::Error> {
    let mut writer = BinBufWriter::new(std::fs::File::create(file)?);

    let h = s.range_block_size + m.keys().map(|rb| rb.pos.0).max().unwrap();
    let w = s.range_block_size + m.keys().map(|rb| rb.pos.1).max().unwrap();

    // saves the NaiveCompressionSettings
    writer.write_int(s.range_block_size, 32)?;
    writer.write_int(s.domain_block_size, 32)?;
    writer.write_int(s.domain_block_stepx, 32)?;
    writer.write_int(s.domain_block_stepy, 32)?;
    let q = std::cmp::max(h, w);
    s.coord_bits = q.ilog2() as usize + 1;
    writer.write_int(s.coord_bits, 15)?;

    // writes the size of the image
    writer.write_int(h / s.range_block_size, 32)?;
    writer.write_int(w / s.range_block_size, 32)?;

    // saves all of the transformations
    for i in 0..(h / s.range_block_size) {
        for j in 0..(w / s.range_block_size) {
            let rb = RangeBlockLocation {
                pos: (i * s.range_block_size, j * s.range_block_size),
                size: (s.range_block_size, s.range_block_size),
            };

            let &(db, c, b) = m.get(&rb).unwrap();

            writer.write_int(db.pos.0 / s.domain_block_stepy, s.coord_bits)?;
            writer.write_int(db.pos.1 / s.domain_block_stepx, s.coord_bits)?;
            writer.add_bit(db.flipped)?;
            writer.write_rotation(db.rotation)?;
            writer.write_float(c)?;
            writer.write_float(b)?;
        }
    }

    Ok(writer.size_written)
}

pub fn load_mappings(
    file: String,
) -> Result<(NaiveCompressionSettings, (usize, usize), Mappings), std::io::Error> {
    let mut reader = BinBufReader::new(std::fs::File::open(file)?);

    let mut mappings = Mappings::default();

    let mut s = NaiveCompressionSettings::default();
    s.range_block_size = reader.read_int(32)?;
    s.domain_block_size = reader.read_int(32)?;
    s.domain_block_stepx = reader.read_int(32)?;
    s.domain_block_stepy = reader.read_int(32)?;
    s.coord_bits = reader.read_int(15)?;

    let h = reader.read_int(32)? * s.range_block_size;
    let w = reader.read_int(32)? * s.range_block_size;

    for i in 0..(h / s.range_block_size) {
        for j in 0..(w / s.range_block_size) {
            let py = reader.read_int(s.coord_bits)?;
            let px = reader.read_int(s.coord_bits)?;
            let flipped = reader.read_bit()?;
            let rotation = reader.read_rotation()?;

            let c: f32 = reader.read_float()?;
            let b: f32 = reader.read_float()?;
            mappings.insert(
                RangeBlockLocation {
                    pos: (i * s.range_block_size, j * s.range_block_size),
                    size: (s.range_block_size, s.range_block_size),
                },
                (
                    DomainBlockLocation {
                        size: (s.domain_block_size, s.domain_block_size),
                        flipped,
                        rotation,
                        pos: (py, px),
                    },
                    c,
                    b,
                ),
            );
        }
    }

    Ok((s, (h, w), mappings))
}

impl<T: Write> BinBufWriter<T> {
    #[inline]
    pub fn new(w: T) -> Self {
        Self {
            buf: w,
            n: 0,
            x: 0,
            size_written: 0,
        }
    }

    pub fn add_bit(&mut self, bit: bool) -> Result<(), std::io::Error> {
        self.x += (bit as u64) << self.n;
        self.n += 1;
        // println!("{}, {}", self.n, self.x);
        if self.n >= 8 {
            for _ in 0..=0 {
                self.buf.write_all(&[(self.x % 256) as u8])?;
                self.x >>= 8;
                self.n -= 8;
            }
        }

        self.size_written += 1;

        Ok(())
    }

    pub fn add_bits(&mut self, bits: &[bool]) -> Result<(), std::io::Error> {
        for &b in bits {
            self.add_bit(b)?;
        }
        Ok(())
    }

    pub fn write_byte(&mut self, mut byte: u8) -> Result<(), std::io::Error> {
        for i in (0..8).rev() {
            self.add_bit((byte >> i) & 1 == 1);
        }
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), std::io::Error> {
        while self.n > 0 {
            self.buf.write_all(&[(self.x % 256) as u8])?;
            self.x >>= 8;
            self.n = self.n.saturating_sub(8);
        }
        self.buf.flush();
        Ok(())
    }
    pub fn write_int(&mut self, n: usize, size: usize) -> Result<(), std::io::Error> {
        debug_assert!(n < ((1 << size) - 1));
        for i in 0..size {
            self.add_bit((n >> i) & 1 == 1)?;
        }
        Ok(())
    }

    pub fn write_float(&mut self, x: f32) -> Result<(), std::io::Error> {
        for b in x.to_le_bytes() {
            self.write_byte(b);
        }

        Ok(())
    }

    pub fn write_rotation(&mut self, r: Rotation) -> Result<(), std::io::Error> {
        self.add_bit((r == Rotation::Quarter) || (r == Rotation::ThreeQuarter))?;
        self.add_bit((r == Rotation::ThreeQuarter) || (r == Rotation::Half))
    }
}

impl<T: Write> Drop for BinBufWriter<T> {
    fn drop(&mut self) {
        self.close().unwrap();
    }
}

impl<T: Read> BinBufReader<T> {
    pub fn new(r: T) -> Self {
        Self {
            x: [0],
            n: 0,
            buf: r,
        }
    }

    pub fn read_bit(&mut self) -> Result<bool, std::io::Error> {
        if self.n == 0 {
            self.buf.read_exact(&mut self.x)?;
            self.n += 8;
        }

        let res = (self.x[0] & 1) == 1;
        self.x[0] >>= 1;
        self.n -= 1;
        Ok(res)
    }

    pub fn read_byte(&mut self) -> Result<u8, std::io::Error> {
        let mut x = 0;
        for _ in 0..8 {
            x = x * 2 + (self.read_bit()? as u8);
        }
        Ok(x)
    }

    pub fn read_int(&mut self, size: usize) -> Result<usize, std::io::Error> {
        let mut x = 0;
        for i in 0..size {
            x += (self.read_bit()? as usize) << i;
        }
        Ok(x)
    }

    pub fn read_float(&mut self) -> Result<f32, std::io::Error> {
        Ok(f32::from_le_bytes([
            self.read_byte()?,
            self.read_byte()?,
            self.read_byte()?,
            self.read_byte()?,
        ]))
    }

    pub fn read_rotation(&mut self) -> Result<Rotation, std::io::Error> {
        let b1 = self.read_bit()?;
        let b2 = self.read_bit()?;
        Ok(match (b1, b2) {
            (false, false) => Rotation::Zero,
            (true, false) => Rotation::Quarter,
            (false, true) => Rotation::Half,
            (true, true) => Rotation::ThreeQuarter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::compression::*;

    #[allow(non_upper_case_globals)]
    const s: NaiveCompressionSettings = NaiveCompressionSettings {
        domain_block_size: 14,
        range_block_size: 8,
        domain_block_stepx: 9,
        domain_block_stepy: 3,
        coord_bits: 13,
    };

    #[test]
    fn test_io_bit() -> Result<(), std::io::Error> {
        let mut buf = [0, 0, 0, 0, 0, 0];

        let mut w = BinBufWriter::new(buf.as_mut_slice());

        let data = [
            true, true, false, false, true, false, false, false, false, false, false, false, false,
            true, true, false, false, true, false, false, true, false, false, true, true,
        ];

        w.add_bits(&data)?;

        drop(w);

        println!("{buf:?}\n");

        let mut r = BinBufReader {
            x: [0],
            n: 0,
            buf: buf.as_slice(),
        };

        for _ in 0..data.len() {
            println!("{}", r.read_bit()?);
        }

        Ok(())
    }

    #[test]
    fn tests_structs_io() -> Result<(), std::io::Error> {
        let mut buf = [0; 500];

        let mut w = BinBufWriter::new(buf.as_mut_slice());

        w.add_bit(true);

        let rb = RangeBlockLocation {
            pos: (18 * s.range_block_size, 33 * s.range_block_size),
            size: (s.range_block_size, s.range_block_size),
        };
        let db = DomainBlockLocation {
            pos: (7 * s.domain_block_stepy, 36 * s.domain_block_stepx),
            flipped: true,
            rotation: crate::compression::Rotation::ThreeQuarter,
            size: (s.domain_block_size, s.domain_block_size),
        };

        w.write_int(69, 21)?;
        w.write_float(0.861)?;
        w.write_rotation(Rotation::Quarter)?;
        w.write_rotation(Rotation::Zero)?;
        w.write_rotation(Rotation::ThreeQuarter)?;
        w.write_byte(42)?;
        w.write_rotation(Rotation::Half)?;

        drop(w);

        println!("{buf:?}\n");

        let mut r = BinBufReader {
            x: [0],
            n: 0,
            buf: buf.as_slice(),
        };

        r.read_bit()?;

        println!("int");
        assert_eq!(69, r.read_int(21)?);
        println!("f32_block");
        assert_eq!(r.read_float()?, (0.861));
        println!("rotation 0");
        assert_eq!(r.read_rotation()?, Rotation::Quarter);
        println!("rotation 0");
        assert_eq!(r.read_rotation()?, Rotation::Zero);
        println!("rotation 0");
        assert_eq!(r.read_rotation()?, Rotation::ThreeQuarter);
        println!("byte");
        assert_eq!(42, r.read_byte()?);
        println!("rotation 0");
        assert_eq!(r.read_rotation()?, Rotation::Half);

        Ok(())
    }
}
