//! This crate provides an encoder/decoder for Reed-Solomon erasure code.
//!
//! Please note that erasure coding means errors are not directly detected or corrected,
//! but missing data pieces (shards) can be reconstructed given that
//! the configuration provides high enough redundancy.
//!
//! You will have to implement error detection separately (e.g. via checksums)
//! and simply leave out the corrupted shards when attempting to reconstruct
//! the missing data.

#![cfg_attr(
    feature = "unstable",
    feature(
        stdsimd,
        avx512_target_feature,
        arm_target_feature
    )
)]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    // clippy::cargo,
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cognitive_complexity,
    clippy::doc_markdown,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::shadow_unrelated,
    clippy::similar_names,
    clippy::too_many_lines
)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::iter::{self, FromIterator};

#[macro_use]
mod macros;

mod core;
mod errors;
mod inversion_tree;
mod matrix;
mod platform;

#[cfg(test)]
mod tests;

pub mod galois_16;
pub mod galois_8;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod galois_8_avx2;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "unstable"))]
mod galois_8_avx512;
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", feature = "unstable")
))]
mod galois_8_neon;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod galois_8_sse3;

pub use crate::errors::Error;
pub use crate::errors::SBSError;

pub use crate::core::ReedSolomon;
pub use crate::core::ShardByShard;

type Result<T> = std::result::Result<T, std::result::Result<T, Error>>;

/// A finite field to perform encoding over.
pub trait Field: Sized {
    /// The order of the field. This is a limit on the number of shards
    /// in an encoding.
    const ORDER: usize;

    /// The representational type of the field.
    type Elem: Default + Clone + Copy + PartialEq + std::fmt::Debug;

    /// Add two elements together.
    fn add(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// Multiply two elements together.
    fn mul(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// Divide a by b. Panics is b is zero.
    fn div(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// Raise `a` to the n'th power.
    fn exp(a: Self::Elem, n: usize) -> Self::Elem;

    /// The "zero" element or additive identity.
    fn zero() -> Self::Elem;

    /// The "one" element or multiplicative identity.
    fn one() -> Self::Elem;

    fn nth_internal(n: usize) -> Self::Elem;

    /// Yield the nth element of the field. Panics if n >= ORDER.
    /// Assignment is arbitrary but must be unique to `n`.
    fn nth(n: usize) -> Self::Elem {
        if n >= Self::ORDER {
            let pow = (Self::ORDER as f32).log2() as usize;
            panic!("{} out of bounds for GF(2^{}) member", n, pow);
        }

        Self::nth_internal(n)
    }

    /// Multiply a slice of elements by another. Writes into the output slice.
    ///
    /// # Panics
    /// Panics if the output slice does not have equal length to the input.
    fn mul_slice(elem: Self::Elem, input: &[Self::Elem], out: &mut [Self::Elem]) {
        assert_eq!(input.len(), out.len());

        for (i, o) in input.iter().zip(out) {
            *o = Self::mul(elem, *i);
        }
    }

    /// Multiply a slice of elements by another, adding each result to the corresponding value in
    /// `out`.
    ///
    /// # Panics
    /// Panics if the output slice does not have equal length to the input.
    fn mul_slice_add(elem: Self::Elem, input: &[Self::Elem], out: &mut [Self::Elem]) {
        assert_eq!(input.len(), out.len());

        for (i, o) in input.iter().zip(out) {
            *o = Self::add(*o, Self::mul(elem, *i));
        }
    }
}

/// Something which might hold a shard.
///
/// This trait is used in reconstruction, where some of the shards
/// may be unknown.
pub trait ReconstructShard<F: Field> {
    /// The size of the shard data; `None` if empty.
    fn len(&self) -> Option<usize>;

    /// Returns `true` if the slice has a length of 0.
    fn is_empty(&self) -> bool {
        self.len().is_none()
    }

    /// Get a mutable reference to the shard data, returning `None` if uninitialized.
    fn get(&mut self) -> Option<&mut [F::Elem]>;

    /// Get a mutable reference to the shard data, initializing it to the
    /// given length if it was `None`. Returns an error if initialization fails.
    fn get_or_initialize(&mut self, len: usize) -> Result<&mut [F::Elem]>;
}

impl<F: Field, T: AsRef<[F::Elem]> + AsMut<[F::Elem]> + FromIterator<F::Elem>> ReconstructShard<F>
    for Option<T>
{
    fn len(&self) -> Option<usize> {
        self.as_ref().map(|x| x.as_ref().len())
    }

    fn get(&mut self) -> Option<&mut [F::Elem]> {
        self.as_mut().map(AsMut::as_mut)
    }

    fn get_or_initialize(&mut self, len: usize) -> Result<&mut [F::Elem]> {
        let is_some = self.is_some();
        let x = self
            .get_or_insert_with(|| iter::repeat(F::zero()).take(len).collect())
            .as_mut();

        if is_some {
            Ok(x)
        } else {
            Err(Ok(x))
        }
    }
}

impl<F: Field, T: AsRef<[F::Elem]> + AsMut<[F::Elem]>> ReconstructShard<F> for (T, bool) {
    fn len(&self) -> Option<usize> {
        if self.1 {
            Some(self.0.as_ref().len())
        } else {
            None
        }
    }

    fn get(&mut self) -> Option<&mut [F::Elem]> {
        if self.1 {
            Some(self.0.as_mut())
        } else {
            None
        }
    }

    fn get_or_initialize(&mut self, len: usize) -> Result<&mut [F::Elem]> {
        let x = self.0.as_mut();
        if x.len() == len {
            if self.1 {
                Ok(x)
            } else {
                Err(Ok(x))
            }
        } else {
            Err(Err(Error::IncorrectShardSize))
        }
    }
}
