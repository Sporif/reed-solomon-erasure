#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _mm256_and_si256, _mm256_broadcastsi128_si256, _mm256_loadu_si256,
    _mm256_set1_epi8, _mm256_shuffle_epi8, _mm256_srli_epi64, _mm256_storeu_si256,
    _mm256_xor_si256, _mm_loadu_si128,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_and_si256, _mm256_broadcastsi128_si256, _mm256_loadu_si256,
    _mm256_set1_epi8, _mm256_shuffle_epi8, _mm256_srli_epi64, _mm256_storeu_si256,
    _mm256_xor_si256, _mm_loadu_si128,
};

type Vec128 = __m128i;
type Vec256 = __m256i;
type Vec = Vec256;

#[inline(always)]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn loadu_v128(in_0: *const u8) -> Vec128 {
    _mm_loadu_si128(in_0.cast::<__m128i>())
}

#[inline(always)]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn loadu_v(in_0: *const u8) -> Vec {
    _mm256_loadu_si256(in_0.cast::<__m256i>())
}

#[inline(always)]
unsafe fn set1_epi8_v(c: i8) -> Vec {
    _mm256_set1_epi8(c)
}

#[inline(always)]
unsafe fn srli_epi64_v<const N: i32>(in_0: Vec) -> Vec {
    _mm256_srli_epi64::<N>(in_0)
}

#[inline(always)]
unsafe fn and_v(a: Vec, b: Vec) -> Vec {
    _mm256_and_si256(a, b)
}

#[inline(always)]
unsafe fn xor_v(a: Vec, b: Vec) -> Vec {
    _mm256_xor_si256(a, b)
}

#[inline(always)]
unsafe fn shuffle_epi8_v(vec: Vec, mask: Vec) -> Vec {
    _mm256_shuffle_epi8(vec, mask)
}

#[inline(always)]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn storeu_v(out: *mut u8, vec: Vec) {
    _mm256_storeu_si256(out.cast::<__m256i>(), vec);
}

#[inline(always)]
unsafe fn replicate_v128_v(vec: Vec128) -> Vec {
    _mm256_broadcastsi128_si256(vec)
}

#[inline(always)]
unsafe fn gal_mul_v(
    low_mask_unpacked: Vec,
    low_vector: Vec,
    high_vector: Vec,
    modifier: Option<unsafe fn(_: Vec, _: Vec) -> Vec>,
    in_x: Vec,
    old: Vec,
) -> Vec {
    let low_input = and_v(in_x, low_mask_unpacked);
    let in_x_shifted = srli_epi64_v::<4>(in_x);
    let high_input = and_v(in_x_shifted, low_mask_unpacked);
    let mul_low_part = shuffle_epi8_v(low_vector, low_input);
    let mul_high_part = shuffle_epi8_v(high_vector, high_input);
    let new = xor_v(mul_low_part, mul_high_part);

    modifier.expect("non-null function pointer")(new, old)
}

#[inline(always)]
unsafe fn gal_mul_impl(
    low: *const u8,
    high: *const u8,
    in_0: *const u8,
    out: *mut u8,
    len: usize,
    modifier: Option<unsafe fn(_: Vec, _: Vec) -> Vec>,
) -> usize {
    let low_mask_unpacked = set1_epi8_v(0xf_i8);
    let low_vector128 = loadu_v128(low);
    let high_vector128 = loadu_v128(high);
    let low_vector = replicate_v128_v(low_vector128);
    let high_vector = replicate_v128_v(high_vector128);
    let mut done = 0;
    let mut x = 0;

    let s_v = std::mem::size_of::<Vec>();
    while x < len.wrapping_div(s_v) {
        let in_x = loadu_v(&*in_0.add(done));
        let old = loadu_v(&*out.add(done));
        let result = gal_mul_v(
            low_mask_unpacked,
            low_vector,
            high_vector,
            modifier,
            in_x,
            old,
        );
        storeu_v(&mut *out.add(done), result);
        done = done.wrapping_add(s_v) as usize;
        x = x.wrapping_add(1);
    }

    done
}

#[inline(always)]
const fn noop(new: Vec, _old: Vec) -> Vec {
    new
}

/// # Safety
///
///
#[target_feature(enable = "avx2")]
pub unsafe fn gal_mul(
    low: *const u8,
    high: *const u8,
    in_0: *const u8,
    out: *mut u8,
    len: usize,
) -> usize {
    gal_mul_impl(
        low,
        high,
        in_0,
        out,
        len,
        Some(noop as unsafe fn(_: Vec, _: Vec) -> Vec),
    )
}

/// # Safety
///
///
#[target_feature(enable = "avx2")]
pub unsafe fn gal_mul_xor(
    low: *const u8,
    high: *const u8,
    in_0: *const u8,
    out: *mut u8,
    len: usize,
) -> usize {
    gal_mul_impl(
        low,
        high,
        in_0,
        out,
        len,
        Some(xor_v as unsafe fn(_: Vec, _: Vec) -> Vec),
    )
}
