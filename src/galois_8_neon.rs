#[cfg(target_arch = "arm")]
use std::arch::arm::{uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vshrq_n_u8};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vqtbl1q_u8, vshrq_n_u8};

use std::mem::size_of;
use std::ptr::copy_nonoverlapping as copy;

#[derive(Copy, Clone)]
#[allow(dead_code)]
union Vec128 {
    u8_0: [u8; 16],
    u64_0: [u64; 2],
    uint8x16: uint8x16_t,
}
type Vec = Vec128;

#[inline(always)]
unsafe fn loadu_v128(in_0: *const u8) -> Vec128 {
    let mut out = Vec128 { u64_0: [0; 2] };
    let out_ptr = (&mut out.u64_0).as_mut_ptr() as *mut u8;
    copy(in_0, out_ptr, size_of::<[u64; 2]>());
    out
}

#[inline(always)]
unsafe fn loadu_v(in_0: *const u8) -> Vec {
    loadu_v128(in_0)
}

#[inline(always)]
unsafe fn set1_epi8_v(c: u8) -> Vec {
    Vec {
        uint8x16: vdupq_n_u8(c),
    }
}

#[inline(always)]
unsafe fn srli_epi64_v<const N: i32>(in_0: Vec) -> Vec {
    Vec {
        uint8x16: vshrq_n_u8(in_0.uint8x16, N),
    }
}

#[inline(always)]
unsafe fn and_v(a: Vec, b: Vec) -> Vec {
    Vec {
        uint8x16: vandq_u8(a.uint8x16, b.uint8x16),
    }
}

#[inline(always)]
unsafe fn xor_v(a: Vec, b: Vec) -> Vec {
    Vec {
        uint8x16: veorq_u8(a.uint8x16, b.uint8x16),
    }
}

#[inline(always)]
unsafe fn shuffle_epi8_v(vec: Vec, mask: Vec) -> Vec {
    #[cfg(target_arch = "aarch64")]
    {
        Vec {
            uint8x16: vqtbl1q_u8(vec.uint8x16, mask.uint8x16),
        }
    }
    #[cfg(target_arch = "arm")]
    {
        let mut out = Vec { u64_0: [0; 2] };
        let mut do_byte = |i| {
            out.u8_0[i] = if (mask.u8_0[i] & 0x80) == 0 {
                vec.u8_0[(mask.u8_0[i] & 0x0F) as usize]
            } else {
                0
            }
        };

        do_byte(0);
        do_byte(1);
        do_byte(2);
        do_byte(3);
        do_byte(4);
        do_byte(5);
        do_byte(6);
        do_byte(7);
        do_byte(8);
        do_byte(9);
        do_byte(10);
        do_byte(11);
        do_byte(12);
        do_byte(13);
        do_byte(14);
        do_byte(15);

        out
    }
}

#[inline(always)]
unsafe fn storeu_v(out: *mut u8, vec: Vec) {
    let vec_ptr = (&vec.u64_0).as_ptr() as *const u8;
    copy(vec_ptr, out, size_of::<[u64; 2]>());
}

#[inline(always)]
const fn replicate_v128_v(vec: Vec128) -> Vec {
    vec
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
    let low_mask_unpacked = set1_epi8_v(0xf_u8);
    let low_vector128 = loadu_v128(low);
    let high_vector128 = loadu_v128(high);
    let low_vector = replicate_v128_v(low_vector128);
    let high_vector = replicate_v128_v(high_vector128);
    let mut done = 0;
    let mut x = 0;

    let s_v = size_of::<Vec>();
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
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
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
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
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
