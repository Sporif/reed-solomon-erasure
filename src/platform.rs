#![allow(
    dead_code,
    unreachable_code,
    clippy::missing_const_for_fn,
    clippy::upper_case_acronyms
)]

#[derive(Clone, Copy, Debug)]
pub enum Platform {
    Portable,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE3,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX512,
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    NEON,
}

impl Platform {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if avx512_detected() {
                return Self::AVX512;
            }
            if avx2_detected() {
                return Self::AVX2;
            }
            if sse3_detected() {
                return Self::SSE3;
            }
        }

        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        {
            if neon_detected() {
                return Self::NEON;
            }
        }

        Self::Portable
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn avx512_detected() -> bool {
    if cfg!(feature = "no_avx512") {
        return false;
    }
    #[cfg(target_feature = "avx512f")]
    {
        return true;
    }
    is_x86_feature_detected!("avx512f")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn avx2_detected() -> bool {
    if cfg!(feature = "no_avx2") {
        return false;
    }
    #[cfg(target_feature = "avx2")]
    {
        return true;
    }
    is_x86_feature_detected!("avx2")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn sse3_detected() -> bool {
    if cfg!(feature = "no_sse3") {
        return false;
    }
    #[cfg(target_feature = "sse3")]
    {
        return true;
    }
    is_x86_feature_detected!("sse3")
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[inline(always)]
pub fn neon_detected() -> bool {
    if cfg!(feature = "no_neon") {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Neon is always enabled on android aarch64 targets
        #[cfg(any(target_feature = "neon", target_os = "android"))]
        {
            return true;
        }
        return std::arch::is_aarch64_feature_detected!("neon");
    }
    #[cfg(all(target_arch = "arm", feature = "unstable"))]
    {
        #[cfg(all(target_feature = "neon", target_feature = "v7"))]
        {
            return true;
        }
        return std::arch::is_arm_feature_detected!("neon") &&
               std::arch::is_arm_feature_detected!("v7");
    }
    false
}
