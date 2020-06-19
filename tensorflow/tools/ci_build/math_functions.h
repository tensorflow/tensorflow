/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include "hip_fp16_math_fwd.h"
#include "hip_vector_types.h"
#include "math_fwd.h"

#include <hip/hcc_detail/host_defines.h>

#include <algorithm>

// assert.h is only for the host version of assert.
// The device version of assert is implemented in hip/hcc_detail/hip_runtime.h.
// Users should include hip_runtime.h for the device version of assert.
#if !__HIP_DEVICE_COMPILE__
#include <assert.h>
#endif

#include <limits.h>
#include <limits>
#include <stdint.h>

// HCC's own math functions should be included first, otherwise there will
// be conflicts when hip/math_functions.h is included before hip/hip_runtime.h.
#ifdef __HCC__
#include "kalmar_math.h"
#endif

#if _LIBCPP_VERSION && __HIP__
namespace std {
template <>
struct __numeric_type<_Float16>
{
   static _Float16 __test(_Float16);

   typedef _Float16 type;
   static const bool value = true;
};
}
#endif // _LIBCPP_VERSION

#pragma push_macro("__DEVICE__")
#pragma push_macro("__RETURN_TYPE")

#ifdef __HCC__
#define __DEVICE__ __device__
#define __RETURN_TYPE int
#else // to be consistent with __clang_cuda_math_forward_declares
#define __DEVICE__ static __device__
#define __RETURN_TYPE bool
#endif

__DEVICE__
inline
uint64_t __make_mantissa_base8(const char* tagp)
{
    uint64_t r = 0;
    while (tagp) {
        char tmp = *tagp;

        if (tmp >= '0' && tmp <= '7') r = (r * 8u) + tmp - '0';
        else return 0;

        ++tagp;
    }

    return r;
}

__DEVICE__
inline
uint64_t __make_mantissa_base10(const char* tagp)
{
    uint64_t r = 0;
    while (tagp) {
        char tmp = *tagp;

        if (tmp >= '0' && tmp <= '9') r = (r * 10u) + tmp - '0';
        else return 0;

        ++tagp;
    }

    return r;
}

__DEVICE__
inline
uint64_t __make_mantissa_base16(const char* tagp)
{
    uint64_t r = 0;
    while (tagp) {
        char tmp = *tagp;

        if (tmp >= '0' && tmp <= '9') r = (r * 16u) + tmp - '0';
        else if (tmp >= 'a' && tmp <= 'f') r = (r * 16u) + tmp - 'a' + 10;
        else if (tmp >= 'A' && tmp <= 'F') r = (r * 16u) + tmp - 'A' + 10;
        else return 0;

        ++tagp;
    }

    return r;
}

__DEVICE__
inline
uint64_t __make_mantissa(const char* tagp)
{
    if (!tagp) return 0u;

    if (*tagp == '0') {
        ++tagp;

        if (*tagp == 'x' || *tagp == 'X') return __make_mantissa_base16(tagp);
        else return __make_mantissa_base8(tagp);
    }

    return __make_mantissa_base10(tagp);
}

// DOT FUNCTIONS
#if (__hcc_workweek__ >= 19015) || __HIP_CLANG_ONLY__
__DEVICE__
inline
int amd_mixed_dot(short2 a, short2 b, int c, bool saturate) {
    return __ockl_sdot2(a.data, b.data, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(ushort2 a, ushort2 b, uint c, bool saturate) {
    return __ockl_udot2(a.data, b.data, c, saturate);
}
__DEVICE__
inline
int amd_mixed_dot(char4 a, char4 b, int c, bool saturate) {
    return __ockl_sdot4(a.data, b.data, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate) {
    return __ockl_udot4(a.data, b.data, c, saturate);
}
__DEVICE__
inline
int amd_mixed_dot(int a, int b, int c, bool saturate) {
    return __ockl_sdot8(a, b, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(uint a, uint b, uint c, bool saturate) {
    return __ockl_udot8(a, b, c, saturate);
}
#endif

// BEGIN FLOAT
__DEVICE__
inline
float abs(float x) { return __ocml_fabs_f32(x); }
__DEVICE__
inline
float acosf(float x) { return __ocml_acos_f32(x); }
__DEVICE__
inline
float acoshf(float x) { return __ocml_acosh_f32(x); }
__DEVICE__
inline
float asinf(float x) { return __ocml_asin_f32(x); }
__DEVICE__
inline
float asinhf(float x) { return __ocml_asinh_f32(x); }
__DEVICE__
inline
float atan2f(float x, float y) { return __ocml_atan2_f32(x, y); }
__DEVICE__
inline
float atanf(float x) { return __ocml_atan_f32(x); }
__DEVICE__
inline
float atanhf(float x) { return __ocml_atanh_f32(x); }
__DEVICE__
inline
float cbrtf(float x) { return __ocml_cbrt_f32(x); }
__DEVICE__
inline
float ceilf(float x) { return __ocml_ceil_f32(x); }
__DEVICE__
inline
float copysignf(float x, float y) { return __ocml_copysign_f32(x, y); }
__DEVICE__
inline
float cosf(float x) { return __ocml_cos_f32(x); }
__DEVICE__
inline
float coshf(float x) { return __ocml_cosh_f32(x); }
__DEVICE__
inline
float cospif(float x) { return __ocml_cospi_f32(x); }
__DEVICE__
inline
float cyl_bessel_i0f(float x) { return __ocml_i0_f32(x); }
__DEVICE__
inline
float cyl_bessel_i1f(float x) { return __ocml_i1_f32(x); }
__DEVICE__
inline
float erfcf(float x) { return __ocml_erfc_f32(x); }
__DEVICE__
inline
float erfcinvf(float x) { return __ocml_erfcinv_f32(x); }
__DEVICE__
inline
float erfcxf(float x) { return __ocml_erfcx_f32(x); }
__DEVICE__
inline
float erff(float x) { return __ocml_erf_f32(x); }
__DEVICE__
inline
float erfinvf(float x) { return __ocml_erfinv_f32(x); }
__DEVICE__
inline
float exp10f(float x) { return __ocml_exp10_f32(x); }
__DEVICE__
inline
float exp2f(float x) { return __ocml_exp2_f32(x); }
__DEVICE__
inline
float expf(float x) { return __ocml_exp_f32(x); }
__DEVICE__
inline
float expm1f(float x) { return __ocml_expm1_f32(x); }
__DEVICE__
inline
float fabsf(float x) { return __ocml_fabs_f32(x); }
__DEVICE__
inline
float fdimf(float x, float y) { return __ocml_fdim_f32(x, y); }
__DEVICE__
inline
float fdividef(float x, float y) { return x / y; }
__DEVICE__
inline
float floorf(float x) { return __ocml_floor_f32(x); }
__DEVICE__
inline
float fmaf(float x, float y, float z) { return __ocml_fma_f32(x, y, z); }
__DEVICE__
inline
float fmaxf(float x, float y) { return __ocml_fmax_f32(x, y); }
__DEVICE__
inline
float fminf(float x, float y) { return __ocml_fmin_f32(x, y); }
__DEVICE__
inline
float fmodf(float x, float y) { return __ocml_fmod_f32(x, y); }
__DEVICE__
inline
float frexpf(float x, int* nptr)
{
    int tmp;
    float r =
        __ocml_frexp_f32(x, (__attribute__((address_space(5))) int*) &tmp);
    *nptr = tmp;

    return r;
}
__DEVICE__
inline
float hypotf(float x, float y) { return __ocml_hypot_f32(x, y); }
__DEVICE__
inline
int ilogbf(float x) { return __ocml_ilogb_f32(x); }
__DEVICE__
inline
__RETURN_TYPE isfinite(float x) { return __ocml_isfinite_f32(x); }
__DEVICE__
inline
__RETURN_TYPE isinf(float x) { return __ocml_isinf_f32(x); }
__DEVICE__
inline
__RETURN_TYPE isnan(float x) { return __ocml_isnan_f32(x); }
__DEVICE__
inline
float j0f(float x) { return __ocml_j0_f32(x); }
__DEVICE__
inline
float j1f(float x) { return __ocml_j1_f32(x); }
__DEVICE__
inline
float jnf(int n, float x)
{   // TODO: we could use Ahmes multiplication and the Miller & Brown algorithm
    //       for linear recurrences to get O(log n) steps, but it's unclear if
    //       it'd be beneficial in this case.
    if (n == 0) return j0f(x);
    if (n == 1) return j1f(x);

    float x0 = j0f(x);
    float x1 = j1f(x);
    for (int i = 1; i < n; ++i) {
        float x2 = (2 * i) / x * x1 - x0;
        x0 = x1;
        x1 = x2;
    }

    return x1;
}
__DEVICE__
inline
float ldexp(float x, int e) { return __ocml_ldexp_f32(x, e); }
__DEVICE__
inline
float ldexpf(float x, int e) { return __ocml_ldexp_f32(x, e); }
__DEVICE__
inline
float lgammaf(float x) { return __ocml_lgamma_f32(x); }
__DEVICE__
inline
long long int llrintf(float x) { return __ocml_rint_f32(x); }
__DEVICE__
inline
long long int llroundf(float x) { return __ocml_round_f32(x); }
__DEVICE__
inline
float log10f(float x) { return __ocml_log10_f32(x); }
__DEVICE__
inline
float log1pf(float x) { return __ocml_log1p_f32(x); }
__DEVICE__
inline
float log2f(float x) { return __ocml_log2_f32(x); }
__DEVICE__
inline
float logbf(float x) { return __ocml_logb_f32(x); }
__DEVICE__
inline
float logf(float x) { return __ocml_log_f32(x); }
__DEVICE__
inline
long int lrintf(float x) { return __ocml_rint_f32(x); }
__DEVICE__
inline
long int lroundf(float x) { return __ocml_round_f32(x); }
__DEVICE__
inline
float modff(float x, float* iptr)
{
    float tmp;
    float r =
        __ocml_modf_f32(x, (__attribute__((address_space(5))) float*) &tmp);
    *iptr = tmp;

    return r;
}
__DEVICE__
inline
float nanf(const char* tagp)
{
    union {
        float val;
        struct ieee_float {
            uint32_t mantissa : 22;
            uint32_t quiet : 1;
            uint32_t exponent : 8;
            uint32_t sign : 1;
        } bits;

        static_assert(sizeof(float) == sizeof(ieee_float), "");
    } tmp;

    tmp.bits.sign = 0u;
    tmp.bits.exponent = ~0u;
    tmp.bits.quiet = 1u;
    tmp.bits.mantissa = __make_mantissa(tagp);

    return tmp.val;
}
__DEVICE__
inline
float nearbyintf(float x) { return __ocml_nearbyint_f32(x); }
__DEVICE__
inline
float nextafterf(float x, float y) { return __ocml_nextafter_f32(x, y); }
__DEVICE__
inline
float norm3df(float x, float y, float z) { return __ocml_len3_f32(x, y, z); }
__DEVICE__
inline
float norm4df(float x, float y, float z, float w)
{
    return __ocml_len4_f32(x, y, z, w);
}
__DEVICE__
inline
float normcdff(float x) { return __ocml_ncdf_f32(x); }
__DEVICE__
inline
float normcdfinvf(float x) { return __ocml_ncdfinv_f32(x); }
__DEVICE__
inline
float normf(int dim, const float* a)
{   // TODO: placeholder until OCML adds support.
    float r = 0;
    while (dim--) { r += a[0] * a[0]; ++a; }

    return __ocml_sqrt_f32(r);
}
__DEVICE__
inline
float powf(float x, float y) { return __ocml_pow_f32(x, y); }
__DEVICE__
inline
float rcbrtf(float x) { return __ocml_rcbrt_f32(x); }
__DEVICE__
inline
float remainderf(float x, float y) { return __ocml_remainder_f32(x, y); }
__DEVICE__
inline
float remquof(float x, float y, int* quo)
{
    int tmp;
    float r =
        __ocml_remquo_f32(x, y, (__attribute__((address_space(5))) int*) &tmp);
    *quo = tmp;

    return r;
}
__DEVICE__
inline
float rhypotf(float x, float y) { return __ocml_rhypot_f32(x, y); }
__DEVICE__
inline
float rintf(float x) { return __ocml_rint_f32(x); }
__DEVICE__
inline
float rnorm3df(float x, float y, float z)
{
    return __ocml_rlen3_f32(x, y, z);
}

__DEVICE__
inline
float rnorm4df(float x, float y, float z, float w)
{
    return __ocml_rlen4_f32(x, y, z, w);
}
__DEVICE__
inline
float rnormf(int dim, const float* a)
{   // TODO: placeholder until OCML adds support.
    float r = 0;
    while (dim--) { r += a[0] * a[0]; ++a; }

    return __ocml_rsqrt_f32(r);
}
__DEVICE__
inline
float roundf(float x) { return __ocml_round_f32(x); }
__DEVICE__
inline
float rsqrtf(float x) { return __ocml_rsqrt_f32(x); }
__DEVICE__
inline
float scalblnf(float x, long int n)
{
    return (n < INT_MAX) ? __ocml_scalbn_f32(x, n) : __ocml_scalb_f32(x, n);
}
__DEVICE__
inline
float scalbnf(float x, int n) { return __ocml_scalbn_f32(x, n); }
__DEVICE__
inline
__RETURN_TYPE signbit(float x) { return __ocml_signbit_f32(x); }
__DEVICE__
inline
void sincosf(float x, float* sptr, float* cptr)
{
    float tmp;

    *sptr =
        __ocml_sincos_f32(x, (__attribute__((address_space(5))) float*) &tmp);
    *cptr = tmp;
}
__DEVICE__
inline
void sincospif(float x, float* sptr, float* cptr)
{
    float tmp;

    *sptr =
        __ocml_sincospi_f32(x, (__attribute__((address_space(5))) float*) &tmp);
    *cptr = tmp;
}
__DEVICE__
inline
float sinf(float x) { return __ocml_sin_f32(x); }
__DEVICE__
inline
float sinhf(float x) { return __ocml_sinh_f32(x); }
__DEVICE__
inline
float sinpif(float x) { return __ocml_sinpi_f32(x); }
__DEVICE__
inline
float sqrtf(float x) { return __ocml_sqrt_f32(x); }
__DEVICE__
inline
float tanf(float x) { return __ocml_tan_f32(x); }
__DEVICE__
inline
float tanhf(float x) { return __ocml_tanh_f32(x); }
__DEVICE__
inline
float tgammaf(float x) { return __ocml_tgamma_f32(x); }
__DEVICE__
inline
float truncf(float x) { return __ocml_trunc_f32(x); }
__DEVICE__
inline
float y0f(float x) { return __ocml_y0_f32(x); }
__DEVICE__
inline
float y1f(float x) { return __ocml_y1_f32(x); }
__DEVICE__
inline
float ynf(int n, float x)
{   // TODO: we could use Ahmes multiplication and the Miller & Brown algorithm
    //       for linear recurrences to get O(log n) steps, but it's unclear if
    //       it'd be beneficial in this case. Placeholder until OCML adds
    //       support.
    if (n == 0) return y0f(x);
    if (n == 1) return y1f(x);

    float x0 = y0f(x);
    float x1 = y1f(x);
    for (int i = 1; i < n; ++i) {
        float x2 = (2 * i) / x * x1 - x0;
        x0 = x1;
        x1 = x2;
    }

    return x1;
}

// BEGIN INTRINSICS
__DEVICE__
inline
float __cosf(float x) { return __ocml_native_cos_f32(x); }
__DEVICE__
inline
float __exp10f(float x) { return __ocml_native_exp10_f32(x); }
__DEVICE__
inline
float __expf(float x) { return __ocml_native_exp_f32(x); }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fadd_rd(float x, float y) { return __ocml_add_rtn_f32(x, y); }
#endif
__DEVICE__
inline
float __fadd_rn(float x, float y) { return x + y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fadd_ru(float x, float y) { return __ocml_add_rtp_f32(x, y); }
__DEVICE__
inline
float __fadd_rz(float x, float y) { return __ocml_add_rtz_f32(x, y); }
__DEVICE__
inline
float __fdiv_rd(float x, float y) { return __ocml_div_rtn_f32(x, y); }
#endif
__DEVICE__
inline
float __fdiv_rn(float x, float y) { return x / y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fdiv_ru(float x, float y) { return __ocml_div_rtp_f32(x, y); }
__DEVICE__
inline
float __fdiv_rz(float x, float y) { return __ocml_div_rtz_f32(x, y); }
#endif
__DEVICE__
inline
float __fdividef(float x, float y) { return x / y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fmaf_rd(float x, float y, float z)
{
    return __ocml_fma_rtn_f32(x, y, z);
}
#endif
__DEVICE__
inline
float __fmaf_rn(float x, float y, float z)
{
    return __ocml_fma_f32(x, y, z);
}
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fmaf_ru(float x, float y, float z)
{
    return __ocml_fma_rtp_f32(x, y, z);
}
__DEVICE__
inline
float __fmaf_rz(float x, float y, float z)
{
   return __ocml_fma_rtz_f32(x, y, z);
}
__DEVICE__
inline
float __fmul_rd(float x, float y) { return __ocml_mul_rtn_f32(x, y); }
#endif
__DEVICE__
inline
float __fmul_rn(float x, float y) { return x * y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fmul_ru(float x, float y)  { return __ocml_mul_rtp_f32(x, y); }
__DEVICE__
inline
float __fmul_rz(float x, float y) { return __ocml_mul_rtz_f32(x, y); }
__DEVICE__
inline
float __frcp_rd(float x) { return __llvm_amdgcn_rcp_f32(x); }
#endif
__DEVICE__
inline
float __frcp_rn(float x) { return __llvm_amdgcn_rcp_f32(x); }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __frcp_ru(float x) { return __llvm_amdgcn_rcp_f32(x); }
__DEVICE__
inline
float __frcp_rz(float x) { return __llvm_amdgcn_rcp_f32(x); }
#endif
__DEVICE__
inline
float __frsqrt_rn(float x) { return __llvm_amdgcn_rsq_f32(x); }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fsqrt_rd(float x) { return __ocml_sqrt_rtn_f32(x); }
#endif
__DEVICE__
inline
float __fsqrt_rn(float x) { return __ocml_native_sqrt_f32(x); }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fsqrt_ru(float x) { return __ocml_sqrt_rtp_f32(x); }
__DEVICE__
inline
float __fsqrt_rz(float x) { return __ocml_sqrt_rtz_f32(x); }
__DEVICE__
inline
float __fsub_rd(float x, float y) { return __ocml_sub_rtn_f32(x, y); }
#endif
__DEVICE__
inline
float __fsub_rn(float x, float y) { return x - y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
float __fsub_ru(float x, float y) { return __ocml_sub_rtp_f32(x, y); }
__DEVICE__
inline
float __fsub_rz(float x, float y) { return __ocml_sub_rtz_f32(x, y); }
#endif
__DEVICE__
inline
float __log10f(float x) { return __ocml_native_log10_f32(x); }
__DEVICE__
inline
float __log2f(float x) { return __ocml_native_log2_f32(x); }
__DEVICE__
inline
float __logf(float x) { return __ocml_native_log_f32(x); }
__DEVICE__
inline
float __powf(float x, float y) { return __ocml_pow_f32(x, y); }
__DEVICE__
inline
float __saturatef(float x) { return (x < 0) ? 0 : ((x > 1) ? 1 : x); }
__DEVICE__
inline
void __sincosf(float x, float* sptr, float* cptr)
{
    *sptr = __ocml_native_sin_f32(x);
    *cptr = __ocml_native_cos_f32(x);
}
__DEVICE__
inline
float __sinf(float x) { return __ocml_native_sin_f32(x); }
__DEVICE__
inline
float __tanf(float x) { return __ocml_tan_f32(x); }
// END INTRINSICS
// END FLOAT

// BEGIN DOUBLE
__DEVICE__
inline
double abs(double x) { return __ocml_fabs_f64(x); }
__DEVICE__
inline
double acos(double x) { return __ocml_acos_f64(x); }
__DEVICE__
inline
double acosh(double x) { return __ocml_acosh_f64(x); }
__DEVICE__
inline
double asin(double x) { return __ocml_asin_f64(x); }
__DEVICE__
inline
double asinh(double x) { return __ocml_asinh_f64(x); }
__DEVICE__
inline
double atan(double x) { return __ocml_atan_f64(x); }
__DEVICE__
inline
double atan2(double x, double y) { return __ocml_atan2_f64(x, y); }
__DEVICE__
inline
double atanh(double x) { return __ocml_atanh_f64(x); }
__DEVICE__
inline
double cbrt(double x) { return __ocml_cbrt_f64(x); }
__DEVICE__
inline
double ceil(double x) { return __ocml_ceil_f64(x); }
__DEVICE__
inline
double copysign(double x, double y) { return __ocml_copysign_f64(x, y); }
__DEVICE__
inline
double cos(double x)  { return __ocml_cos_f64(x); }
__DEVICE__
inline
double cosh(double x) { return __ocml_cosh_f64(x); }
__DEVICE__
inline
double cospi(double x) { return __ocml_cospi_f64(x); }
__DEVICE__
inline
double cyl_bessel_i0(double x) { return __ocml_i0_f64(x); }
__DEVICE__
inline
double cyl_bessel_i1(double x) { return __ocml_i1_f64(x); }
__DEVICE__
inline
double erf(double x) { return __ocml_erf_f64(x); }
__DEVICE__
inline
double erfc(double x) { return __ocml_erfc_f64(x); }
__DEVICE__
inline
double erfcinv(double x) { return __ocml_erfcinv_f64(x); }
__DEVICE__
inline
double erfcx(double x) { return __ocml_erfcx_f64(x); }
__DEVICE__
inline
double erfinv(double x) { return __ocml_erfinv_f64(x); }
__DEVICE__
inline
double exp(double x) { return __ocml_exp_f64(x); }
__DEVICE__
inline
double exp10(double x) { return __ocml_exp10_f64(x); }
__DEVICE__
inline
double exp2(double x) { return __ocml_exp2_f64(x); }
__DEVICE__
inline
double expm1(double x) { return __ocml_expm1_f64(x); }
__DEVICE__
inline
double fabs(double x) { return __ocml_fabs_f64(x); }
__DEVICE__
inline
double fdim(double x, double y) { return __ocml_fdim_f64(x, y); }
__DEVICE__
inline
double floor(double x) { return __ocml_floor_f64(x); }
__DEVICE__
inline
double fma(double x, double y, double z) { return __ocml_fma_f64(x, y, z); }
__DEVICE__
inline
double fmax(double x, double y) { return __ocml_fmax_f64(x, y); }
__DEVICE__
inline
double fmin(double x, double y) { return __ocml_fmin_f64(x, y); }
__DEVICE__
inline
double fmod(double x, double y) { return __ocml_fmod_f64(x, y); }
__DEVICE__
inline
double frexp(double x, int* nptr)
{
    int tmp;
    double r =
        __ocml_frexp_f64(x, (__attribute__((address_space(5))) int*) &tmp);
    *nptr = tmp;

    return r;
}
__DEVICE__
inline
double hypot(double x, double y) { return __ocml_hypot_f64(x, y); }
__DEVICE__
inline
int ilogb(double x) { return __ocml_ilogb_f64(x); }
__DEVICE__
inline
__RETURN_TYPE isfinite(double x) { return __ocml_isfinite_f64(x); }
__DEVICE__
inline
__RETURN_TYPE isinf(double x) { return __ocml_isinf_f64(x); }
__DEVICE__
inline
__RETURN_TYPE isnan(double x) { return __ocml_isnan_f64(x); }
__DEVICE__
inline
double j0(double x) { return __ocml_j0_f64(x); }
__DEVICE__
inline
double j1(double x) { return __ocml_j1_f64(x); }
__DEVICE__
inline
double jn(int n, double x)
{   // TODO: we could use Ahmes multiplication and the Miller & Brown algorithm
    //       for linear recurrences to get O(log n) steps, but it's unclear if
    //       it'd be beneficial in this case. Placeholder until OCML adds
    //       support.
    if (n == 0) return j0f(x);
    if (n == 1) return j1f(x);

    double x0 = j0f(x);
    double x1 = j1f(x);
    for (int i = 1; i < n; ++i) {
        double x2 = (2 * i) / x * x1 - x0;
        x0 = x1;
        x1 = x2;
    }

    return x1;
}
__DEVICE__
inline
double ldexp(double x, int e) { return __ocml_ldexp_f64(x, e); }
__DEVICE__
inline
double lgamma(double x) { return __ocml_lgamma_f64(x); }
__DEVICE__
inline
long long int llrint(double x) { return __ocml_rint_f64(x); }
__DEVICE__
inline
long long int llround(double x) { return __ocml_round_f64(x); }
__DEVICE__
inline
double log(double x) { return __ocml_log_f64(x); }
__DEVICE__
inline
double log10(double x) { return __ocml_log10_f64(x); }
__DEVICE__
inline
double log1p(double x) { return __ocml_log1p_f64(x); }
__DEVICE__
inline
double log2(double x) { return __ocml_log2_f64(x); }
__DEVICE__
inline
double logb(double x) { return __ocml_logb_f64(x); }
__DEVICE__
inline
long int lrint(double x) { return __ocml_rint_f64(x); }
__DEVICE__
inline
long int lround(double x) { return __ocml_round_f64(x); }
__DEVICE__
inline
double modf(double x, double* iptr)
{
    double tmp;
    double r =
        __ocml_modf_f64(x, (__attribute__((address_space(5))) double*) &tmp);
    *iptr = tmp;

    return r;
}
__DEVICE__
inline
double nan(const char* tagp)
{
#if !_WIN32
    union {
        double val;
        struct ieee_double {
            uint64_t mantissa : 51;
            uint32_t quiet : 1;
            uint32_t exponent : 11;
            uint32_t sign : 1;
        }  bits;
        static_assert(sizeof(double) == sizeof(ieee_double), "");
    } tmp;

    tmp.bits.sign = 0u;
    tmp.bits.exponent = ~0u;
    tmp.bits.quiet = 1u;
    tmp.bits.mantissa = __make_mantissa(tagp);

    return tmp.val;
#else
    static_assert(sizeof(uint64_t)==sizeof(double));
    uint64_t val = __make_mantissa(tagp);
    val |= 0xFFF << 51;
    return *reinterpret_cast<double*>(&val);
#endif
}
__DEVICE__
inline
double nearbyint(double x) { return __ocml_nearbyint_f64(x); }
__DEVICE__
inline
double nextafter(double x, double y) { return __ocml_nextafter_f64(x, y); }
__DEVICE__
inline
double norm(int dim, const double* a)
{   // TODO: placeholder until OCML adds support.
    double r = 0;
    while (dim--) { r += a[0] * a[0]; ++a; }

    return __ocml_sqrt_f64(r);
}
__DEVICE__
inline
double norm3d(double x, double y, double z)
{
    return __ocml_len3_f64(x, y, z);
}
__DEVICE__
inline
double norm4d(double x, double y, double z, double w)
{
    return __ocml_len4_f64(x, y, z, w);
}
__DEVICE__
inline
double normcdf(double x) { return __ocml_ncdf_f64(x); }
__DEVICE__
inline
double normcdfinv(double x) { return __ocml_ncdfinv_f64(x); }
__DEVICE__
inline
double pow(double x, double y) { return __ocml_pow_f64(x, y); }
__DEVICE__
inline
double rcbrt(double x) { return __ocml_rcbrt_f64(x); }
__DEVICE__
inline
double remainder(double x, double y) { return __ocml_remainder_f64(x, y); }
__DEVICE__
inline
double remquo(double x, double y, int* quo)
{
    int tmp;
    double r =
        __ocml_remquo_f64(x, y, (__attribute__((address_space(5))) int*) &tmp);
    *quo = tmp;

    return r;
}
__DEVICE__
inline
double rhypot(double x, double y) { return __ocml_rhypot_f64(x, y); }
__DEVICE__
inline
double rint(double x) { return __ocml_rint_f64(x); }
__DEVICE__
inline
double rnorm(int dim, const double* a)
{   // TODO: placeholder until OCML adds support.
    double r = 0;
    while (dim--) { r += a[0] * a[0]; ++a; }

    return __ocml_rsqrt_f64(r);
}
__DEVICE__
inline
double rnorm3d(double x, double y, double z)
{
    return __ocml_rlen3_f64(x, y, z);
}
__DEVICE__
inline
double rnorm4d(double x, double y, double z, double w)
{
    return __ocml_rlen4_f64(x, y, z, w);
}
__DEVICE__
inline
double round(double x) { return __ocml_round_f64(x); }
__DEVICE__
inline
double rsqrt(double x) { return __ocml_rsqrt_f64(x); }
__DEVICE__
inline
double scalbln(double x, long int n)
{
    return (n < INT_MAX) ? __ocml_scalbn_f64(x, n) : __ocml_scalb_f64(x, n);
}
__DEVICE__
inline
double scalbn(double x, int n) { return __ocml_scalbn_f64(x, n); }
__DEVICE__
inline
__RETURN_TYPE signbit(double x) { return __ocml_signbit_f64(x); }
__DEVICE__
inline
double sin(double x) { return __ocml_sin_f64(x); }
__DEVICE__
inline
void sincos(double x, double* sptr, double* cptr)
{
    double tmp;
    *sptr =
        __ocml_sincos_f64(x, (__attribute__((address_space(5))) double*) &tmp);
    *cptr = tmp;
}
__DEVICE__
inline
void sincospi(double x, double* sptr, double* cptr)
{
    double tmp;
    *sptr = __ocml_sincospi_f64(
        x, (__attribute__((address_space(5))) double*) &tmp);
    *cptr = tmp;
}
__DEVICE__
inline
double sinh(double x) { return __ocml_sinh_f64(x); }
__DEVICE__
inline
double sinpi(double x) { return __ocml_sinpi_f64(x); }
__DEVICE__
inline
double sqrt(double x) { return __ocml_sqrt_f64(x); }
__DEVICE__
inline
double tan(double x) { return __ocml_tan_f64(x); }
__DEVICE__
inline
double tanh(double x) { return __ocml_tanh_f64(x); }
__DEVICE__
inline
double tgamma(double x) { return __ocml_tgamma_f64(x); }
__DEVICE__
inline
double trunc(double x) { return __ocml_trunc_f64(x); }
__DEVICE__
inline
double y0(double x) { return __ocml_y0_f64(x); }
__DEVICE__
inline
double y1(double x) { return __ocml_y1_f64(x); }
__DEVICE__
inline
double yn(int n, double x)
{   // TODO: we could use Ahmes multiplication and the Miller & Brown algorithm
    //       for linear recurrences to get O(log n) steps, but it's unclear if
    //       it'd be beneficial in this case. Placeholder until OCML adds
    //       support.
    if (n == 0) return j0f(x);
    if (n == 1) return j1f(x);

    double x0 = j0f(x);
    double x1 = j1f(x);
    for (int i = 1; i < n; ++i) {
        double x2 = (2 * i) / x * x1 - x0;
        x0 = x1;
        x1 = x2;
    }

    return x1;
}

// BEGIN INTRINSICS
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __dadd_rd(double x, double y) { return __ocml_add_rtn_f64(x, y); }
#endif
__DEVICE__
inline
double __dadd_rn(double x, double y) { return x + y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __dadd_ru(double x, double y) { return __ocml_add_rtp_f64(x, y); }
__DEVICE__
inline
double __dadd_rz(double x, double y) { return __ocml_add_rtz_f64(x, y); }
__DEVICE__
inline
double __ddiv_rd(double x, double y) { return __ocml_div_rtn_f64(x, y); }
#endif
__DEVICE__
inline
double __ddiv_rn(double x, double y) { return x / y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __ddiv_ru(double x, double y) { return __ocml_div_rtp_f64(x, y); }
__DEVICE__
inline
double __ddiv_rz(double x, double y) { return __ocml_div_rtz_f64(x, y); }
__DEVICE__
inline
double __dmul_rd(double x, double y) { return __ocml_mul_rtn_f64(x, y); }
#endif
__DEVICE__
inline
double __dmul_rn(double x, double y) { return x * y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __dmul_ru(double x, double y) { return __ocml_mul_rtp_f64(x, y); }
__DEVICE__
inline
double __dmul_rz(double x, double y) { return __ocml_mul_rtz_f64(x, y); }
__DEVICE__
inline
double __drcp_rd(double x) { return __llvm_amdgcn_rcp_f64(x); }
#endif
__DEVICE__
inline
double __drcp_rn(double x) { return __llvm_amdgcn_rcp_f64(x); }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __drcp_ru(double x) { return __llvm_amdgcn_rcp_f64(x); }
__DEVICE__
inline
double __drcp_rz(double x) { return __llvm_amdgcn_rcp_f64(x); }
__DEVICE__
inline
double __dsqrt_rd(double x) { return __ocml_sqrt_rtn_f64(x); }
#endif
__DEVICE__
inline
double __dsqrt_rn(double x) { return __ocml_sqrt_f64(x); }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __dsqrt_ru(double x) { return __ocml_sqrt_rtp_f64(x); }
__DEVICE__
inline
double __dsqrt_rz(double x) { return __ocml_sqrt_rtz_f64(x); }
__DEVICE__
inline
double __dsub_rd(double x, double y) { return __ocml_sub_rtn_f64(x, y); }
#endif
__DEVICE__
inline
double __dsub_rn(double x, double y) { return x - y; }
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __dsub_ru(double x, double y) { return __ocml_sub_rtp_f64(x, y); }
__DEVICE__
inline
double __dsub_rz(double x, double y) { return __ocml_sub_rtz_f64(x, y); }
__DEVICE__
inline
double __fma_rd(double x, double y, double z)
{
    return __ocml_fma_rtn_f64(x, y, z);
}
#endif
__DEVICE__
inline
double __fma_rn(double x, double y, double z)
{
    return __ocml_fma_f64(x, y, z);
}
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__
inline
double __fma_ru(double x, double y, double z)
{
    return __ocml_fma_rtp_f64(x, y, z);
}
__DEVICE__
inline
double __fma_rz(double x, double y, double z)
{
    return __ocml_fma_rtz_f64(x, y, z);
}
#endif
// END INTRINSICS
// END DOUBLE

// BEGIN INTEGER
__DEVICE__
inline
int abs(int x)
{
    int sgn = x >> (sizeof(int) * CHAR_BIT - 1);
    return (x ^ sgn) - sgn;
}
__DEVICE__
inline
long labs(long x)
{
    long sgn = x >> (sizeof(long) * CHAR_BIT - 1);
    return (x ^ sgn) - sgn;
}
__DEVICE__
inline
long long llabs(long long x)
{
    long long sgn = x >> (sizeof(long long) * CHAR_BIT - 1);
    return (x ^ sgn) - sgn;
}

#if defined(__cplusplus)
    __DEVICE__
    inline
    long abs(long x) { return labs(x); }
    __DEVICE__
    inline
    long long abs(long long x) { return llabs(x); }
#endif
// END INTEGER

__DEVICE__
inline _Float16 fma(_Float16 x, _Float16 y, _Float16 z) {
    return __ocml_fma_f16(x, y, z);
}

__DEVICE__
inline float fma(float x, float y, float z) {
    return fmaf(x, y, z);
}

#pragma push_macro("__DEF_FLOAT_FUN")
#pragma push_macro("__DEF_FLOAT_FUN2")
#pragma push_macro("__DEF_FLOAT_FUN2I")
#pragma push_macro("__HIP_OVERLOAD")
#pragma push_macro("__HIP_OVERLOAD2")

// __hip_enable_if::type is a type function which returns __T if __B is true.
template<bool __B, class __T = void>
struct __hip_enable_if {};

template <class __T> struct __hip_enable_if<true, __T> {
  typedef __T type;
};

// __HIP_OVERLOAD1 is used to resolve function calls with integer argument to
// avoid compilation error due to ambibuity. e.g. floor(5) is resolved with
// floor(double).
#define __HIP_OVERLOAD1(__retty, __fn)                                         \
  template <typename __T>                                                      \
  __DEVICE__                                                                   \
      typename __hip_enable_if<std::numeric_limits<__T>::is_integer,           \
                                      __retty>::type                           \
      __fn(__T __x) {                                                          \
    return ::__fn((double)__x);                                                \
  }

// __HIP_OVERLOAD2 is used to resolve function calls with mixed float/double
// or integer argument to avoid compilation error due to ambibuity. e.g.
// max(5.0f, 6.0) is resolved with max(double, double).
#define __HIP_OVERLOAD2(__retty, __fn)                                         \
  template <typename __T1, typename __T2>                                      \
  __DEVICE__ typename __hip_enable_if<                                         \
      std::numeric_limits<__T1>::is_specialized &&                             \
          std::numeric_limits<__T2>::is_specialized,                           \
      __retty>::type                                                           \
  __fn(__T1 __x, __T2 __y) {                                                   \
    return __fn((double)__x, (double)__y);                                     \
  }

// Define cmath functions with float argument and returns float.
#define __DEF_FUN1(retty, func) \
__DEVICE__ \
inline \
float func(float x) \
{ \
  return func##f(x); \
} \
__HIP_OVERLOAD1(retty, func)

// Define cmath functions with float argument and returns retty.
#define __DEF_FUNI(retty, func) \
__DEVICE__ \
inline \
retty func(float x) \
{ \
  return func##f(x); \
} \
__HIP_OVERLOAD1(retty, func)

// define cmath functions with two float arguments.
#define __DEF_FUN2(retty, func) \
__DEVICE__ \
inline \
float func(float x, float y) \
{ \
  return func##f(x, y); \
} \
__HIP_OVERLOAD2(retty, func)

__DEF_FUN1(double, acos)
__DEF_FUN1(double, acosh)
__DEF_FUN1(double, asin)
__DEF_FUN1(double, asinh)
__DEF_FUN1(double, atan)
__DEF_FUN2(double, atan2);
__DEF_FUN1(double, atanh)
__DEF_FUN1(double, cbrt)
__DEF_FUN1(double, ceil)
__DEF_FUN2(double, copysign);
__DEF_FUN1(double, cos)
__DEF_FUN1(double, cosh)
__DEF_FUN1(double, erf)
__DEF_FUN1(double, erfc)
__DEF_FUN1(double, exp)
__DEF_FUN1(double, exp2)
__DEF_FUN1(double, expm1)
__DEF_FUN1(double, fabs)
__DEF_FUN2(double, fdim);
__DEF_FUN1(double, floor)
__DEF_FUN2(double, fmax);
__DEF_FUN2(double, fmin);
__DEF_FUN2(double, fmod);
//__HIP_OVERLOAD1(int, fpclassify)
__DEF_FUN2(double, hypot);
__DEF_FUNI(int, ilogb)
__HIP_OVERLOAD1(bool, isfinite)
__HIP_OVERLOAD2(bool, isgreater);
__HIP_OVERLOAD2(bool, isgreaterequal);
__HIP_OVERLOAD1(bool, isinf);
__HIP_OVERLOAD2(bool, isless);
__HIP_OVERLOAD2(bool, islessequal);
__HIP_OVERLOAD2(bool, islessgreater);
__HIP_OVERLOAD1(bool, isnan);
//__HIP_OVERLOAD1(bool, isnormal)
__HIP_OVERLOAD2(bool, isunordered);
__DEF_FUN1(double, lgamma)
__DEF_FUN1(double, log)
__DEF_FUN1(double, log10)
__DEF_FUN1(double, log1p)
__DEF_FUN1(double, log2)
__DEF_FUN1(double, logb)
__DEF_FUNI(long long, llrint)
__DEF_FUNI(long long, llround)
__DEF_FUNI(long, lrint)
__DEF_FUNI(long, lround)
__DEF_FUN1(double, nearbyint);
__DEF_FUN2(double, nextafter);
__DEF_FUN2(double, pow);
__DEF_FUN2(double, remainder);
__DEF_FUN1(double, rint);
__DEF_FUN1(double, round);
__HIP_OVERLOAD1(bool, signbit)
__DEF_FUN1(double, sin)
__DEF_FUN1(double, sinh)
__DEF_FUN1(double, sqrt)
__DEF_FUN1(double, tan)
__DEF_FUN1(double, tanh)
__DEF_FUN1(double, tgamma)
__DEF_FUN1(double, trunc);

// define cmath functions with a float and an integer argument.
#define __DEF_FLOAT_FUN2I(func) \
__DEVICE__ \
inline \
float func(float x, int y) \
{ \
  return func##f(x, y); \
}
__DEF_FLOAT_FUN2I(scalbn)

template<class T>
__DEVICE__ inline static T min(T arg1, T arg2) {
  return (arg1 < arg2) ? arg1 : arg2;
}

template<class T>
__DEVICE__ inline static T max(T arg1, T arg2) {
  return (arg1 > arg2) ? arg1 : arg2;
}

#if __HCC__

__DEVICE__ inline static uint32_t min(uint32_t arg1, int32_t arg2) {
  return min(arg1, (uint32_t) arg2);
}
/*__DEVICE__ inline static uint32_t min(int32_t arg1, uint32_t arg2) {
  return min((uint32_t) arg1, arg2);
}

__DEVICE__ inline static uint64_t min(uint64_t arg1, int64_t arg2) {
  return min(arg1, (uint64_t) arg2);
}
__DEVICE__ inline static uint64_t min(int64_t arg1, uint64_t arg2) {
  return min((uint64_t) arg1, arg2);
}

__DEVICE__ inline static unsigned long long min(unsigned long long arg1, long long arg2) {
  return min(arg1, (unsigned long long) arg2);
}
__DEVICE__ inline static unsigned long long min(long long arg1, unsigned long long arg2) {
  return min((unsigned long long) arg1, arg2);
}*/

__DEVICE__ inline static uint32_t max(uint32_t arg1, int32_t arg2) {
  return max(arg1, (uint32_t) arg2);
}
__DEVICE__ inline static uint32_t max(int32_t arg1, uint32_t arg2) {
  return max((uint32_t) arg1, arg2);
}

/*__DEVICE__ inline static uint64_t max(uint64_t arg1, int64_t arg2) {
  return max(arg1, (uint64_t) arg2);
}
__DEVICE__ inline static uint64_t max(int64_t arg1, uint64_t arg2) {
  return max((uint64_t) arg1, arg2);
}

__DEVICE__ inline static unsigned long long max(unsigned long long arg1, long long arg2) {
  return max(arg1, (unsigned long long) arg2);
}
__DEVICE__ inline static unsigned long long max(long long arg1, unsigned long long arg2) {
  return max((unsigned long long) arg1, arg2);
}*/
#else
__DEVICE__ inline int min(int arg1, int arg2) {
  return (arg1 < arg2) ? arg1 : arg2;
}
__DEVICE__ inline int max(int arg1, int arg2) {
  return (arg1 > arg2) ? arg1 : arg2;
}

__DEVICE__
inline
float max(float x, float y) {
  return fmaxf(x, y);
}

__DEVICE__
inline
double max(double x, double y) {
  return fmax(x, y);
}

__DEVICE__
inline
float min(float x, float y) {
  return fminf(x, y);
}

__DEVICE__
inline
double min(double x, double y) {
  return fmin(x, y);
}

__HIP_OVERLOAD2(double, max)
__HIP_OVERLOAD2(double, min)

#endif

__host__ inline static int min(int arg1, int arg2) {
  return std::min(arg1, arg2);
}

__host__ inline static int max(int arg1, int arg2) {
  return std::max(arg1, arg2);
}


#pragma pop_macro("__DEF_FLOAT_FUN")
#pragma pop_macro("__DEF_FLOAT_FUN2")
#pragma pop_macro("__DEF_FLOAT_FUN2I")
#pragma pop_macro("__HIP_OVERLOAD")
#pragma pop_macro("__HIP_OVERLOAD2")
#pragma pop_macro("__DEVICE__")
#pragma pop_macro("__RETURN_TYPE")

// For backward compatibility.
// There are HIP applications e.g. TensorFlow, expecting __HIP_ARCH_* macros
// defined after including math_functions.h.
#include <hip/hcc_detail/hip_runtime.h>
