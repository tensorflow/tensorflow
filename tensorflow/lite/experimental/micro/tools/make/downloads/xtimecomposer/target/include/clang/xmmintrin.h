/*===---- xmmintrin.h - SSE intrinsics -------------------------------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */
 
#ifndef __XMMINTRIN_H
#define __XMMINTRIN_H
 
#ifndef __SSE__
#error "SSE instruction set not enabled"
#else

#include <mmintrin.h>

typedef int __v4si __attribute__((__vector_size__(16)));
typedef float __v4sf __attribute__((__vector_size__(16)));
typedef float __m128 __attribute__((__vector_size__(16)));

/* This header should only be included in a hosted environment as it depends on
 * a standard library to provide allocation routines. */
#if __STDC_HOSTED__
#include <mm_malloc.h>
#endif

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_add_ss(__m128 __a, __m128 __b)
{
  __a[0] += __b[0];
  return __a;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_add_ps(__m128 __a, __m128 __b)
{
  return __a + __b;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sub_ss(__m128 __a, __m128 __b)
{
  __a[0] -= __b[0];
  return __a;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sub_ps(__m128 __a, __m128 __b)
{
  return __a - __b;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_mul_ss(__m128 __a, __m128 __b)
{
  __a[0] *= __b[0];
  return __a;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_mul_ps(__m128 __a, __m128 __b)
{
  return __a * __b;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_div_ss(__m128 __a, __m128 __b)
{
  __a[0] /= __b[0];
  return __a;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_div_ps(__m128 __a, __m128 __b)
{
  return __a / __b;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sqrt_ss(__m128 __a)
{
  __m128 __c = __builtin_ia32_sqrtss(__a);
  return (__m128) { __c[0], __a[1], __a[2], __a[3] };
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sqrt_ps(__m128 __a)
{
  return __builtin_ia32_sqrtps(__a);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rcp_ss(__m128 __a)
{
  __m128 __c = __builtin_ia32_rcpss(__a);
  return (__m128) { __c[0], __a[1], __a[2], __a[3] };
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rcp_ps(__m128 __a)
{
  return __builtin_ia32_rcpps(__a);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rsqrt_ss(__m128 __a)
{
  __m128 __c = __builtin_ia32_rsqrtss(__a);
  return (__m128) { __c[0], __a[1], __a[2], __a[3] };
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rsqrt_ps(__m128 __a)
{
  return __builtin_ia32_rsqrtps(__a);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_min_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_minss(__a, __b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_min_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_minps(__a, __b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_max_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_maxss(__a, __b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_max_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_maxps(__a, __b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_and_ps(__m128 __a, __m128 __b)
{
  return (__m128)((__v4si)__a & (__v4si)__b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_andnot_ps(__m128 __a, __m128 __b)
{
  return (__m128)(~(__v4si)__a & (__v4si)__b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_or_ps(__m128 __a, __m128 __b)
{
  return (__m128)((__v4si)__a | (__v4si)__b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_xor_ps(__m128 __a, __m128 __b)
{
  return (__m128)((__v4si)__a ^ (__v4si)__b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 0);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 0);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 1);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 1);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 2);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 2);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpss(__b, __a, 1),
                                         4, 1, 2, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__b, __a, 1);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpss(__b, __a, 2),
                                         4, 1, 2, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__b, __a, 2);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 4);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 4);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnlt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 5);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnlt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 5);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnle_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 6);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnle_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 6);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpngt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpss(__b, __a, 5),
                                         4, 1, 2, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpngt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__b, __a, 5);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnge_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpss(__b, __a, 6),
                                         4, 1, 2, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnge_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__b, __a, 6);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpord_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 7);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpord_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 7);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpunord_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpss(__a, __b, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpunord_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpps(__a, __b, 3);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comieq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comieq(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comilt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comilt(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comile_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comile(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comigt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comigt(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comige_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comige(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_comineq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comineq(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomieq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomieq(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomilt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomilt(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomile_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomile(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomigt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomigt(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomige_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomige(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_ucomineq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomineq(__a, __b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_cvtss_si32(__m128 __a)
{
  return __builtin_ia32_cvtss2si(__a);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_cvt_ss2si(__m128 __a)
{
  return _mm_cvtss_si32(__a);
}

#ifdef __x86_64__

static __inline__ long long __attribute__((__always_inline__, __nodebug__))
_mm_cvtss_si64(__m128 __a)
{
  return __builtin_ia32_cvtss2si64(__a);
}

#endif

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_pi32(__m128 __a)
{
  return (__m64)__builtin_ia32_cvtps2pi(__a);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvt_ps2pi(__m128 __a)
{
  return _mm_cvtps_pi32(__a);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_cvttss_si32(__m128 __a)
{
  return __a[0];
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_cvtt_ss2si(__m128 __a)
{
  return _mm_cvttss_si32(__a);
}

static __inline__ long long __attribute__((__always_inline__, __nodebug__))
_mm_cvttss_si64(__m128 __a)
{
  return __a[0];
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvttps_pi32(__m128 __a)
{
  return (__m64)__builtin_ia32_cvttps2pi(__a);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtt_ps2pi(__m128 __a)
{
  return _mm_cvttps_pi32(__a);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi32_ss(__m128 __a, int __b)
{
  __a[0] = __b;
  return __a;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvt_si2ss(__m128 __a, int __b)
{
  return _mm_cvtsi32_ss(__a, __b);
}

#ifdef __x86_64__

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi64_ss(__m128 __a, long long __b)
{
  __a[0] = __b;
  return __a;
}

#endif

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi32_ps(__m128 __a, __m64 __b)
{
  return __builtin_ia32_cvtpi2ps(__a, (__v2si)__b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvt_pi2ps(__m128 __a, __m64 __b)
{
  return _mm_cvtpi32_ps(__a, __b);
}

static __inline__ float __attribute__((__always_inline__, __nodebug__))
_mm_cvtss_f32(__m128 __a)
{
  return __a[0];
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadh_pi(__m128 __a, const __m64 *__p)
{
  typedef float __mm_loadh_pi_v2f32 __attribute__((__vector_size__(8)));
  struct __mm_loadh_pi_struct {
    __mm_loadh_pi_v2f32 __u;
  } __attribute__((__packed__, __may_alias__));
  __mm_loadh_pi_v2f32 __b = ((struct __mm_loadh_pi_struct*)__p)->__u;
  __m128 __bb = __builtin_shufflevector(__b, __b, 0, 1, 0, 1);
  return __builtin_shufflevector(__a, __bb, 0, 1, 4, 5);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadl_pi(__m128 __a, const __m64 *__p)
{
  typedef float __mm_loadl_pi_v2f32 __attribute__((__vector_size__(8)));
  struct __mm_loadl_pi_struct {
    __mm_loadl_pi_v2f32 __u;
  } __attribute__((__packed__, __may_alias__));
  __mm_loadl_pi_v2f32 __b = ((struct __mm_loadl_pi_struct*)__p)->__u;
  __m128 __bb = __builtin_shufflevector(__b, __b, 0, 1, 0, 1);
  return __builtin_shufflevector(__a, __bb, 4, 5, 2, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_load_ss(const float *__p)
{
  struct __mm_load_ss_struct {
    float __u;
  } __attribute__((__packed__, __may_alias__));
  float __u = ((struct __mm_load_ss_struct*)__p)->__u;
  return (__m128){ __u, 0, 0, 0 };
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_load1_ps(const float *__p)
{
  struct __mm_load1_ps_struct {
    float __u;
  } __attribute__((__packed__, __may_alias__));
  float __u = ((struct __mm_load1_ps_struct*)__p)->__u;
  return (__m128){ __u, __u, __u, __u };
}

#define        _mm_load_ps1(p) _mm_load1_ps(p)

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_load_ps(const float *__p)
{
  return *(__m128*)__p;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadu_ps(const float *__p)
{
  struct __loadu_ps {
    __m128 __v;
  } __attribute__((__packed__, __may_alias__));
  return ((struct __loadu_ps*)__p)->__v;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadr_ps(const float *__p)
{
  __m128 __a = _mm_load_ps(__p);
  return __builtin_shufflevector(__a, __a, 3, 2, 1, 0);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set_ss(float __w)
{
  return (__m128){ __w, 0, 0, 0 };
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set1_ps(float __w)
{
  return (__m128){ __w, __w, __w, __w };
}

/* Microsoft specific. */
static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set_ps1(float __w)
{
    return _mm_set1_ps(__w);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set_ps(float __z, float __y, float __x, float __w)
{
  return (__m128){ __w, __x, __y, __z };
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_setr_ps(float __z, float __y, float __x, float __w)
{
  return (__m128){ __z, __y, __x, __w };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_setzero_ps(void)
{
  return (__m128){ 0, 0, 0, 0 };
}

static __inline__ void __attribute__((__always_inline__))
_mm_storeh_pi(__m64 *__p, __m128 __a)
{
  __builtin_ia32_storehps((__v2si *)__p, __a);
}

static __inline__ void __attribute__((__always_inline__))
_mm_storel_pi(__m64 *__p, __m128 __a)
{
  __builtin_ia32_storelps((__v2si *)__p, __a);
}

static __inline__ void __attribute__((__always_inline__))
_mm_store_ss(float *__p, __m128 __a)
{
  struct __mm_store_ss_struct {
    float __u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_store_ss_struct*)__p)->__u = __a[0];
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storeu_ps(float *__p, __m128 __a)
{
  __builtin_ia32_storeups(__p, __a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_store1_ps(float *__p, __m128 __a)
{
  __a = __builtin_shufflevector(__a, __a, 0, 0, 0, 0);
  _mm_storeu_ps(__p, __a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_store_ps1(float *__p, __m128 __a)
{
    return _mm_store1_ps(__p, __a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_store_ps(float *__p, __m128 __a)
{
  *(__m128 *)__p = __a;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_storer_ps(float *__p, __m128 __a)
{
  __a = __builtin_shufflevector(__a, __a, 3, 2, 1, 0);
  _mm_store_ps(__p, __a);
}

#define _MM_HINT_T0 3
#define _MM_HINT_T1 2
#define _MM_HINT_T2 1
#define _MM_HINT_NTA 0

#ifndef _MSC_VER
/* FIXME: We have to #define this because "sel" must be a constant integer, and
   Sema doesn't do any form of constant propagation yet. */

#define _mm_prefetch(a, sel) (__builtin_prefetch((void *)(a), 0, (sel)))
#endif

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_stream_pi(__m64 *__p, __m64 __a)
{
  __builtin_ia32_movntq(__p, __a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_stream_ps(float *__p, __m128 __a)
{
  __builtin_ia32_movntps(__p, __a);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_sfence(void)
{
  __builtin_ia32_sfence();
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_extract_pi16(__m64 __a, int __n)
{
  __v4hi __b = (__v4hi)__a;
  return (unsigned short)__b[__n & 3];
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_insert_pi16(__m64 __a, int __d, int __n)
{
   __v4hi __b = (__v4hi)__a;
   __b[__n & 3] = __d;
   return (__m64)__b;
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_max_pi16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pmaxsw((__v4hi)__a, (__v4hi)__b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_max_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pmaxub((__v8qi)__a, (__v8qi)__b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_min_pi16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pminsw((__v4hi)__a, (__v4hi)__b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_min_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pminub((__v8qi)__a, (__v8qi)__b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_movemask_pi8(__m64 __a)
{
  return __builtin_ia32_pmovmskb((__v8qi)__a);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_mulhi_pu16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pmulhuw((__v4hi)__a, (__v4hi)__b);
}

#define _mm_shuffle_pi16(a, n) __extension__ ({ \
  __m64 __a = (a); \
  (__m64)__builtin_ia32_pshufw((__v4hi)__a, (n)); })

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_maskmove_si64(__m64 __d, __m64 __n, char *__p)
{
  __builtin_ia32_maskmovq((__v8qi)__d, (__v8qi)__n, __p);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_avg_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pavgb((__v8qi)__a, (__v8qi)__b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_avg_pu16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pavgw((__v4hi)__a, (__v4hi)__b);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sad_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_psadbw((__v8qi)__a, (__v8qi)__b);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
_mm_getcsr(void)
{
  return __builtin_ia32_stmxcsr();
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_setcsr(unsigned int __i)
{
  __builtin_ia32_ldmxcsr(__i);
}

#define _mm_shuffle_ps(a, b, mask) __extension__ ({ \
  __m128 __a = (a); \
  __m128 __b = (b); \
  (__m128)__builtin_shufflevector((__v4sf)__a, (__v4sf)__b, \
                                  (mask) & 0x3, ((mask) & 0xc) >> 2, \
                                  (((mask) & 0x30) >> 4) + 4, \
                                  (((mask) & 0xc0) >> 6) + 4); })

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 2, 6, 3, 7);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 0, 4, 1, 5);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_move_ss(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 4, 1, 2, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_movehl_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 6, 7, 2, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_movelh_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 0, 1, 4, 5);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi16_ps(__m64 __a)
{
  __m64 __b, __c;
  __m128 __r;

  __b = _mm_setzero_si64();
  __b = _mm_cmpgt_pi16(__b, __a);
  __c = _mm_unpackhi_pi16(__a, __b);
  __r = _mm_setzero_ps();
  __r = _mm_cvtpi32_ps(__r, __c);
  __r = _mm_movelh_ps(__r, __r);
  __c = _mm_unpacklo_pi16(__a, __b);
  __r = _mm_cvtpi32_ps(__r, __c);

  return __r;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpu16_ps(__m64 __a)
{
  __m64 __b, __c;
  __m128 __r;

  __b = _mm_setzero_si64();
  __c = _mm_unpackhi_pi16(__a, __b);
  __r = _mm_setzero_ps();
  __r = _mm_cvtpi32_ps(__r, __c);
  __r = _mm_movelh_ps(__r, __r);
  __c = _mm_unpacklo_pi16(__a, __b);
  __r = _mm_cvtpi32_ps(__r, __c);

  return __r;
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi8_ps(__m64 __a)
{
  __m64 __b;
  
  __b = _mm_setzero_si64();
  __b = _mm_cmpgt_pi8(__b, __a);
  __b = _mm_unpacklo_pi8(__a, __b);

  return _mm_cvtpi16_ps(__b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpu8_ps(__m64 __a)
{
  __m64 __b;
  
  __b = _mm_setzero_si64();
  __b = _mm_unpacklo_pi8(__a, __b);

  return _mm_cvtpi16_ps(__b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi32x2_ps(__m64 __a, __m64 __b)
{
  __m128 __c;
  
  __c = _mm_setzero_ps();
  __c = _mm_cvtpi32_ps(__c, __b);
  __c = _mm_movelh_ps(__c, __c);

  return _mm_cvtpi32_ps(__c, __a);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_pi16(__m128 __a)
{
  __m64 __b, __c;
  
  __b = _mm_cvtps_pi32(__a);
  __a = _mm_movehl_ps(__a, __a);
  __c = _mm_cvtps_pi32(__a);
  
  return _mm_packs_pi32(__b, __c);
}

static __inline__ __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_pi8(__m128 __a)
{
  __m64 __b, __c;
  
  __b = _mm_cvtps_pi16(__a);
  __c = _mm_setzero_si64();
  
  return _mm_packs_pi16(__b, __c);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_movemask_ps(__m128 __a)
{
  return __builtin_ia32_movmskps(__a);
}

#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

#define _MM_EXCEPT_INVALID    (0x0001)
#define _MM_EXCEPT_DENORM     (0x0002)
#define _MM_EXCEPT_DIV_ZERO   (0x0004)
#define _MM_EXCEPT_OVERFLOW   (0x0008)
#define _MM_EXCEPT_UNDERFLOW  (0x0010)
#define _MM_EXCEPT_INEXACT    (0x0020)
#define _MM_EXCEPT_MASK       (0x003f)

#define _MM_MASK_INVALID      (0x0080)
#define _MM_MASK_DENORM       (0x0100)
#define _MM_MASK_DIV_ZERO     (0x0200)
#define _MM_MASK_OVERFLOW     (0x0400)
#define _MM_MASK_UNDERFLOW    (0x0800)
#define _MM_MASK_INEXACT      (0x1000)
#define _MM_MASK_MASK         (0x1f80)

#define _MM_ROUND_NEAREST     (0x0000)
#define _MM_ROUND_DOWN        (0x2000)
#define _MM_ROUND_UP          (0x4000)
#define _MM_ROUND_TOWARD_ZERO (0x6000)
#define _MM_ROUND_MASK        (0x6000)

#define _MM_FLUSH_ZERO_MASK   (0x8000)
#define _MM_FLUSH_ZERO_ON     (0x8000)
#define _MM_FLUSH_ZERO_OFF    (0x0000)

#define _MM_GET_EXCEPTION_MASK() (_mm_getcsr() & _MM_MASK_MASK)
#define _MM_GET_EXCEPTION_STATE() (_mm_getcsr() & _MM_EXCEPT_MASK)
#define _MM_GET_FLUSH_ZERO_MODE() (_mm_getcsr() & _MM_FLUSH_ZERO_MASK)
#define _MM_GET_ROUNDING_MODE() (_mm_getcsr() & _MM_ROUND_MASK)

#define _MM_SET_EXCEPTION_MASK(x) (_mm_setcsr((_mm_getcsr() & ~_MM_MASK_MASK) | (x)))
#define _MM_SET_EXCEPTION_STATE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_EXCEPT_MASK) | (x)))
#define _MM_SET_FLUSH_ZERO_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_FLUSH_ZERO_MASK) | (x)))
#define _MM_SET_ROUNDING_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_ROUND_MASK) | (x)))

#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)

/* Aliases for compatibility. */
#define _m_pextrw _mm_extract_pi16
#define _m_pinsrw _mm_insert_pi16
#define _m_pmaxsw _mm_max_pi16
#define _m_pmaxub _mm_max_pu8
#define _m_pminsw _mm_min_pi16
#define _m_pminub _mm_min_pu8
#define _m_pmovmskb _mm_movemask_pi8
#define _m_pmulhuw _mm_mulhi_pu16
#define _m_pshufw _mm_shuffle_pi16
#define _m_maskmovq _mm_maskmove_si64
#define _m_pavgb _mm_avg_pu8
#define _m_pavgw _mm_avg_pu16
#define _m_psadbw _mm_sad_pu8
#define _m_ _mm_
#define _m_ _mm_

/* Ugly hack for backwards-compatibility (compatible with gcc) */
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#endif /* __SSE__ */

#endif /* __XMMINTRIN_H */
