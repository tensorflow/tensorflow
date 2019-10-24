/*===---- avxintrin.h - AVX intrinsics -------------------------------------===
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

#ifndef __IMMINTRIN_H
#error "Never use <avxintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVXINTRIN_H
#define __AVXINTRIN_H

typedef double __v4df __attribute__ ((__vector_size__ (32)));
typedef float __v8sf __attribute__ ((__vector_size__ (32)));
typedef long long __v4di __attribute__ ((__vector_size__ (32)));
typedef int __v8si __attribute__ ((__vector_size__ (32)));
typedef short __v16hi __attribute__ ((__vector_size__ (32)));
typedef char __v32qi __attribute__ ((__vector_size__ (32)));

typedef float __m256 __attribute__ ((__vector_size__ (32)));
typedef double __m256d __attribute__((__vector_size__(32)));
typedef long long __m256i __attribute__((__vector_size__(32)));

/* Arithmetic */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_add_pd(__m256d __a, __m256d __b)
{
  return __a+__b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_add_ps(__m256 __a, __m256 __b)
{
  return __a+__b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_sub_pd(__m256d __a, __m256d __b)
{
  return __a-__b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_sub_ps(__m256 __a, __m256 __b)
{
  return __a-__b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_addsub_pd(__m256d __a, __m256d __b)
{
  return (__m256d)__builtin_ia32_addsubpd256((__v4df)__a, (__v4df)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_addsub_ps(__m256 __a, __m256 __b)
{
  return (__m256)__builtin_ia32_addsubps256((__v8sf)__a, (__v8sf)__b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_div_pd(__m256d __a, __m256d __b)
{
  return __a / __b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_div_ps(__m256 __a, __m256 __b)
{
  return __a / __b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_max_pd(__m256d __a, __m256d __b)
{
  return (__m256d)__builtin_ia32_maxpd256((__v4df)__a, (__v4df)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_max_ps(__m256 __a, __m256 __b)
{
  return (__m256)__builtin_ia32_maxps256((__v8sf)__a, (__v8sf)__b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_min_pd(__m256d __a, __m256d __b)
{
  return (__m256d)__builtin_ia32_minpd256((__v4df)__a, (__v4df)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_min_ps(__m256 __a, __m256 __b)
{
  return (__m256)__builtin_ia32_minps256((__v8sf)__a, (__v8sf)__b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_mul_pd(__m256d __a, __m256d __b)
{
  return __a * __b;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_mul_ps(__m256 __a, __m256 __b)
{
  return __a * __b;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_sqrt_pd(__m256d __a)
{
  return (__m256d)__builtin_ia32_sqrtpd256((__v4df)__a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_sqrt_ps(__m256 __a)
{
  return (__m256)__builtin_ia32_sqrtps256((__v8sf)__a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_rsqrt_ps(__m256 __a)
{
  return (__m256)__builtin_ia32_rsqrtps256((__v8sf)__a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_rcp_ps(__m256 __a)
{
  return (__m256)__builtin_ia32_rcpps256((__v8sf)__a);
}

#define _mm256_round_pd(V, M) __extension__ ({ \
    __m256d __V = (V); \
    (__m256d)__builtin_ia32_roundpd256((__v4df)__V, (M)); })

#define _mm256_round_ps(V, M) __extension__ ({ \
  __m256 __V = (V); \
  (__m256)__builtin_ia32_roundps256((__v8sf)__V, (M)); })

#define _mm256_ceil_pd(V)  _mm256_round_pd((V), _MM_FROUND_CEIL)
#define _mm256_floor_pd(V) _mm256_round_pd((V), _MM_FROUND_FLOOR)
#define _mm256_ceil_ps(V)  _mm256_round_ps((V), _MM_FROUND_CEIL)
#define _mm256_floor_ps(V) _mm256_round_ps((V), _MM_FROUND_FLOOR)

/* Logical */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_and_pd(__m256d __a, __m256d __b)
{
  return (__m256d)((__v4di)__a & (__v4di)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_and_ps(__m256 __a, __m256 __b)
{
  return (__m256)((__v8si)__a & (__v8si)__b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_andnot_pd(__m256d __a, __m256d __b)
{
  return (__m256d)(~(__v4di)__a & (__v4di)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_andnot_ps(__m256 __a, __m256 __b)
{
  return (__m256)(~(__v8si)__a & (__v8si)__b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_or_pd(__m256d __a, __m256d __b)
{
  return (__m256d)((__v4di)__a | (__v4di)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_or_ps(__m256 __a, __m256 __b)
{
  return (__m256)((__v8si)__a | (__v8si)__b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_xor_pd(__m256d __a, __m256d __b)
{
  return (__m256d)((__v4di)__a ^ (__v4di)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_xor_ps(__m256 __a, __m256 __b)
{
  return (__m256)((__v8si)__a ^ (__v8si)__b);
}

/* Horizontal arithmetic */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_pd(__m256d __a, __m256d __b)
{
  return (__m256d)__builtin_ia32_haddpd256((__v4df)__a, (__v4df)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_ps(__m256 __a, __m256 __b)
{
  return (__m256)__builtin_ia32_haddps256((__v8sf)__a, (__v8sf)__b);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_pd(__m256d __a, __m256d __b)
{
  return (__m256d)__builtin_ia32_hsubpd256((__v4df)__a, (__v4df)__b);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_ps(__m256 __a, __m256 __b)
{
  return (__m256)__builtin_ia32_hsubps256((__v8sf)__a, (__v8sf)__b);
}

/* Vector permutations */
static __inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm_permutevar_pd(__m128d __a, __m128i __c)
{
  return (__m128d)__builtin_ia32_vpermilvarpd((__v2df)__a, (__v2di)__c);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_permutevar_pd(__m256d __a, __m256i __c)
{
  return (__m256d)__builtin_ia32_vpermilvarpd256((__v4df)__a, (__v4di)__c);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_permutevar_ps(__m128 __a, __m128i __c)
{
  return (__m128)__builtin_ia32_vpermilvarps((__v4sf)__a, (__v4si)__c);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_permutevar_ps(__m256 __a, __m256i __c)
{
  return (__m256)__builtin_ia32_vpermilvarps256((__v8sf)__a,
						  (__v8si)__c);
}

#define _mm_permute_pd(A, C) __extension__ ({ \
  __m128d __A = (A); \
  (__m128d)__builtin_shufflevector((__v2df)__A, (__v2df) _mm_setzero_pd(), \
                                   (C) & 0x1, ((C) & 0x2) >> 1); })

#define _mm256_permute_pd(A, C) __extension__ ({ \
  __m256d __A = (A); \
  (__m256d)__builtin_shufflevector((__v4df)__A, (__v4df) _mm256_setzero_pd(), \
                                   (C) & 0x1, ((C) & 0x2) >> 1, \
                                   2 + (((C) & 0x4) >> 2), \
                                   2 + (((C) & 0x8) >> 3)); })

#define _mm_permute_ps(A, C) __extension__ ({ \
  __m128 __A = (A); \
  (__m128)__builtin_shufflevector((__v4sf)__A, (__v4sf) _mm_setzero_ps(), \
                                   (C) & 0x3, ((C) & 0xc) >> 2, \
                                   ((C) & 0x30) >> 4, ((C) & 0xc0) >> 6); })

#define _mm256_permute_ps(A, C) __extension__ ({ \
  __m256 __A = (A); \
  (__m256)__builtin_shufflevector((__v8sf)__A, (__v8sf) _mm256_setzero_ps(), \
                                  (C) & 0x3, ((C) & 0xc) >> 2, \
                                  ((C) & 0x30) >> 4, ((C) & 0xc0) >> 6, \
                                  4 + (((C) & 0x03) >> 0), \
                                  4 + (((C) & 0x0c) >> 2), \
                                  4 + (((C) & 0x30) >> 4), \
                                  4 + (((C) & 0xc0) >> 6)); })

#define _mm256_permute2f128_pd(V1, V2, M) __extension__ ({ \
  __m256d __V1 = (V1); \
  __m256d __V2 = (V2); \
  (__m256d)__builtin_ia32_vperm2f128_pd256((__v4df)__V1, (__v4df)__V2, (M)); })

#define _mm256_permute2f128_ps(V1, V2, M) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m256 __V2 = (V2); \
  (__m256)__builtin_ia32_vperm2f128_ps256((__v8sf)__V1, (__v8sf)__V2, (M)); })

#define _mm256_permute2f128_si256(V1, V2, M) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m256i __V2 = (V2); \
  (__m256i)__builtin_ia32_vperm2f128_si256((__v8si)__V1, (__v8si)__V2, (M)); })

/* Vector Blend */
#define _mm256_blend_pd(V1, V2, M) __extension__ ({ \
  __m256d __V1 = (V1); \
  __m256d __V2 = (V2); \
  (__m256d)__builtin_shufflevector((__v4df)__V1, (__v4df)__V2, \
                                   (((M) & 0x01) ? 4 : 0), \
                                   (((M) & 0x02) ? 5 : 1), \
                                   (((M) & 0x04) ? 6 : 2), \
                                   (((M) & 0x08) ? 7 : 3)); })

#define _mm256_blend_ps(V1, V2, M) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m256 __V2 = (V2); \
  (__m256)__builtin_shufflevector((__v8sf)__V1, (__v8sf)__V2, \
                                  (((M) & 0x01) ?  8 : 0), \
                                  (((M) & 0x02) ?  9 : 1), \
                                  (((M) & 0x04) ? 10 : 2), \
                                  (((M) & 0x08) ? 11 : 3), \
                                  (((M) & 0x10) ? 12 : 4), \
                                  (((M) & 0x20) ? 13 : 5), \
                                  (((M) & 0x40) ? 14 : 6), \
                                  (((M) & 0x80) ? 15 : 7)); })

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_blendv_pd(__m256d __a, __m256d __b, __m256d __c)
{
  return (__m256d)__builtin_ia32_blendvpd256(
    (__v4df)__a, (__v4df)__b, (__v4df)__c);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_blendv_ps(__m256 __a, __m256 __b, __m256 __c)
{
  return (__m256)__builtin_ia32_blendvps256(
    (__v8sf)__a, (__v8sf)__b, (__v8sf)__c);
}

/* Vector Dot Product */
#define _mm256_dp_ps(V1, V2, M) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m256 __V2 = (V2); \
  (__m256)__builtin_ia32_dpps256((__v8sf)__V1, (__v8sf)__V2, (M)); })

/* Vector shuffle */
#define _mm256_shuffle_ps(a, b, mask) __extension__ ({ \
        __m256 __a = (a); \
        __m256 __b = (b); \
        (__m256)__builtin_shufflevector((__v8sf)__a, (__v8sf)__b, \
        (mask) & 0x3,                ((mask) & 0xc) >> 2, \
        (((mask) & 0x30) >> 4) + 8,  (((mask) & 0xc0) >> 6) + 8, \
        ((mask) & 0x3) + 4,          (((mask) & 0xc) >> 2) + 4, \
        (((mask) & 0x30) >> 4) + 12, (((mask) & 0xc0) >> 6) + 12); })

#define _mm256_shuffle_pd(a, b, mask) __extension__ ({ \
        __m256d __a = (a); \
        __m256d __b = (b); \
        (__m256d)__builtin_shufflevector((__v4df)__a, (__v4df)__b, \
        (mask) & 0x1, \
        (((mask) & 0x2) >> 1) + 4, \
        (((mask) & 0x4) >> 2) + 2, \
        (((mask) & 0x8) >> 3) + 6); })

/* Compare */
#define _CMP_EQ_OQ    0x00 /* Equal (ordered, non-signaling)  */
#define _CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
#define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
#define _CMP_UNORD_Q  0x03 /* Unordered (non-signaling)  */
#define _CMP_NEQ_UQ   0x04 /* Not-equal (unordered, non-signaling)  */
#define _CMP_NLT_US   0x05 /* Not-less-than (unordered, signaling)  */
#define _CMP_NLE_US   0x06 /* Not-less-than-or-equal (unordered, signaling)  */
#define _CMP_ORD_Q    0x07 /* Ordered (nonsignaling)   */
#define _CMP_EQ_UQ    0x08 /* Equal (unordered, non-signaling)  */
#define _CMP_NGE_US   0x09 /* Not-greater-than-or-equal (unord, signaling)  */
#define _CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */
#define _CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
#define _CMP_NEQ_OQ   0x0c /* Not-equal (ordered, non-signaling)  */
#define _CMP_GE_OS    0x0d /* Greater-than-or-equal (ordered, signaling)  */
#define _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
#define _CMP_TRUE_UQ  0x0f /* True (unordered, non-signaling)  */
#define _CMP_EQ_OS    0x10 /* Equal (ordered, signaling)  */
#define _CMP_LT_OQ    0x11 /* Less-than (ordered, non-signaling)  */
#define _CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
#define _CMP_UNORD_S  0x13 /* Unordered (signaling)  */
#define _CMP_NEQ_US   0x14 /* Not-equal (unordered, signaling)  */
#define _CMP_NLT_UQ   0x15 /* Not-less-than (unordered, non-signaling)  */
#define _CMP_NLE_UQ   0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
#define _CMP_ORD_S    0x17 /* Ordered (signaling)  */
#define _CMP_EQ_US    0x18 /* Equal (unordered, signaling)  */
#define _CMP_NGE_UQ   0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
#define _CMP_NGT_UQ   0x1a /* Not-greater-than (unordered, non-signaling)  */
#define _CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
#define _CMP_NEQ_OS   0x1c /* Not-equal (ordered, signaling)  */
#define _CMP_GE_OQ    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
#define _CMP_GT_OQ    0x1e /* Greater-than (ordered, non-signaling)  */
#define _CMP_TRUE_US  0x1f /* True (unordered, signaling)  */

#define _mm_cmp_pd(a, b, c) __extension__ ({ \
  __m128d __a = (a); \
  __m128d __b = (b); \
  (__m128d)__builtin_ia32_cmppd((__v2df)__a, (__v2df)__b, (c)); })

#define _mm_cmp_ps(a, b, c) __extension__ ({ \
  __m128 __a = (a); \
  __m128 __b = (b); \
  (__m128)__builtin_ia32_cmpps((__v4sf)__a, (__v4sf)__b, (c)); })

#define _mm256_cmp_pd(a, b, c) __extension__ ({ \
  __m256d __a = (a); \
  __m256d __b = (b); \
  (__m256d)__builtin_ia32_cmppd256((__v4df)__a, (__v4df)__b, (c)); })

#define _mm256_cmp_ps(a, b, c) __extension__ ({ \
  __m256 __a = (a); \
  __m256 __b = (b); \
  (__m256)__builtin_ia32_cmpps256((__v8sf)__a, (__v8sf)__b, (c)); })

#define _mm_cmp_sd(a, b, c) __extension__ ({ \
  __m128d __a = (a); \
  __m128d __b = (b); \
  (__m128d)__builtin_ia32_cmpsd((__v2df)__a, (__v2df)__b, (c)); })

#define _mm_cmp_ss(a, b, c) __extension__ ({ \
  __m128 __a = (a); \
  __m128 __b = (b); \
  (__m128)__builtin_ia32_cmpss((__v4sf)__a, (__v4sf)__b, (c)); })

/* Vector extract */
#define _mm256_extractf128_pd(A, O) __extension__ ({ \
  __m256d __A = (A); \
  (__m128d)__builtin_ia32_vextractf128_pd256((__v4df)__A, (O)); })

#define _mm256_extractf128_ps(A, O) __extension__ ({ \
  __m256 __A = (A); \
  (__m128)__builtin_ia32_vextractf128_ps256((__v8sf)__A, (O)); })

#define _mm256_extractf128_si256(A, O) __extension__ ({ \
  __m256i __A = (A); \
  (__m128i)__builtin_ia32_vextractf128_si256((__v8si)__A, (O)); })

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi32(__m256i __a, int const __imm)
{
  __v8si __b = (__v8si)__a;
  return __b[__imm & 7];
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi16(__m256i __a, int const __imm)
{
  __v16hi __b = (__v16hi)__a;
  return __b[__imm & 15];
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi8(__m256i __a, int const __imm)
{
  __v32qi __b = (__v32qi)__a;
  return __b[__imm & 31];
}

#ifdef __x86_64__
static __inline long long  __attribute__((__always_inline__, __nodebug__))
_mm256_extract_epi64(__m256i __a, const int __imm)
{
  __v4di __b = (__v4di)__a;
  return __b[__imm & 3];
}
#endif

/* Vector insert */
#define _mm256_insertf128_pd(V1, V2, O) __extension__ ({ \
  __m256d __V1 = (V1); \
  __m128d __V2 = (V2); \
  (__m256d)__builtin_ia32_vinsertf128_pd256((__v4df)__V1, (__v2df)__V2, (O)); })

#define _mm256_insertf128_ps(V1, V2, O) __extension__ ({ \
  __m256 __V1 = (V1); \
  __m128 __V2 = (V2); \
  (__m256)__builtin_ia32_vinsertf128_ps256((__v8sf)__V1, (__v4sf)__V2, (O)); })

#define _mm256_insertf128_si256(V1, V2, O) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m128i __V2 = (V2); \
  (__m256i)__builtin_ia32_vinsertf128_si256((__v8si)__V1, (__v4si)__V2, (O)); })

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi32(__m256i __a, int __b, int const __imm)
{
  __v8si __c = (__v8si)__a;
  __c[__imm & 7] = __b;
  return (__m256i)__c;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi16(__m256i __a, int __b, int const __imm)
{
  __v16hi __c = (__v16hi)__a;
  __c[__imm & 15] = __b;
  return (__m256i)__c;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi8(__m256i __a, int __b, int const __imm)
{
  __v32qi __c = (__v32qi)__a;
  __c[__imm & 31] = __b;
  return (__m256i)__c;
}

#ifdef __x86_64__
static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_insert_epi64(__m256i __a, int __b, int const __imm)
{
  __v4di __c = (__v4di)__a;
  __c[__imm & 3] = __b;
  return (__m256i)__c;
}
#endif

/* Conversion */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi32_pd(__m128i __a)
{
  return (__m256d)__builtin_ia32_cvtdq2pd256((__v4si) __a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi32_ps(__m256i __a)
{
  return (__m256)__builtin_ia32_cvtdq2ps256((__v8si) __a);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm256_cvtpd_ps(__m256d __a)
{
  return (__m128)__builtin_ia32_cvtpd2ps256((__v4df) __a);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtps_epi32(__m256 __a)
{
  return (__m256i)__builtin_ia32_cvtps2dq256((__v8sf) __a);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_cvtps_pd(__m128 __a)
{
  return (__m256d)__builtin_ia32_cvtps2pd256((__v4sf) __a);
}

static __inline __m128i __attribute__((__always_inline__, __nodebug__))
_mm256_cvttpd_epi32(__m256d __a)
{
  return (__m128i)__builtin_ia32_cvttpd2dq256((__v4df) __a);
}

static __inline __m128i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtpd_epi32(__m256d __a)
{
  return (__m128i)__builtin_ia32_cvtpd2dq256((__v4df) __a);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvttps_epi32(__m256 __a)
{
  return (__m256i)__builtin_ia32_cvttps2dq256((__v8sf) __a);
}

/* Vector replicate */
static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_movehdup_ps(__m256 __a)
{
  return __builtin_shufflevector(__a, __a, 1, 1, 3, 3, 5, 5, 7, 7);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_moveldup_ps(__m256 __a)
{
  return __builtin_shufflevector(__a, __a, 0, 0, 2, 2, 4, 4, 6, 6);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_movedup_pd(__m256d __a)
{
  return __builtin_shufflevector(__a, __a, 0, 0, 2, 2);
}

/* Unpack and Interleave */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_pd(__m256d __a, __m256d __b)
{
  return __builtin_shufflevector(__a, __b, 1, 5, 1+2, 5+2);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_pd(__m256d __a, __m256d __b)
{
  return __builtin_shufflevector(__a, __b, 0, 4, 0+2, 4+2);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_ps(__m256 __a, __m256 __b)
{
  return __builtin_shufflevector(__a, __b, 2, 10, 2+1, 10+1, 6, 14, 6+1, 14+1);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_ps(__m256 __a, __m256 __b)
{
  return __builtin_shufflevector(__a, __b, 0, 8, 0+1, 8+1, 4, 12, 4+1, 12+1);
}

/* Bit Test */
static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testz_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_vtestzpd((__v2df)__a, (__v2df)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testc_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_vtestcpd((__v2df)__a, (__v2df)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testnzc_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_vtestnzcpd((__v2df)__a, (__v2df)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testz_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_vtestzps((__v4sf)__a, (__v4sf)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testc_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_vtestcps((__v4sf)__a, (__v4sf)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm_testnzc_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_vtestnzcps((__v4sf)__a, (__v4sf)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testz_pd(__m256d __a, __m256d __b)
{
  return __builtin_ia32_vtestzpd256((__v4df)__a, (__v4df)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testc_pd(__m256d __a, __m256d __b)
{
  return __builtin_ia32_vtestcpd256((__v4df)__a, (__v4df)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testnzc_pd(__m256d __a, __m256d __b)
{
  return __builtin_ia32_vtestnzcpd256((__v4df)__a, (__v4df)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testz_ps(__m256 __a, __m256 __b)
{
  return __builtin_ia32_vtestzps256((__v8sf)__a, (__v8sf)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testc_ps(__m256 __a, __m256 __b)
{
  return __builtin_ia32_vtestcps256((__v8sf)__a, (__v8sf)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testnzc_ps(__m256 __a, __m256 __b)
{
  return __builtin_ia32_vtestnzcps256((__v8sf)__a, (__v8sf)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testz_si256(__m256i __a, __m256i __b)
{
  return __builtin_ia32_ptestz256((__v4di)__a, (__v4di)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testc_si256(__m256i __a, __m256i __b)
{
  return __builtin_ia32_ptestc256((__v4di)__a, (__v4di)__b);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_testnzc_si256(__m256i __a, __m256i __b)
{
  return __builtin_ia32_ptestnzc256((__v4di)__a, (__v4di)__b);
}

/* Vector extract sign mask */
static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_movemask_pd(__m256d __a)
{
  return __builtin_ia32_movmskpd256((__v4df)__a);
}

static __inline int __attribute__((__always_inline__, __nodebug__))
_mm256_movemask_ps(__m256 __a)
{
  return __builtin_ia32_movmskps256((__v8sf)__a);
}

/* Vector __zero */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_zeroall(void)
{
  __builtin_ia32_vzeroall();
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_zeroupper(void)
{
  __builtin_ia32_vzeroupper();
}

/* Vector load with broadcast */
static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_broadcast_ss(float const *__a)
{
  float __f = *__a;
  return (__m128)(__v4sf){ __f, __f, __f, __f };
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_sd(double const *__a)
{
  double __d = *__a;
  return (__m256d)(__v4df){ __d, __d, __d, __d };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_ss(float const *__a)
{
  float __f = *__a;
  return (__m256)(__v8sf){ __f, __f, __f, __f, __f, __f, __f, __f };
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_pd(__m128d const *__a)
{
  return (__m256d)__builtin_ia32_vbroadcastf128_pd256(__a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_broadcast_ps(__m128 const *__a)
{
  return (__m256)__builtin_ia32_vbroadcastf128_ps256(__a);
}

/* SIMD load ops */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_load_pd(double const *__p)
{
  return *(__m256d *)__p;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_load_ps(float const *__p)
{
  return *(__m256 *)__p;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_loadu_pd(double const *__p)
{
  struct __loadu_pd {
    __m256d __v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_pd*)__p)->__v;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_loadu_ps(float const *__p)
{
  struct __loadu_ps {
    __m256 __v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_ps*)__p)->__v;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_load_si256(__m256i const *__p)
{
  return *__p;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_loadu_si256(__m256i const *__p)
{
  struct __loadu_si256 {
    __m256i __v;
  } __attribute__((packed, may_alias));
  return ((struct __loadu_si256*)__p)->__v;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_lddqu_si256(__m256i const *__p)
{
  return (__m256i)__builtin_ia32_lddqu256((char const *)__p);
}

/* SIMD store ops */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_store_pd(double *__p, __m256d __a)
{
  *(__m256d *)__p = __a;
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_store_ps(float *__p, __m256 __a)
{
  *(__m256 *)__p = __a;
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu_pd(double *__p, __m256d __a)
{
  __builtin_ia32_storeupd256(__p, (__v4df)__a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu_ps(float *__p, __m256 __a)
{
  __builtin_ia32_storeups256(__p, (__v8sf)__a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_store_si256(__m256i *__p, __m256i __a)
{
  *__p = __a;
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu_si256(__m256i *__p, __m256i __a)
{
  __builtin_ia32_storedqu256((char *)__p, (__v32qi)__a);
}

/* Conditional load ops */
static __inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm_maskload_pd(double const *__p, __m128d __m)
{
  return (__m128d)__builtin_ia32_maskloadpd((const __v2df *)__p, (__v2df)__m);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_maskload_pd(double const *__p, __m256d __m)
{
  return (__m256d)__builtin_ia32_maskloadpd256((const __v4df *)__p,
                                               (__v4df)__m);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_maskload_ps(float const *__p, __m128 __m)
{
  return (__m128)__builtin_ia32_maskloadps((const __v4sf *)__p, (__v4sf)__m);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_maskload_ps(float const *__p, __m256 __m)
{
  return (__m256)__builtin_ia32_maskloadps256((const __v8sf *)__p, (__v8sf)__m);
}

/* Conditional store ops */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_maskstore_ps(float *__p, __m256 __m, __m256 __a)
{
  __builtin_ia32_maskstoreps256((__v8sf *)__p, (__v8sf)__m, (__v8sf)__a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm_maskstore_pd(double *__p, __m128d __m, __m128d __a)
{
  __builtin_ia32_maskstorepd((__v2df *)__p, (__v2df)__m, (__v2df)__a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_maskstore_pd(double *__p, __m256d __m, __m256d __a)
{
  __builtin_ia32_maskstorepd256((__v4df *)__p, (__v4df)__m, (__v4df)__a);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm_maskstore_ps(float *__p, __m128 __m, __m128 __a)
{
  __builtin_ia32_maskstoreps((__v4sf *)__p, (__v4sf)__m, (__v4sf)__a);
}

/* Cacheability support ops */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_stream_si256(__m256i *__a, __m256i __b)
{
  __builtin_ia32_movntdq256((__v4di *)__a, (__v4di)__b);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_stream_pd(double *__a, __m256d __b)
{
  __builtin_ia32_movntpd256(__a, (__v4df)__b);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_stream_ps(float *__p, __m256 __a)
{
  __builtin_ia32_movntps256(__p, (__v8sf)__a);
}

/* Create vectors */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_set_pd(double __a, double __b, double __c, double __d)
{
  return (__m256d){ __d, __c, __b, __a };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_set_ps(float __a, float __b, float __c, float __d,
	            float __e, float __f, float __g, float __h)
{
  return (__m256){ __h, __g, __f, __e, __d, __c, __b, __a };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi32(int __i0, int __i1, int __i2, int __i3,
		             int __i4, int __i5, int __i6, int __i7)
{
  return (__m256i)(__v8si){ __i7, __i6, __i5, __i4, __i3, __i2, __i1, __i0 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi16(short __w15, short __w14, short __w13, short __w12,
		             short __w11, short __w10, short __w09, short __w08,
		             short __w07, short __w06, short __w05, short __w04,
		             short __w03, short __w02, short __w01, short __w00)
{
  return (__m256i)(__v16hi){ __w00, __w01, __w02, __w03, __w04, __w05, __w06,
    __w07, __w08, __w09, __w10, __w11, __w12, __w13, __w14, __w15 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi8(char __b31, char __b30, char __b29, char __b28,
		            char __b27, char __b26, char __b25, char __b24,
		            char __b23, char __b22, char __b21, char __b20,
		            char __b19, char __b18, char __b17, char __b16,
		            char __b15, char __b14, char __b13, char __b12,
		            char __b11, char __b10, char __b09, char __b08,
		            char __b07, char __b06, char __b05, char __b04,
		            char __b03, char __b02, char __b01, char __b00)
{
  return (__m256i)(__v32qi){
    __b00, __b01, __b02, __b03, __b04, __b05, __b06, __b07,
    __b08, __b09, __b10, __b11, __b12, __b13, __b14, __b15,
    __b16, __b17, __b18, __b19, __b20, __b21, __b22, __b23,
    __b24, __b25, __b26, __b27, __b28, __b29, __b30, __b31
  };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set_epi64x(long long __a, long long __b, long long __c, long long __d)
{
  return (__m256i)(__v4di){ __d, __c, __b, __a };
}

/* Create vectors with elements in reverse order */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_setr_pd(double __a, double __b, double __c, double __d)
{
  return (__m256d){ __a, __b, __c, __d };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_setr_ps(float __a, float __b, float __c, float __d,
		           float __e, float __f, float __g, float __h)
{
  return (__m256){ __a, __b, __c, __d, __e, __f, __g, __h };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi32(int __i0, int __i1, int __i2, int __i3,
		              int __i4, int __i5, int __i6, int __i7)
{
  return (__m256i)(__v8si){ __i0, __i1, __i2, __i3, __i4, __i5, __i6, __i7 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi16(short __w15, short __w14, short __w13, short __w12,
		   short __w11, short __w10, short __w09, short __w08,
		   short __w07, short __w06, short __w05, short __w04,
		   short __w03, short __w02, short __w01, short __w00)
{
  return (__m256i)(__v16hi){ __w15, __w14, __w13, __w12, __w11, __w10, __w09,
    __w08, __w07, __w06, __w05, __w04, __w03, __w02, __w01, __w00 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi8(char __b31, char __b30, char __b29, char __b28,
		             char __b27, char __b26, char __b25, char __b24,
		             char __b23, char __b22, char __b21, char __b20,
		             char __b19, char __b18, char __b17, char __b16,
		             char __b15, char __b14, char __b13, char __b12,
		             char __b11, char __b10, char __b09, char __b08,
		             char __b07, char __b06, char __b05, char __b04,
		             char __b03, char __b02, char __b01, char __b00)
{
  return (__m256i)(__v32qi){
    __b31, __b30, __b29, __b28, __b27, __b26, __b25, __b24,
		__b23, __b22, __b21, __b20, __b19, __b18, __b17, __b16,
		__b15, __b14, __b13, __b12, __b11, __b10, __b09, __b08,
		__b07, __b06, __b05, __b04, __b03, __b02, __b01, __b00 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setr_epi64x(long long __a, long long __b, long long __c, long long __d)
{
  return (__m256i)(__v4di){ __a, __b, __c, __d };
}

/* Create vectors with repeated elements */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_set1_pd(double __w)
{
  return (__m256d){ __w, __w, __w, __w };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_set1_ps(float __w)
{
  return (__m256){ __w, __w, __w, __w, __w, __w, __w, __w };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi32(int __i)
{
  return (__m256i)(__v8si){ __i, __i, __i, __i, __i, __i, __i, __i };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi16(short __w)
{
  return (__m256i)(__v16hi){ __w, __w, __w, __w, __w, __w, __w, __w, __w, __w,
    __w, __w, __w, __w, __w, __w };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi8(char __b)
{
  return (__m256i)(__v32qi){ __b, __b, __b, __b, __b, __b, __b, __b, __b, __b,
    __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b, __b,
    __b, __b, __b, __b, __b, __b, __b };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_set1_epi64x(long long __q)
{
  return (__m256i)(__v4di){ __q, __q, __q, __q };
}

/* Create __zeroed vectors */
static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_setzero_pd(void)
{
  return (__m256d){ 0, 0, 0, 0 };
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_setzero_ps(void)
{
  return (__m256){ 0, 0, 0, 0, 0, 0, 0, 0 };
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_setzero_si256(void)
{
  return (__m256i){ 0LL, 0LL, 0LL, 0LL };
}

/* Cast between vector types */
static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_castpd_ps(__m256d __a)
{
  return (__m256)__a;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_castpd_si256(__m256d __a)
{
  return (__m256i)__a;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_castps_pd(__m256 __a)
{
  return (__m256d)__a;
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_castps_si256(__m256 __a)
{
  return (__m256i)__a;
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_castsi256_ps(__m256i __a)
{
  return (__m256)__a;
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_castsi256_pd(__m256i __a)
{
  return (__m256d)__a;
}

static __inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm256_castpd256_pd128(__m256d __a)
{
  return __builtin_shufflevector(__a, __a, 0, 1);
}

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm256_castps256_ps128(__m256 __a)
{
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3);
}

static __inline __m128i __attribute__((__always_inline__, __nodebug__))
_mm256_castsi256_si128(__m256i __a)
{
  return __builtin_shufflevector(__a, __a, 0, 1);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_castpd128_pd256(__m128d __a)
{
  return __builtin_shufflevector(__a, __a, 0, 1, -1, -1);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_castps128_ps256(__m128 __a)
{
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, -1, -1, -1, -1);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_castsi128_si256(__m128i __a)
{
  return __builtin_shufflevector(__a, __a, 0, 1, -1, -1);
}

/* SIMD load ops (unaligned) */
static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_loadu2_m128(float const *__addr_hi, float const *__addr_lo)
{
  struct __loadu_ps {
    __m128 __v;
  } __attribute__((__packed__, __may_alias__));

  __m256 __v256 = _mm256_castps128_ps256(((struct __loadu_ps*)__addr_lo)->__v);
  return _mm256_insertf128_ps(__v256, ((struct __loadu_ps*)__addr_hi)->__v, 1);
}

static __inline __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_loadu2_m128d(double const *__addr_hi, double const *__addr_lo)
{
  struct __loadu_pd {
    __m128d __v;
  } __attribute__((__packed__, __may_alias__));
  
  __m256d __v256 = _mm256_castpd128_pd256(((struct __loadu_pd*)__addr_lo)->__v);
  return _mm256_insertf128_pd(__v256, ((struct __loadu_pd*)__addr_hi)->__v, 1);
}

static __inline __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_loadu2_m128i(__m128i const *__addr_hi, __m128i const *__addr_lo)
{
  struct __loadu_si128 {
    __m128i __v;
  } __attribute__((packed, may_alias));
  __m256i __v256 = _mm256_castsi128_si256(
    ((struct __loadu_si128*)__addr_lo)->__v);
  return _mm256_insertf128_si256(__v256,
                                 ((struct __loadu_si128*)__addr_hi)->__v, 1);
}

/* SIMD store ops (unaligned) */
static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu2_m128(float *__addr_hi, float *__addr_lo, __m256 __a)
{
  __m128 __v128;

  __v128 = _mm256_castps256_ps128(__a);
  __builtin_ia32_storeups(__addr_lo, __v128);
  __v128 = _mm256_extractf128_ps(__a, 1);
  __builtin_ia32_storeups(__addr_hi, __v128);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu2_m128d(double *__addr_hi, double *__addr_lo, __m256d __a)
{
  __m128d __v128;

  __v128 = _mm256_castpd256_pd128(__a);
  __builtin_ia32_storeupd(__addr_lo, __v128);
  __v128 = _mm256_extractf128_pd(__a, 1);
  __builtin_ia32_storeupd(__addr_hi, __v128);
}

static __inline void __attribute__((__always_inline__, __nodebug__))
_mm256_storeu2_m128i(__m128i *__addr_hi, __m128i *__addr_lo, __m256i __a)
{
  __m128i __v128;

  __v128 = _mm256_castsi256_si128(__a);
  __builtin_ia32_storedqu((char *)__addr_lo, (__v16qi)__v128);
  __v128 = _mm256_extractf128_si256(__a, 1);
  __builtin_ia32_storedqu((char *)__addr_hi, (__v16qi)__v128);
}

#endif /* __AVXINTRIN_H */
