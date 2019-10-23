/*===---- avx2intrin.h - AVX2 intrinsics -----------------------------------===
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
#error "Never use <avx2intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX2INTRIN_H
#define __AVX2INTRIN_H

/* SSE4 Multiple Packed Sums of Absolute Difference.  */
#define _mm256_mpsadbw_epu8(X, Y, M) __builtin_ia32_mpsadbw256((X), (Y), (M))

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_abs_epi8(__m256i __a)
{
    return (__m256i)__builtin_ia32_pabsb256((__v32qi)__a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_abs_epi16(__m256i __a)
{
    return (__m256i)__builtin_ia32_pabsw256((__v16hi)__a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_abs_epi32(__m256i __a)
{
    return (__m256i)__builtin_ia32_pabsd256((__v8si)__a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packs_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_packsswb256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packs_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_packssdw256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packus_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_packuswb256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packus_epi32(__m256i __V1, __m256i __V2)
{
  return (__m256i) __builtin_ia32_packusdw256((__v8si)__V1, (__v8si)__V2);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qi)__a + (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hi)__a + (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a + (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi64(__m256i __a, __m256i __b)
{
  return __a + __b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_paddsb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_paddsw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_paddusb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_paddusw256((__v16hi)__a, (__v16hi)__b);
}

#define _mm256_alignr_epi8(a, b, n) __extension__ ({ \
  __m256i __a = (a); \
  __m256i __b = (b); \
  (__m256i)__builtin_ia32_palignr256((__v32qi)__a, (__v32qi)__b, (n)); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_and_si256(__m256i __a, __m256i __b)
{
  return __a & __b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_andnot_si256(__m256i __a, __m256i __b)
{
  return ~__a & __b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_avg_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pavgb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_avg_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pavgw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_blendv_epi8(__m256i __V1, __m256i __V2, __m256i __M)
{
  return (__m256i)__builtin_ia32_pblendvb256((__v32qi)__V1, (__v32qi)__V2,
                                              (__v32qi)__M);
}

#define _mm256_blend_epi16(V1, V2, M) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m256i __V2 = (V2); \
  (__m256d)__builtin_shufflevector((__v16hi)__V1, (__v16hi)__V2, \
                                   (((M) & 0x01) ? 16 : 0), \
                                   (((M) & 0x02) ? 17 : 1), \
                                   (((M) & 0x04) ? 18 : 2), \
                                   (((M) & 0x08) ? 19 : 3), \
                                   (((M) & 0x10) ? 20 : 4), \
                                   (((M) & 0x20) ? 21 : 5), \
                                   (((M) & 0x40) ? 22 : 6), \
                                   (((M) & 0x80) ? 23 : 7), \
                                   (((M) & 0x01) ? 24 : 8), \
                                   (((M) & 0x02) ? 25 : 9), \
                                   (((M) & 0x04) ? 26 : 10), \
                                   (((M) & 0x08) ? 27 : 11), \
                                   (((M) & 0x10) ? 28 : 12), \
                                   (((M) & 0x20) ? 29 : 13), \
                                   (((M) & 0x40) ? 30 : 14), \
                                   (((M) & 0x80) ? 31 : 15)); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qi)__a == (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hi)__a == (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a == (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)(__a == __b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qi)__a > (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hi)__a > (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a > (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)(__a > __b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phaddw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_epi32(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phaddd256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hadds_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phaddsw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phsubw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_epi32(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phsubd256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hsubs_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_phsubsw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maddubs_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_pmaddubsw256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_madd_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaddwd256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaxsb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaxsw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaxsd256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaxub256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaxuw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epu32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmaxud256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pminsb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pminsw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pminsd256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pminub256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pminuw256 ((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epu32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pminud256((__v8si)__a, (__v8si)__b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm256_movemask_epi8(__m256i __a)
{
  return __builtin_ia32_pmovmskb256((__v32qi)__a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi8_epi16(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxbw256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi8_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxbd256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi8_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxbq256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi16_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxwd256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi16_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxwq256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi32_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxdq256((__v4si)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu8_epi16(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxbw256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu8_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxbd256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu8_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxbq256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu16_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxwd256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu16_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxwq256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu32_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxdq256((__v4si)__V);
}

static __inline__  __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mul_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmuldq256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mulhrs_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmulhrsw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mulhi_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmulhuw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mulhi_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pmulhw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mullo_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hi)__a * (__v16hi)__b);
}

static __inline__  __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mullo_epi32 (__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a * (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mul_epu32(__m256i __a, __m256i __b)
{
  return __builtin_ia32_pmuludq256((__v8si)__a, (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_or_si256(__m256i __a, __m256i __b)
{
  return __a | __b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sad_epu8(__m256i __a, __m256i __b)
{
  return __builtin_ia32_psadbw256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_shuffle_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_pshufb256((__v32qi)__a, (__v32qi)__b);
}

#define _mm256_shuffle_epi32(a, imm) __extension__ ({ \
  __m256i __a = (a); \
  (__m256i)__builtin_shufflevector((__v8si)__a, (__v8si)_mm256_set1_epi32(0), \
                                   (imm) & 0x3, ((imm) & 0xc) >> 2, \
                                   ((imm) & 0x30) >> 4, ((imm) & 0xc0) >> 6, \
                                   4 + (((imm) & 0x03) >> 0), \
                                   4 + (((imm) & 0x0c) >> 2), \
                                   4 + (((imm) & 0x30) >> 4), \
                                   4 + (((imm) & 0xc0) >> 6)); })

#define _mm256_shufflehi_epi16(a, imm) __extension__ ({ \
  __m256i __a = (a); \
  (__m256i)__builtin_shufflevector((__v16hi)__a, (__v16hi)_mm256_set1_epi16(0), \
                                   0, 1, 2, 3, \
                                   4 + (((imm) & 0x03) >> 0), \
                                   4 + (((imm) & 0x0c) >> 2), \
                                   4 + (((imm) & 0x30) >> 4), \
                                   4 + (((imm) & 0xc0) >> 6), \
                                   8, 9, 10, 11, \
                                   12 + (((imm) & 0x03) >> 0), \
                                   12 + (((imm) & 0x0c) >> 2), \
                                   12 + (((imm) & 0x30) >> 4), \
                                   12 + (((imm) & 0xc0) >> 6)); })

#define _mm256_shufflelo_epi16(a, imm) __extension__ ({ \
  __m256i __a = (a); \
  (__m256i)__builtin_shufflevector((__v16hi)__a, (__v16hi)_mm256_set1_epi16(0), \
                                   (imm) & 0x3,((imm) & 0xc) >> 2, \
                                   ((imm) & 0x30) >> 4, ((imm) & 0xc0) >> 6, \
                                   4, 5, 6, 7, \
                                   8 + (((imm) & 0x03) >> 0), \
                                   8 + (((imm) & 0x0c) >> 2), \
                                   8 + (((imm) & 0x30) >> 4), \
                                   8 + (((imm) & 0xc0) >> 6), \
                                   12, 13, 14, 15); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sign_epi8(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_psignb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sign_epi16(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_psignw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sign_epi32(__m256i __a, __m256i __b)
{
    return (__m256i)__builtin_ia32_psignd256((__v8si)__a, (__v8si)__b);
}

#define _mm256_slli_si256(a, count) __extension__ ({ \
  __m256i __a = (a); \
  (__m256i)__builtin_ia32_pslldqi256(__a, (count)*8); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_slli_epi16(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psllwi256((__v16hi)__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sll_epi16(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psllw256((__v16hi)__a, (__v8hi)__count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_slli_epi32(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_pslldi256((__v8si)__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sll_epi32(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_pslld256((__v8si)__a, (__v4si)__count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_slli_epi64(__m256i __a, int __count)
{
  return __builtin_ia32_psllqi256(__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sll_epi64(__m256i __a, __m128i __count)
{
  return __builtin_ia32_psllq256(__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srai_epi16(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psrawi256((__v16hi)__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sra_epi16(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psraw256((__v16hi)__a, (__v8hi)__count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srai_epi32(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psradi256((__v8si)__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sra_epi32(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psrad256((__v8si)__a, (__v4si)__count);
}

#define _mm256_srli_si256(a, count) __extension__ ({ \
  __m256i __a = (a); \
  (__m256i)__builtin_ia32_psrldqi256(__a, (count)*8); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srli_epi16(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psrlwi256((__v16hi)__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srl_epi16(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psrlw256((__v16hi)__a, (__v8hi)__count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srli_epi32(__m256i __a, int __count)
{
  return (__m256i)__builtin_ia32_psrldi256((__v8si)__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srl_epi32(__m256i __a, __m128i __count)
{
  return (__m256i)__builtin_ia32_psrld256((__v8si)__a, (__v4si)__count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srli_epi64(__m256i __a, int __count)
{
  return __builtin_ia32_psrlqi256(__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srl_epi64(__m256i __a, __m128i __count)
{
  return __builtin_ia32_psrlq256(__a, __count);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)((__v32qi)__a - (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)((__v16hi)__a - (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)((__v8si)__a - (__v8si)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi64(__m256i __a, __m256i __b)
{
  return __a - __b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_psubsb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_psubsw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epu8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_psubusb256((__v32qi)__a, (__v32qi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epu16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_psubusw256((__v16hi)__a, (__v16hi)__b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v32qi)__a, (__v32qi)__b, 8, 32+8, 9, 32+9, 10, 32+10, 11, 32+11, 12, 32+12, 13, 32+13, 14, 32+14, 15, 32+15, 24, 32+24, 25, 32+25, 26, 32+26, 27, 32+27, 28, 32+28, 29, 32+29, 30, 32+30, 31, 32+31);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v16hi)__a, (__v16hi)__b, 4, 16+4, 5, 16+5, 6, 16+6, 7, 16+7, 12, 16+12, 13, 16+13, 14, 16+14, 15, 16+15);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v8si)__a, (__v8si)__b, 2, 8+2, 3, 8+3, 6, 8+6, 7, 8+7);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpackhi_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector(__a, __b, 1, 4+1, 3, 4+3);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_epi8(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v32qi)__a, (__v32qi)__b, 0, 32+0, 1, 32+1, 2, 32+2, 3, 32+3, 4, 32+4, 5, 32+5, 6, 32+6, 7, 32+7, 16, 32+16, 17, 32+17, 18, 32+18, 19, 32+19, 20, 32+20, 21, 32+21, 22, 32+22, 23, 32+23);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_epi16(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v16hi)__a, (__v16hi)__b, 0, 16+0, 1, 16+1, 2, 16+2, 3, 16+3, 8, 16+8, 9, 16+9, 10, 16+10, 11, 16+11);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector((__v8si)__a, (__v8si)__b, 0, 8+0, 1, 8+1, 4, 8+4, 5, 8+5);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_unpacklo_epi64(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_shufflevector(__a, __b, 0, 4+0, 2, 4+2);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_xor_si256(__m256i __a, __m256i __b)
{
  return __a ^ __b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_stream_load_si256(__m256i *__V)
{
  return (__m256i)__builtin_ia32_movntdqa256((__v4di *)__V);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_broadcastss_ps(__m128 __X)
{
  return (__m128)__builtin_ia32_vbroadcastss_ps((__v4sf)__X);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_broadcastss_ps(__m128 __X)
{
  return (__m256)__builtin_ia32_vbroadcastss_ps256((__v4sf)__X);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_broadcastsd_pd(__m128d __X)
{
  return (__m256d)__builtin_ia32_vbroadcastsd_pd256((__v2df)__X);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_broadcastsi128_si256(__m128i __X)
{
  return (__m256i)__builtin_ia32_vbroadcastsi256(__X);
}

#define _mm_blend_epi32(V1, V2, M) __extension__ ({ \
  __m128i __V1 = (V1); \
  __m128i __V2 = (V2); \
  (__m128i)__builtin_shufflevector((__v4si)__V1, (__v4si)__V2, \
                                   (((M) & 0x01) ? 4 : 0), \
                                   (((M) & 0x02) ? 5 : 1), \
                                   (((M) & 0x04) ? 6 : 2), \
                                   (((M) & 0x08) ? 7 : 3)); })

#define _mm256_blend_epi32(V1, V2, M) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m256i __V2 = (V2); \
  (__m256i)__builtin_shufflevector((__v8si)__V1, (__v8si)__V2, \
                                   (((M) & 0x01) ?  8 : 0), \
                                   (((M) & 0x02) ?  9 : 1), \
                                   (((M) & 0x04) ? 10 : 2), \
                                   (((M) & 0x08) ? 11 : 3), \
                                   (((M) & 0x10) ? 12 : 4), \
                                   (((M) & 0x20) ? 13 : 5), \
                                   (((M) & 0x40) ? 14 : 6), \
                                   (((M) & 0x80) ? 15 : 7)); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_broadcastb_epi8(__m128i __X)
{
  return (__m256i)__builtin_ia32_pbroadcastb256((__v16qi)__X);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_broadcastw_epi16(__m128i __X)
{
  return (__m256i)__builtin_ia32_pbroadcastw256((__v8hi)__X);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_broadcastd_epi32(__m128i __X)
{
  return (__m256i)__builtin_ia32_pbroadcastd256((__v4si)__X);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_broadcastq_epi64(__m128i __X)
{
  return (__m256i)__builtin_ia32_pbroadcastq256(__X);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_broadcastb_epi8(__m128i __X)
{
  return (__m128i)__builtin_ia32_pbroadcastb128((__v16qi)__X);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_broadcastw_epi16(__m128i __X)
{
  return (__m128i)__builtin_ia32_pbroadcastw128((__v8hi)__X);
}


static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_broadcastd_epi32(__m128i __X)
{
  return (__m128i)__builtin_ia32_pbroadcastd128((__v4si)__X);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_broadcastq_epi64(__m128i __X)
{
  return (__m128i)__builtin_ia32_pbroadcastq128(__X);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_permutevar8x32_epi32(__m256i __a, __m256i __b)
{
  return (__m256i)__builtin_ia32_permvarsi256((__v8si)__a, (__v8si)__b);
}

#define _mm256_permute4x64_pd(V, M) __extension__ ({ \
  __m256d __V = (V); \
  (__m256d)__builtin_shufflevector((__v4df)__V, (__v4df) _mm256_setzero_pd(), \
                                   (M) & 0x3, ((M) & 0xc) >> 2, \
                                   ((M) & 0x30) >> 4, ((M) & 0xc0) >> 6); })

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_permutevar8x32_ps(__m256 __a, __m256 __b)
{
  return (__m256)__builtin_ia32_permvarsf256((__v8sf)__a, (__v8sf)__b);
}

#define _mm256_permute4x64_epi64(V, M) __extension__ ({ \
  __m256i __V = (V); \
  (__m256i)__builtin_shufflevector((__v4di)__V, (__v4di) _mm256_setzero_si256(), \
                                   (M) & 0x3, ((M) & 0xc) >> 2, \
                                   ((M) & 0x30) >> 4, ((M) & 0xc0) >> 6); })

#define _mm256_permute2x128_si256(V1, V2, M) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m256i __V2 = (V2); \
  (__m256i)__builtin_ia32_permti256(__V1, __V2, (M)); })

#define _mm256_extracti128_si256(A, O) __extension__ ({ \
  __m256i __A = (A); \
  (__m128i)__builtin_ia32_extract128i256(__A, (O)); })

#define _mm256_inserti128_si256(V1, V2, O) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m128i __V2 = (V2); \
  (__m256i)__builtin_ia32_insert128i256(__V1, __V2, (O)); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskload_epi32(int const *__X, __m256i __M)
{
  return (__m256i)__builtin_ia32_maskloadd256((const __v8si *)__X, (__v8si)__M);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskload_epi64(long long const *__X, __m256i __M)
{
  return (__m256i)__builtin_ia32_maskloadq256((const __v4di *)__X, __M);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskload_epi32(int const *__X, __m128i __M)
{
  return (__m128i)__builtin_ia32_maskloadd((const __v4si *)__X, (__v4si)__M);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskload_epi64(long long const *__X, __m128i __M)
{
  return (__m128i)__builtin_ia32_maskloadq((const __v2di *)__X, (__v2di)__M);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm256_maskstore_epi32(int *__X, __m256i __M, __m256i __Y)
{
  __builtin_ia32_maskstored256((__v8si *)__X, (__v8si)__M, (__v8si)__Y);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm256_maskstore_epi64(long long *__X, __m256i __M, __m256i __Y)
{
  __builtin_ia32_maskstoreq256((__v4di *)__X, __M, __Y);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_maskstore_epi32(int *__X, __m128i __M, __m128i __Y)
{
  __builtin_ia32_maskstored((__v4si *)__X, (__v4si)__M, (__v4si)__Y);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_maskstore_epi64(long long *__X, __m128i __M, __m128i __Y)
{
  __builtin_ia32_maskstoreq(( __v2di *)__X, __M, __Y);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sllv_epi32(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psllv8si((__v8si)__X, (__v8si)__Y);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sllv_epi32(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psllv4si((__v4si)__X, (__v4si)__Y);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sllv_epi64(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psllv4di(__X, __Y);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sllv_epi64(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psllv2di(__X, __Y);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srav_epi32(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psrav8si((__v8si)__X, (__v8si)__Y);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srav_epi32(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psrav4si((__v4si)__X, (__v4si)__Y);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srlv_epi32(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psrlv8si((__v8si)__X, (__v8si)__Y);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srlv_epi32(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psrlv4si((__v4si)__X, (__v4si)__Y);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_srlv_epi64(__m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_psrlv4di(__X, __Y);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_srlv_epi64(__m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_psrlv2di(__X, __Y);
}

#define _mm_mask_i32gather_pd(a, m, i, mask, s) __extension__ ({ \
  __m128d __a = (a); \
  double const *__m = (m); \
  __m128i __i = (i); \
  __m128d __mask = (mask); \
  (__m128d)__builtin_ia32_gatherd_pd((__v2df)__a, (const __v2df *)__m, \
             (__v4si)__i, (__v2df)__mask, (s)); })

#define _mm256_mask_i32gather_pd(a, m, i, mask, s) __extension__ ({ \
  __m256d __a = (a); \
  double const *__m = (m); \
  __m128i __i = (i); \
  __m256d __mask = (mask); \
  (__m256d)__builtin_ia32_gatherd_pd256((__v4df)__a, (const __v4df *)__m, \
             (__v4si)__i, (__v4df)__mask, (s)); })

#define _mm_mask_i64gather_pd(a, m, i, mask, s) __extension__ ({ \
  __m128d __a = (a); \
  double const *__m = (m); \
  __m128i __i = (i); \
  __m128d __mask = (mask); \
  (__m128d)__builtin_ia32_gatherq_pd((__v2df)__a, (const __v2df *)__m, \
             (__v2di)__i, (__v2df)__mask, (s)); })

#define _mm256_mask_i64gather_pd(a, m, i, mask, s) __extension__ ({ \
  __m256d __a = (a); \
  double const *__m = (m); \
  __m256i __i = (i); \
  __m256d __mask = (mask); \
  (__m256d)__builtin_ia32_gatherq_pd256((__v4df)__a, (const __v4df *)__m, \
             (__v4di)__i, (__v4df)__mask, (s)); })

#define _mm_mask_i32gather_ps(a, m, i, mask, s) __extension__ ({ \
  __m128 __a = (a); \
  float const *__m = (m); \
  __m128i __i = (i); \
  __m128 __mask = (mask); \
  (__m128)__builtin_ia32_gatherd_ps((__v4sf)__a, (const __v4sf *)__m, \
            (__v4si)__i, (__v4sf)__mask, (s)); })

#define _mm256_mask_i32gather_ps(a, m, i, mask, s) __extension__ ({ \
  __m256 __a = (a); \
  float const *__m = (m); \
  __m256i __i = (i); \
  __m256 __mask = (mask); \
  (__m256)__builtin_ia32_gatherd_ps256((__v8sf)__a, (const __v8sf *)__m, \
            (__v8si)__i, (__v8sf)__mask, (s)); })

#define _mm_mask_i64gather_ps(a, m, i, mask, s) __extension__ ({ \
  __m128 __a = (a); \
  float const *__m = (m); \
  __m128i __i = (i); \
  __m128 __mask = (mask); \
  (__m128)__builtin_ia32_gatherq_ps((__v4sf)__a, (const __v4sf *)__m, \
            (__v2di)__i, (__v4sf)__mask, (s)); })

#define _mm256_mask_i64gather_ps(a, m, i, mask, s) __extension__ ({ \
  __m128 __a = (a); \
  float const *__m = (m); \
  __m256i __i = (i); \
  __m128 __mask = (mask); \
  (__m128)__builtin_ia32_gatherq_ps256((__v4sf)__a, (const __v4sf *)__m, \
            (__v4di)__i, (__v4sf)__mask, (s)); })

#define _mm_mask_i32gather_epi32(a, m, i, mask, s) __extension__ ({ \
  __m128i __a = (a); \
  int const *__m = (m); \
  __m128i __i = (i); \
  __m128i __mask = (mask); \
  (__m128i)__builtin_ia32_gatherd_d((__v4si)__a, (const __v4si *)__m, \
            (__v4si)__i, (__v4si)__mask, (s)); })

#define _mm256_mask_i32gather_epi32(a, m, i, mask, s) __extension__ ({ \
  __m256i __a = (a); \
  int const *__m = (m); \
  __m256i __i = (i); \
  __m256i __mask = (mask); \
  (__m256i)__builtin_ia32_gatherd_d256((__v8si)__a, (const __v8si *)__m, \
            (__v8si)__i, (__v8si)__mask, (s)); })

#define _mm_mask_i64gather_epi32(a, m, i, mask, s) __extension__ ({ \
  __m128i __a = (a); \
  int const *__m = (m); \
  __m128i __i = (i); \
  __m128i __mask = (mask); \
  (__m128i)__builtin_ia32_gatherq_d((__v4si)__a, (const __v4si *)__m, \
            (__v2di)__i, (__v4si)__mask, (s)); })

#define _mm256_mask_i64gather_epi32(a, m, i, mask, s) __extension__ ({ \
  __m128i __a = (a); \
  int const *__m = (m); \
  __m256i __i = (i); \
  __m128i __mask = (mask); \
  (__m128i)__builtin_ia32_gatherq_d256((__v4si)__a, (const __v4si *)__m, \
            (__v4di)__i, (__v4si)__mask, (s)); })

#define _mm_mask_i32gather_epi64(a, m, i, mask, s) __extension__ ({ \
  __m128i __a = (a); \
  long long const *__m = (m); \
  __m128i __i = (i); \
  __m128i __mask = (mask); \
  (__m128i)__builtin_ia32_gatherd_q((__v2di)__a, (const __v2di *)__m, \
             (__v4si)__i, (__v2di)__mask, (s)); })

#define _mm256_mask_i32gather_epi64(a, m, i, mask, s) __extension__ ({ \
  __m256i __a = (a); \
  long long const *__m = (m); \
  __m128i __i = (i); \
  __m256i __mask = (mask); \
  (__m256i)__builtin_ia32_gatherd_q256((__v4di)__a, (const __v4di *)__m, \
             (__v4si)__i, (__v4di)__mask, (s)); })

#define _mm_mask_i64gather_epi64(a, m, i, mask, s) __extension__ ({ \
  __m128i __a = (a); \
  long long const *__m = (m); \
  __m128i __i = (i); \
  __m128i __mask = (mask); \
  (__m128i)__builtin_ia32_gatherq_q((__v2di)__a, (const __v2di *)__m, \
             (__v2di)__i, (__v2di)__mask, (s)); })

#define _mm256_mask_i64gather_epi64(a, m, i, mask, s) __extension__ ({ \
  __m256i __a = (a); \
  long long const *__m = (m); \
  __m256i __i = (i); \
  __m256i __mask = (mask); \
  (__m256i)__builtin_ia32_gatherq_q256((__v4di)__a, (const __v4di *)__m, \
             (__v4di)__i, (__v4di)__mask, (s)); })

#define _mm_i32gather_pd(m, i, s) __extension__ ({ \
  double const *__m = (m); \
  __m128i __i = (i); \
  (__m128d)__builtin_ia32_gatherd_pd((__v2df)_mm_setzero_pd(), \
             (const __v2df *)__m, (__v4si)__i, \
             (__v2df)_mm_set1_pd((double)(long long int)-1), (s)); })

#define _mm256_i32gather_pd(m, i, s) __extension__ ({ \
  double const *__m = (m); \
  __m128i __i = (i); \
  (__m256d)__builtin_ia32_gatherd_pd256((__v4df)_mm256_setzero_pd(), \
             (const __v4df *)__m, (__v4si)__i, \
             (__v4df)_mm256_set1_pd((double)(long long int)-1), (s)); })

#define _mm_i64gather_pd(m, i, s) __extension__ ({ \
  double const *__m = (m); \
  __m128i __i = (i); \
  (__m128d)__builtin_ia32_gatherq_pd((__v2df)_mm_setzero_pd(), \
             (const __v2df *)__m, (__v2di)__i, \
             (__v2df)_mm_set1_pd((double)(long long int)-1), (s)); })

#define _mm256_i64gather_pd(m, i, s) __extension__ ({ \
  double const *__m = (m); \
  __m256i __i = (i); \
  (__m256d)__builtin_ia32_gatherq_pd256((__v4df)_mm256_setzero_pd(), \
             (const __v4df *)__m, (__v4di)__i, \
             (__v4df)_mm256_set1_pd((double)(long long int)-1), (s)); })

#define _mm_i32gather_ps(m, i, s) __extension__ ({ \
  float const *__m = (m); \
  __m128i __i = (i); \
  (__m128)__builtin_ia32_gatherd_ps((__v4sf)_mm_setzero_ps(), \
             (const __v4sf *)__m, (__v4si)__i, \
             (__v4sf)_mm_set1_ps((float)(int)-1), (s)); })

#define _mm256_i32gather_ps(m, i, s) __extension__ ({ \
  float const *__m = (m); \
  __m256i __i = (i); \
  (__m256)__builtin_ia32_gatherd_ps256((__v8sf)_mm256_setzero_ps(), \
             (const __v8sf *)__m, (__v8si)__i, \
             (__v8sf)_mm256_set1_ps((float)(int)-1), (s)); })

#define _mm_i64gather_ps(m, i, s) __extension__ ({ \
  float const *__m = (m); \
  __m128i __i = (i); \
  (__m128)__builtin_ia32_gatherq_ps((__v4sf)_mm_setzero_ps(), \
             (const __v4sf *)__m, (__v2di)__i, \
             (__v4sf)_mm_set1_ps((float)(int)-1), (s)); })

#define _mm256_i64gather_ps(m, i, s) __extension__ ({ \
  float const *__m = (m); \
  __m256i __i = (i); \
  (__m128)__builtin_ia32_gatherq_ps256((__v4sf)_mm_setzero_ps(), \
             (const __v4sf *)__m, (__v4di)__i, \
             (__v4sf)_mm_set1_ps((float)(int)-1), (s)); })

#define _mm_i32gather_epi32(m, i, s) __extension__ ({ \
  int const *__m = (m); \
  __m128i __i = (i); \
  (__m128i)__builtin_ia32_gatherd_d((__v4si)_mm_setzero_si128(), \
            (const __v4si *)__m, (__v4si)__i, \
            (__v4si)_mm_set1_epi32(-1), (s)); })

#define _mm256_i32gather_epi32(m, i, s) __extension__ ({ \
  int const *__m = (m); \
  __m256i __i = (i); \
  (__m256i)__builtin_ia32_gatherd_d256((__v8si)_mm256_setzero_si256(), \
            (const __v8si *)__m, (__v8si)__i, \
            (__v8si)_mm256_set1_epi32(-1), (s)); })

#define _mm_i64gather_epi32(m, i, s) __extension__ ({ \
  int const *__m = (m); \
  __m128i __i = (i); \
  (__m128i)__builtin_ia32_gatherq_d((__v4si)_mm_setzero_si128(), \
            (const __v4si *)__m, (__v2di)__i, \
            (__v4si)_mm_set1_epi32(-1), (s)); })

#define _mm256_i64gather_epi32(m, i, s) __extension__ ({ \
  int const *__m = (m); \
  __m256i __i = (i); \
  (__m128i)__builtin_ia32_gatherq_d256((__v4si)_mm_setzero_si128(), \
            (const __v4si *)__m, (__v4di)__i, \
            (__v4si)_mm_set1_epi32(-1), (s)); })

#define _mm_i32gather_epi64(m, i, s) __extension__ ({ \
  long long const *__m = (m); \
  __m128i __i = (i); \
  (__m128i)__builtin_ia32_gatherd_q((__v2di)_mm_setzero_si128(), \
             (const __v2di *)__m, (__v4si)__i, \
             (__v2di)_mm_set1_epi64x(-1), (s)); })

#define _mm256_i32gather_epi64(m, i, s) __extension__ ({ \
  long long const *__m = (m); \
  __m128i __i = (i); \
  (__m256i)__builtin_ia32_gatherd_q256((__v4di)_mm256_setzero_si256(), \
             (const __v4di *)__m, (__v4si)__i, \
             (__v4di)_mm256_set1_epi64x(-1), (s)); })

#define _mm_i64gather_epi64(m, i, s) __extension__ ({ \
  long long const *__m = (m); \
  __m128i __i = (i); \
  (__m128i)__builtin_ia32_gatherq_q((__v2di)_mm_setzero_si128(), \
             (const __v2di *)__m, (__v2di)__i, \
             (__v2di)_mm_set1_epi64x(-1), (s)); })

#define _mm256_i64gather_epi64(m, i, s) __extension__ ({ \
  long long const *__m = (m); \
  __m256i __i = (i); \
  (__m256i)__builtin_ia32_gatherq_q256((__v4di)_mm256_setzero_si256(), \
             (const __v4di *)__m, (__v4di)__i, \
             (__v4di)_mm256_set1_epi64x(-1), (s)); })

#endif /* __AVX2INTRIN_H */
