/*===---- __wmmintrin_aes.h - AES intrinsics -------------------------------===
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
#ifndef _WMMINTRIN_AES_H
#define _WMMINTRIN_AES_H

#include <emmintrin.h>

#if !defined (__AES__)
#  error "AES instructions not enabled"
#else

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_aesenc_si128(__m128i __V, __m128i __R)
{
  return (__m128i)__builtin_ia32_aesenc128(__V, __R);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_aesenclast_si128(__m128i __V, __m128i __R)
{
  return (__m128i)__builtin_ia32_aesenclast128(__V, __R);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_aesdec_si128(__m128i __V, __m128i __R)
{
  return (__m128i)__builtin_ia32_aesdec128(__V, __R);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_aesdeclast_si128(__m128i __V, __m128i __R)
{
  return (__m128i)__builtin_ia32_aesdeclast128(__V, __R);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_aesimc_si128(__m128i __V)
{
  return (__m128i)__builtin_ia32_aesimc128(__V);
}

#define _mm_aeskeygenassist_si128(C, R) \
  __builtin_ia32_aeskeygenassist128((C), (R))

#endif

#endif  /* _WMMINTRIN_AES_H */
