/*===---- f16cintrin.h - F16C intrinsics -----------------------------------===
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

#if !defined __X86INTRIN_H && !defined __IMMINTRIN_H
#error "Never use <f16cintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __F16C__
# error "F16C instruction is not enabled"
#endif /* __F16C__ */

#ifndef __F16CINTRIN_H
#define __F16CINTRIN_H

typedef float __v8sf __attribute__ ((__vector_size__ (32)));
typedef float __m256 __attribute__ ((__vector_size__ (32)));

#define _mm_cvtps_ph(a, imm) __extension__ ({ \
  __m128 __a = (a); \
 (__m128i)__builtin_ia32_vcvtps2ph((__v4sf)__a, (imm)); })

#define _mm256_cvtps_ph(a, imm) __extension__ ({ \
  __m256 __a = (a); \
 (__m128i)__builtin_ia32_vcvtps2ph256((__v8sf)__a, (imm)); })

static __inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtph_ps(__m128i __a)
{
  return (__m128)__builtin_ia32_vcvtph2ps((__v8hi)__a);
}

static __inline __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_cvtph_ps(__m128i __a)
{
  return (__m256)__builtin_ia32_vcvtph2ps256((__v8hi)__a);
}

#endif /* __F16CINTRIN_H */
