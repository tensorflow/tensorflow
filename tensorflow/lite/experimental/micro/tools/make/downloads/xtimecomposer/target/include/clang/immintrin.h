/*===---- immintrin.h - Intel intrinsics -----------------------------------===
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
#define __IMMINTRIN_H

#ifdef __MMX__
#include <mmintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __SSE3__
#include <pmmintrin.h>
#endif

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#if defined (__SSE4_2__) || defined (__SSE4_1__)
#include <smmintrin.h>
#endif

#if defined (__AES__) || defined (__PCLMUL__)
#include <wmmintrin.h>
#endif

#ifdef __AVX__
#include <avxintrin.h>
#endif

#ifdef __AVX2__
#include <avx2intrin.h>
#endif

#ifdef __BMI__
#include <bmiintrin.h>
#endif

#ifdef __BMI2__
#include <bmi2intrin.h>
#endif

#ifdef __LZCNT__
#include <lzcntintrin.h>
#endif

#ifdef __FMA__
#include <fmaintrin.h>
#endif

#ifdef __AVX512F__
#include <avx512fintrin.h>
#endif

#ifdef __AVX512ER__
#include <avx512erintrin.h>
#endif

#ifdef __RDRND__
static __inline__ int __attribute__((__always_inline__, __nodebug__))
_rdrand16_step(unsigned short *__p)
{
  return __builtin_ia32_rdrand16_step(__p);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_rdrand32_step(unsigned int *__p)
{
  return __builtin_ia32_rdrand32_step(__p);
}

#ifdef __x86_64__
static __inline__ int __attribute__((__always_inline__, __nodebug__))
_rdrand64_step(unsigned long long *__p)
{
  return __builtin_ia32_rdrand64_step(__p);
}
#endif
#endif /* __RDRND__ */

#ifdef __RTM__
#include <rtmintrin.h>
#endif

/* FIXME: check __HLE__ as well when HLE is supported. */
#if defined (__RTM__)
static __inline__ int __attribute__((__always_inline__, __nodebug__))
_xtest(void)
{
  return __builtin_ia32_xtest();
}
#endif

#ifdef __SHA__
#include <shaintrin.h>
#endif

/* Some intrinsics inside adxintrin.h are available only if __ADX__ defined,
 * whereas others are also available if __ADX__ undefined */
#include <adxintrin.h>

#endif /* __IMMINTRIN_H */
