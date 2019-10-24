/*===---- pmmintrin.h - SSE3 intrinsics ------------------------------------===
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
 
#ifndef __PMMINTRIN_H
#define __PMMINTRIN_H

#ifndef __SSE3__
#error "SSE3 instruction set not enabled"
#else

#include <emmintrin.h>

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_lddqu_si128(__m128i const *__p)
{
  return (__m128i)__builtin_ia32_lddqu((char const *)__p);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_addsub_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_addsubps(__a, __b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_hadd_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_haddps(__a, __b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_hsub_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_hsubps(__a, __b);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_movehdup_ps(__m128 __a)
{
  return __builtin_shufflevector(__a, __a, 1, 1, 3, 3);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_moveldup_ps(__m128 __a)
{
  return __builtin_shufflevector(__a, __a, 0, 0, 2, 2);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_addsub_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_addsubpd(__a, __b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_hadd_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_haddpd(__a, __b);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_hsub_pd(__m128d __a, __m128d __b)
{
  return __builtin_ia32_hsubpd(__a, __b);
}

#define        _mm_loaddup_pd(dp)        _mm_load1_pd(dp)

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_movedup_pd(__m128d __a)
{
  return __builtin_shufflevector(__a, __a, 0, 0);
}

#define _MM_DENORMALS_ZERO_ON   (0x0040)
#define _MM_DENORMALS_ZERO_OFF  (0x0000)

#define _MM_DENORMALS_ZERO_MASK (0x0040)

#define _MM_GET_DENORMALS_ZERO_MODE() (_mm_getcsr() & _MM_DENORMALS_ZERO_MASK)
#define _MM_SET_DENORMALS_ZERO_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_DENORMALS_ZERO_MASK) | (x)))

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_monitor(void const *__p, unsigned __extensions, unsigned __hints)
{
  __builtin_ia32_monitor((void *)__p, __extensions, __hints);
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_mwait(unsigned __extensions, unsigned __hints)
{
  __builtin_ia32_mwait(__extensions, __hints);
}

#endif /* __SSE3__ */

#endif /* __PMMINTRIN_H */
