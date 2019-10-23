/*===---- fma4intrin.h - FMA4 intrinsics -----------------------------------===
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
#error "Never use <fmaintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __FMAINTRIN_H
#define __FMAINTRIN_H

#ifndef __FMA__
# error "FMA instruction set is not enabled"
#else

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fmadd_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmaddps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fmadd_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmaddpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fmadd_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmaddss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fmadd_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmaddsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fmsub_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmsubps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fmsub_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmsubpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fmsub_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmsubss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fmsub_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmsubsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fnmadd_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmaddps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fnmadd_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmaddpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fnmadd_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmaddss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fnmadd_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmaddsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fnmsub_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmsubps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fnmsub_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmsubpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fnmsub_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmsubss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fnmsub_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmsubsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fmaddsub_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmaddsubps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fmaddsub_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmaddsubpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_fmsubadd_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmsubaddps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_fmsubadd_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmsubaddpd(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_fmadd_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmaddps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_fmadd_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmaddpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_fmsub_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmsubps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_fmsub_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmsubpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_fnmadd_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfnmaddps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_fnmadd_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfnmaddpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_fnmsub_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfnmsubps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_fnmsub_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfnmsubpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_fmaddsub_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmaddsubps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_fmaddsub_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmaddsubpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_fmsubadd_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmsubaddps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_fmsubadd_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmsubaddpd256(__A, __B, __C);
}

#endif /* __FMA__ */

#endif /* __FMAINTRIN_H */
