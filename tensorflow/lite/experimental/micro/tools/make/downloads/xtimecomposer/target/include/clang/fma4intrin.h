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

#ifndef __X86INTRIN_H
#error "Never use <fma4intrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __FMA4INTRIN_H
#define __FMA4INTRIN_H

#ifndef __FMA4__
# error "FMA4 instruction set is not enabled"
#else

#include <pmmintrin.h>

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_macc_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmaddps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_macc_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmaddpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_macc_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmaddss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_macc_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmaddsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_msub_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmsubps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_msub_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmsubpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_msub_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmsubss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_msub_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmsubsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_nmacc_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmaddps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_nmacc_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmaddpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_nmacc_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmaddss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_nmacc_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmaddsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_nmsub_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmsubps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_nmsub_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmsubpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_nmsub_ss(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfnmsubss(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_nmsub_sd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfnmsubsd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_maddsub_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmaddsubps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_maddsub_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmaddsubpd(__A, __B, __C);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_msubadd_ps(__m128 __A, __m128 __B, __m128 __C)
{
  return (__m128)__builtin_ia32_vfmsubaddps(__A, __B, __C);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_msubadd_pd(__m128d __A, __m128d __B, __m128d __C)
{
  return (__m128d)__builtin_ia32_vfmsubaddpd(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_macc_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmaddps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_macc_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmaddpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_msub_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmsubps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_msub_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmsubpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_nmacc_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfnmaddps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_nmacc_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfnmaddpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_nmsub_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfnmsubps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_nmsub_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfnmsubpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_maddsub_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmaddsubps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_maddsub_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmaddsubpd256(__A, __B, __C);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_msubadd_ps(__m256 __A, __m256 __B, __m256 __C)
{
  return (__m256)__builtin_ia32_vfmsubaddps256(__A, __B, __C);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_msubadd_pd(__m256d __A, __m256d __B, __m256d __C)
{
  return (__m256d)__builtin_ia32_vfmsubaddpd256(__A, __B, __C);
}

#endif /* __FMA4__ */

#endif /* __FMA4INTRIN_H */
