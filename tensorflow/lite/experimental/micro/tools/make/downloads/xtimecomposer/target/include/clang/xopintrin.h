/*===---- xopintrin.h - XOP intrinsics -------------------------------------===
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
#error "Never use <xopintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __XOPINTRIN_H
#define __XOPINTRIN_H

#ifndef __XOP__
# error "XOP instruction set is not enabled"
#else

#include <fma4intrin.h>

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccs_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssww((__v8hi)__A, (__v8hi)__B, (__v8hi)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macc_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsww((__v8hi)__A, (__v8hi)__B, (__v8hi)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccsd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccs_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssdd((__v4si)__A, (__v4si)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macc_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsdd((__v4si)__A, (__v4si)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccslo_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssdql((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macclo_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsdql((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccshi_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssdqh((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macchi_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsdqh((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maddsd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmadcsswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maddd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmadcswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddw_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddbw((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddbd((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddbq((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epi16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddwd((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epi16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddwq((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epi32(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphadddq((__v4si)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddw_epu8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddubw((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epu8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddubd((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epu8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddubq((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epu16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphadduwd((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epu16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphadduwq((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epu32(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddudq((__v4si)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsubw_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphsubbw((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsubd_epi16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphsubwd((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsubq_epi32(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphsubdq((__v4si)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmov_si128(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpcmov(__A, __B, __C);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmov_si256(__m256i __A, __m256i __B, __m256i __C)
{
  return (__m256i)__builtin_ia32_vpcmov_256(__A, __B, __C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_perm_epi8(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpperm((__v16qi)__A, (__v16qi)__B, (__v16qi)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi8(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vprotb((__v16qi)__A, (__v16qi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi16(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vprotw((__v8hi)__A, (__v8hi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi32(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vprotd((__v4si)__A, (__v4si)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi64(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vprotq((__v2di)__A, (__v2di)__B);
}

#define _mm_roti_epi8(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotbi((__v16qi)__A, (N)); })

#define _mm_roti_epi16(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotwi((__v8hi)__A, (N)); })

#define _mm_roti_epi32(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotdi((__v4si)__A, (N)); })

#define _mm_roti_epi64(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotqi((__v2di)__A, (N)); })

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_shl_epi8(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshlb((__v16qi)__A, (__v16qi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_shl_epi16(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshlw((__v8hi)__A, (__v8hi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_shl_epi32(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshld((__v4si)__A, (__v4si)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_shl_epi64(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshlq((__v2di)__A, (__v2di)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sha_epi8(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshab((__v16qi)__A, (__v16qi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sha_epi16(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshaw((__v8hi)__A, (__v8hi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sha_epi32(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshad((__v4si)__A, (__v4si)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_sha_epi64(__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_vpshaq((__v2di)__A, (__v2di)__B);
}

#define _mm_com_epu8(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomub((__v16qi)__A, (__v16qi)__B, (N)); })

#define _mm_com_epu16(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomuw((__v8hi)__A, (__v8hi)__B, (N)); })

#define _mm_com_epu32(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomud((__v4si)__A, (__v4si)__B, (N)); })

#define _mm_com_epu64(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomuq((__v2di)__A, (__v2di)__B, (N)); })

#define _mm_com_epi8(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomb((__v16qi)__A, (__v16qi)__B, (N)); })

#define _mm_com_epi16(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomw((__v8hi)__A, (__v8hi)__B, (N)); })

#define _mm_com_epi32(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomd((__v4si)__A, (__v4si)__B, (N)); })

#define _mm_com_epi64(A, B, N) __extension__ ({ \
  __m128i __A = (A); \
  __m128i __B = (B); \
  (__m128i)__builtin_ia32_vpcomq((__v2di)__A, (__v2di)__B, (N)); })

#define _MM_PCOMCTRL_LT    0
#define _MM_PCOMCTRL_LE    1
#define _MM_PCOMCTRL_GT    2
#define _MM_PCOMCTRL_GE    3
#define _MM_PCOMCTRL_EQ    4
#define _MM_PCOMCTRL_NEQ   5
#define _MM_PCOMCTRL_FALSE 6
#define _MM_PCOMCTRL_TRUE  7

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epu8(__m128i __A, __m128i __B)
{
  return _mm_com_epu8(__A, __B, _MM_PCOMCTRL_TRUE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epu16(__m128i __A, __m128i __B)
{
  return _mm_com_epu16(__A, __B, _MM_PCOMCTRL_TRUE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epu32(__m128i __A, __m128i __B)
{
  return _mm_com_epu32(__A, __B, _MM_PCOMCTRL_TRUE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epu64(__m128i __A, __m128i __B)
{
  return _mm_com_epu64(__A, __B, _MM_PCOMCTRL_TRUE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epi8(__m128i __A, __m128i __B)
{
  return _mm_com_epi8(__A, __B, _MM_PCOMCTRL_TRUE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epi16(__m128i __A, __m128i __B)
{
  return _mm_com_epi16(__A, __B, _MM_PCOMCTRL_TRUE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epi32(__m128i __A, __m128i __B)
{
  return _mm_com_epi32(__A, __B, _MM_PCOMCTRL_TRUE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comlt_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_LT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comle_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_LE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comgt_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_GT);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comge_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_GE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comeq_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_EQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comneq_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_NEQ);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comfalse_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_FALSE);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_comtrue_epi64(__m128i __A, __m128i __B)
{
  return _mm_com_epi64(__A, __B, _MM_PCOMCTRL_TRUE);
}

#define _mm_permute2_pd(X, Y, C, I) __extension__ ({ \
  __m128d __X = (X); \
  __m128d __Y = (Y); \
  __m128i __C = (C); \
  (__m128d)__builtin_ia32_vpermil2pd((__v2df)__X, (__v2df)__Y, \
                                     (__v2di)__C, (I)); })

#define _mm256_permute2_pd(X, Y, C, I) __extension__ ({ \
  __m256d __X = (X); \
  __m256d __Y = (Y); \
  __m256i __C = (C); \
  (__m256d)__builtin_ia32_vpermil2pd256((__v4df)__X, (__v4df)__Y, \
                                        (__v4di)__C, (I)); })

#define _mm_permute2_ps(X, Y, C, I) __extension__ ({ \
  __m128 __X = (X); \
  __m128 __Y = (Y); \
  __m128i __C = (C); \
  (__m128)__builtin_ia32_vpermil2ps((__v4sf)__X, (__v4sf)__Y, \
                                    (__v4si)__C, (I)); })

#define _mm256_permute2_ps(X, Y, C, I) __extension__ ({ \
  __m256 __X = (X); \
  __m256 __Y = (Y); \
  __m256i __C = (C); \
  (__m256)__builtin_ia32_vpermil2ps256((__v8sf)__X, (__v8sf)__Y, \
                                       (__v8si)__C, (I)); })

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_frcz_ss(__m128 __A)
{
  return (__m128)__builtin_ia32_vfrczss((__v4sf)__A);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_frcz_sd(__m128d __A)
{
  return (__m128d)__builtin_ia32_vfrczsd((__v2df)__A);
}

static __inline__ __m128 __attribute__((__always_inline__, __nodebug__))
_mm_frcz_ps(__m128 __A)
{
  return (__m128)__builtin_ia32_vfrczps((__v4sf)__A);
}

static __inline__ __m128d __attribute__((__always_inline__, __nodebug__))
_mm_frcz_pd(__m128d __A)
{
  return (__m128d)__builtin_ia32_vfrczpd((__v2df)__A);
}

static __inline__ __m256 __attribute__((__always_inline__, __nodebug__))
_mm256_frcz_ps(__m256 __A)
{
  return (__m256)__builtin_ia32_vfrczps256((__v8sf)__A);
}

static __inline__ __m256d __attribute__((__always_inline__, __nodebug__))
_mm256_frcz_pd(__m256d __A)
{
  return (__m256d)__builtin_ia32_vfrczpd256((__v4df)__A);
}

#endif /* __XOP__ */

#endif /* __XOPINTRIN_H */
