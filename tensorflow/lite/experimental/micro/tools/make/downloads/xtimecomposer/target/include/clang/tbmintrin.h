/*===---- tbmintrin.h - TBM intrinsics -------------------------------------===
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

#ifndef __TBM__
#error "TBM instruction set is not enabled"
#endif

#ifndef __X86INTRIN_H
#error "Never use <tbmintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __TBMINTRIN_H
#define __TBMINTRIN_H

#define __bextri_u32(a, b) (__builtin_ia32_bextri_u32((a), (b)))

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__blcfill_u32(unsigned int a)
{
  return a & (a + 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__blci_u32(unsigned int a)
{
  return a | ~(a + 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__blcic_u32(unsigned int a)
{
  return ~a & (a + 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__blcmsk_u32(unsigned int a)
{
  return a ^ (a + 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__blcs_u32(unsigned int a)
{
  return a | (a + 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__blsfill_u32(unsigned int a)
{
  return a | (a - 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__blsic_u32(unsigned int a)
{
  return ~a | (a - 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__t1mskc_u32(unsigned int a)
{
  return ~a | (a + 1);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__tzmsk_u32(unsigned int a)
{
  return ~a & (a - 1);
}

#ifdef __x86_64__
#define __bextri_u64(a, b) (__builtin_ia32_bextri_u64((a), (int)(b)))

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__blcfill_u64(unsigned long long a)
{
  return a & (a + 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__blci_u64(unsigned long long a)
{
  return a | ~(a + 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__blcic_u64(unsigned long long a)
{
  return ~a & (a + 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__blcmsk_u64(unsigned long long a)
{
  return a ^ (a + 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__blcs_u64(unsigned long long a)
{
  return a | (a + 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__blsfill_u64(unsigned long long a)
{
  return a | (a - 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__blsic_u64(unsigned long long a)
{
  return ~a | (a - 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__t1mskc_u64(unsigned long long a)
{
  return ~a | (a + 1);
}

static __inline__ unsigned long long __attribute__((__always_inline__,
                                                    __nodebug__))
__tzmsk_u64(unsigned long long a)
{
  return ~a & (a - 1);
}
#endif

#endif /* __TBMINTRIN_H */
