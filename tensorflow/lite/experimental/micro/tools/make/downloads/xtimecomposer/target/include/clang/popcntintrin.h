/*===---- popcntintrin.h - POPCNT intrinsics -------------------------------===
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

#ifndef __POPCNT__
#error "POPCNT instruction set not enabled"
#endif

#ifndef _POPCNTINTRIN_H
#define _POPCNTINTRIN_H

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm_popcnt_u32(unsigned int __A)
{
  return __builtin_popcount(__A);
}

#ifdef __x86_64__
static __inline__ long long __attribute__((__always_inline__, __nodebug__))
_mm_popcnt_u64(unsigned long long __A)
{
  return __builtin_popcountll(__A);
}
#endif /* __x86_64__ */

#endif /* _POPCNTINTRIN_H */
