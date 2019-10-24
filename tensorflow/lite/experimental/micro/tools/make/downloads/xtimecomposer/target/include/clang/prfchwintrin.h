/*===---- prfchwintrin.h - PREFETCHW intrinsic -----------------------------===
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

#if !defined(__X86INTRIN_H) && !defined(_MM3DNOW_H_INCLUDED)
#error "Never use <prfchwintrin.h> directly; include <x86intrin.h> or <mm3dnow.h> instead."
#endif

#ifndef __PRFCHWINTRIN_H
#define __PRFCHWINTRIN_H

#if defined(__PRFCHW__) || defined(__3dNOW__)
static __inline__ void __attribute__((__always_inline__, __nodebug__))
_m_prefetchw(void *__P)
{
  __builtin_prefetch (__P, 1, 3 /* _MM_HINT_T0 */);
}
#endif

#endif /* __PRFCHWINTRIN_H */
