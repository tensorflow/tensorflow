/*===---- x86intrin.h - X86 intrinsics -------------------------------------===
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
#define __X86INTRIN_H

#include <ia32intrin.h>

#include <immintrin.h>

#ifdef __3dNOW__
#include <mm3dnow.h>
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

#ifdef __POPCNT__
#include <popcntintrin.h>
#endif

#ifdef __RDSEED__
#include <rdseedintrin.h>
#endif

#ifdef __PRFCHW__
#include <prfchwintrin.h>
#endif

#ifdef __SSE4A__
#include <ammintrin.h>
#endif

#ifdef __FMA4__
#include <fma4intrin.h>
#endif

#ifdef __XOP__
#include <xopintrin.h>
#endif

#ifdef __TBM__
#include <tbmintrin.h>
#endif

#ifdef __F16C__
#include <f16cintrin.h>
#endif

/* FIXME: LWP */

#endif /* __X86INTRIN_H */
