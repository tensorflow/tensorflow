/*===---- altivec.h - Standard header for type generic math ---------------===*\
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
\*===----------------------------------------------------------------------===*/

#ifndef __ALTIVEC_H
#define __ALTIVEC_H

#ifndef __ALTIVEC__
#error "AltiVec support not enabled"
#endif

/* constants for mapping CR6 bits to predicate result. */

#define __CR6_EQ     0
#define __CR6_EQ_REV 1
#define __CR6_LT     2
#define __CR6_LT_REV 3

#define __ATTRS_o_ai __attribute__((__overloadable__, __always_inline__))

static vector signed char __ATTRS_o_ai
vec_perm(vector signed char __a, vector signed char __b, vector unsigned char __c);

static vector unsigned char __ATTRS_o_ai
vec_perm(vector unsigned char __a,
         vector unsigned char __b,
         vector unsigned char __c);

static vector bool char __ATTRS_o_ai
vec_perm(vector bool char __a, vector bool char __b, vector unsigned char __c);

static vector short __ATTRS_o_ai
vec_perm(vector short __a, vector short __b, vector unsigned char __c);

static vector unsigned short __ATTRS_o_ai
vec_perm(vector unsigned short __a,
         vector unsigned short __b,
         vector unsigned char __c);

static vector bool short __ATTRS_o_ai
vec_perm(vector bool short __a, vector bool short __b, vector unsigned char __c);

static vector pixel __ATTRS_o_ai
vec_perm(vector pixel __a, vector pixel __b, vector unsigned char __c);

static vector int __ATTRS_o_ai
vec_perm(vector int __a, vector int __b, vector unsigned char __c);

static vector unsigned int __ATTRS_o_ai
vec_perm(vector unsigned int __a, vector unsigned int __b, vector unsigned char __c);

static vector bool int __ATTRS_o_ai
vec_perm(vector bool int __a, vector bool int __b, vector unsigned char __c);

static vector float __ATTRS_o_ai
vec_perm(vector float __a, vector float __b, vector unsigned char __c);

static vector unsigned char __ATTRS_o_ai
vec_xor(vector unsigned char __a, vector unsigned char __b);

/* vec_abs */

#define __builtin_altivec_abs_v16qi vec_abs
#define __builtin_altivec_abs_v8hi  vec_abs
#define __builtin_altivec_abs_v4si  vec_abs

static vector signed char __ATTRS_o_ai
vec_abs(vector signed char __a)
{
  return __builtin_altivec_vmaxsb(__a, -__a);
}

static vector signed short __ATTRS_o_ai
vec_abs(vector signed short __a)
{
  return __builtin_altivec_vmaxsh(__a, -__a);
}

static vector signed int __ATTRS_o_ai
vec_abs(vector signed int __a)
{
  return __builtin_altivec_vmaxsw(__a, -__a);
}

static vector float __ATTRS_o_ai
vec_abs(vector float __a)
{
  vector unsigned int __res = (vector unsigned int)__a
                            & (vector unsigned int)(0x7FFFFFFF);
  return (vector float)__res;
}

/* vec_abss */

#define __builtin_altivec_abss_v16qi vec_abss
#define __builtin_altivec_abss_v8hi  vec_abss
#define __builtin_altivec_abss_v4si  vec_abss

static vector signed char __ATTRS_o_ai
vec_abss(vector signed char __a)
{
  return __builtin_altivec_vmaxsb
           (__a, __builtin_altivec_vsubsbs((vector signed char)(0), __a));
}

static vector signed short __ATTRS_o_ai
vec_abss(vector signed short __a)
{
  return __builtin_altivec_vmaxsh
           (__a, __builtin_altivec_vsubshs((vector signed short)(0), __a));
}

static vector signed int __ATTRS_o_ai
vec_abss(vector signed int __a)
{
  return __builtin_altivec_vmaxsw
           (__a, __builtin_altivec_vsubsws((vector signed int)(0), __a));
}

/* vec_add */

static vector signed char __ATTRS_o_ai
vec_add(vector signed char __a, vector signed char __b)
{
  return __a + __b;
}

static vector signed char __ATTRS_o_ai
vec_add(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a + __b;
}

static vector signed char __ATTRS_o_ai
vec_add(vector signed char __a, vector bool char __b)
{
  return __a + (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_add(vector unsigned char __a, vector unsigned char __b)
{
  return __a + __b;
}

static vector unsigned char __ATTRS_o_ai
vec_add(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a + __b;
}

static vector unsigned char __ATTRS_o_ai
vec_add(vector unsigned char __a, vector bool char __b)
{
  return __a + (vector unsigned char)__b;
}

static vector short __ATTRS_o_ai
vec_add(vector short __a, vector short __b)
{
  return __a + __b;
}

static vector short __ATTRS_o_ai
vec_add(vector bool short __a, vector short __b)
{
  return (vector short)__a + __b;
}

static vector short __ATTRS_o_ai
vec_add(vector short __a, vector bool short __b)
{
  return __a + (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_add(vector unsigned short __a, vector unsigned short __b)
{
  return __a + __b;
}

static vector unsigned short __ATTRS_o_ai
vec_add(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a + __b;
}

static vector unsigned short __ATTRS_o_ai
vec_add(vector unsigned short __a, vector bool short __b)
{
  return __a + (vector unsigned short)__b;
}

static vector int __ATTRS_o_ai
vec_add(vector int __a, vector int __b)
{
  return __a + __b;
}

static vector int __ATTRS_o_ai
vec_add(vector bool int __a, vector int __b)
{
  return (vector int)__a + __b;
}

static vector int __ATTRS_o_ai
vec_add(vector int __a, vector bool int __b)
{
  return __a + (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_add(vector unsigned int __a, vector unsigned int __b)
{
  return __a + __b;
}

static vector unsigned int __ATTRS_o_ai
vec_add(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a + __b;
}

static vector unsigned int __ATTRS_o_ai
vec_add(vector unsigned int __a, vector bool int __b)
{
  return __a + (vector unsigned int)__b;
}

static vector float __ATTRS_o_ai
vec_add(vector float __a, vector float __b)
{
  return __a + __b;
}

/* vec_vaddubm */

#define __builtin_altivec_vaddubm vec_vaddubm

static vector signed char __ATTRS_o_ai
vec_vaddubm(vector signed char __a, vector signed char __b)
{
  return __a + __b;
}

static vector signed char __ATTRS_o_ai
vec_vaddubm(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a + __b;
}

static vector signed char __ATTRS_o_ai
vec_vaddubm(vector signed char __a, vector bool char __b)
{
  return __a + (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubm(vector unsigned char __a, vector unsigned char __b)
{
  return __a + __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubm(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a + __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubm(vector unsigned char __a, vector bool char __b)
{
  return __a + (vector unsigned char)__b;
}

/* vec_vadduhm */

#define __builtin_altivec_vadduhm vec_vadduhm

static vector short __ATTRS_o_ai
vec_vadduhm(vector short __a, vector short __b)
{
  return __a + __b;
}

static vector short __ATTRS_o_ai
vec_vadduhm(vector bool short __a, vector short __b)
{
  return (vector short)__a + __b;
}

static vector short __ATTRS_o_ai
vec_vadduhm(vector short __a, vector bool short __b)
{
  return __a + (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhm(vector unsigned short __a, vector unsigned short __b)
{
  return __a + __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhm(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a + __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhm(vector unsigned short __a, vector bool short __b)
{
  return __a + (vector unsigned short)__b;
}

/* vec_vadduwm */

#define __builtin_altivec_vadduwm vec_vadduwm

static vector int __ATTRS_o_ai
vec_vadduwm(vector int __a, vector int __b)
{
  return __a + __b;
}

static vector int __ATTRS_o_ai
vec_vadduwm(vector bool int __a, vector int __b)
{
  return (vector int)__a + __b;
}

static vector int __ATTRS_o_ai
vec_vadduwm(vector int __a, vector bool int __b)
{
  return __a + (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vadduwm(vector unsigned int __a, vector unsigned int __b)
{
  return __a + __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vadduwm(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a + __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vadduwm(vector unsigned int __a, vector bool int __b)
{
  return __a + (vector unsigned int)__b;
}

/* vec_vaddfp */

#define __builtin_altivec_vaddfp  vec_vaddfp

static vector float __attribute__((__always_inline__))
vec_vaddfp(vector float __a, vector float __b)
{
  return __a + __b;
}

/* vec_addc */

static vector unsigned int __attribute__((__always_inline__))
vec_addc(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vaddcuw(__a, __b);
}

/* vec_vaddcuw */

static vector unsigned int __attribute__((__always_inline__))
vec_vaddcuw(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vaddcuw(__a, __b);
}

/* vec_adds */

static vector signed char __ATTRS_o_ai
vec_adds(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vaddsbs(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_adds(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vaddsbs((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_adds(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vaddsbs(__a, (vector signed char)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_adds(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vaddubs(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_adds(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vaddubs((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_adds(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vaddubs(__a, (vector unsigned char)__b);
}

static vector short __ATTRS_o_ai
vec_adds(vector short __a, vector short __b)
{
  return __builtin_altivec_vaddshs(__a, __b);
}

static vector short __ATTRS_o_ai
vec_adds(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vaddshs((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_adds(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vaddshs(__a, (vector short)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_adds(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vadduhs(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_adds(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vadduhs((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_adds(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vadduhs(__a, (vector unsigned short)__b);
}

static vector int __ATTRS_o_ai
vec_adds(vector int __a, vector int __b)
{
  return __builtin_altivec_vaddsws(__a, __b);
}

static vector int __ATTRS_o_ai
vec_adds(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vaddsws((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_adds(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vaddsws(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_adds(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vadduws(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_adds(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vadduws((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_adds(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vadduws(__a, (vector unsigned int)__b);
}

/* vec_vaddsbs */

static vector signed char __ATTRS_o_ai
vec_vaddsbs(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vaddsbs(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vaddsbs(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vaddsbs((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vaddsbs(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vaddsbs(__a, (vector signed char)__b);
}

/* vec_vaddubs */

static vector unsigned char __ATTRS_o_ai
vec_vaddubs(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vaddubs(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubs(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vaddubs((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vaddubs(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vaddubs(__a, (vector unsigned char)__b);
}

/* vec_vaddshs */

static vector short __ATTRS_o_ai
vec_vaddshs(vector short __a, vector short __b)
{
  return __builtin_altivec_vaddshs(__a, __b);
}

static vector short __ATTRS_o_ai
vec_vaddshs(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vaddshs((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_vaddshs(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vaddshs(__a, (vector short)__b);
}

/* vec_vadduhs */

static vector unsigned short __ATTRS_o_ai
vec_vadduhs(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vadduhs(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhs(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vadduhs((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vadduhs(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vadduhs(__a, (vector unsigned short)__b);
}

/* vec_vaddsws */

static vector int __ATTRS_o_ai
vec_vaddsws(vector int __a, vector int __b)
{
  return __builtin_altivec_vaddsws(__a, __b);
}

static vector int __ATTRS_o_ai
vec_vaddsws(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vaddsws((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_vaddsws(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vaddsws(__a, (vector int)__b);
}

/* vec_vadduws */

static vector unsigned int __ATTRS_o_ai
vec_vadduws(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vadduws(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vadduws(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vadduws((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vadduws(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vadduws(__a, (vector unsigned int)__b);
}

/* vec_and */

#define __builtin_altivec_vand vec_and

static vector signed char __ATTRS_o_ai
vec_and(vector signed char __a, vector signed char __b)
{
  return __a & __b;
}

static vector signed char __ATTRS_o_ai
vec_and(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a & __b;
}

static vector signed char __ATTRS_o_ai
vec_and(vector signed char __a, vector bool char __b)
{
  return __a & (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_and(vector unsigned char __a, vector unsigned char __b)
{
  return __a & __b;
}

static vector unsigned char __ATTRS_o_ai
vec_and(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a & __b;
}

static vector unsigned char __ATTRS_o_ai
vec_and(vector unsigned char __a, vector bool char __b)
{
  return __a & (vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_and(vector bool char __a, vector bool char __b)
{
  return __a & __b;
}

static vector short __ATTRS_o_ai
vec_and(vector short __a, vector short __b)
{
  return __a & __b;
}

static vector short __ATTRS_o_ai
vec_and(vector bool short __a, vector short __b)
{
  return (vector short)__a & __b;
}

static vector short __ATTRS_o_ai
vec_and(vector short __a, vector bool short __b)
{
  return __a & (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_and(vector unsigned short __a, vector unsigned short __b)
{
  return __a & __b;
}

static vector unsigned short __ATTRS_o_ai
vec_and(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a & __b;
}

static vector unsigned short __ATTRS_o_ai
vec_and(vector unsigned short __a, vector bool short __b)
{
  return __a & (vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_and(vector bool short __a, vector bool short __b)
{
  return __a & __b;
}

static vector int __ATTRS_o_ai
vec_and(vector int __a, vector int __b)
{
  return __a & __b;
}

static vector int __ATTRS_o_ai
vec_and(vector bool int __a, vector int __b)
{
  return (vector int)__a & __b;
}

static vector int __ATTRS_o_ai
vec_and(vector int __a, vector bool int __b)
{
  return __a & (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_and(vector unsigned int __a, vector unsigned int __b)
{
  return __a & __b;
}

static vector unsigned int __ATTRS_o_ai
vec_and(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a & __b;
}

static vector unsigned int __ATTRS_o_ai
vec_and(vector unsigned int __a, vector bool int __b)
{
  return __a & (vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_and(vector bool int __a, vector bool int __b)
{
  return __a & __b;
}

static vector float __ATTRS_o_ai
vec_and(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_and(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_and(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a & (vector unsigned int)__b;
  return (vector float)__res;
}

/* vec_vand */

static vector signed char __ATTRS_o_ai
vec_vand(vector signed char __a, vector signed char __b)
{
  return __a & __b;
}

static vector signed char __ATTRS_o_ai
vec_vand(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a & __b;
}

static vector signed char __ATTRS_o_ai
vec_vand(vector signed char __a, vector bool char __b)
{
  return __a & (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vand(vector unsigned char __a, vector unsigned char __b)
{
  return __a & __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vand(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a & __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vand(vector unsigned char __a, vector bool char __b)
{
  return __a & (vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_vand(vector bool char __a, vector bool char __b)
{
  return __a & __b;
}

static vector short __ATTRS_o_ai
vec_vand(vector short __a, vector short __b)
{
  return __a & __b;
}

static vector short __ATTRS_o_ai
vec_vand(vector bool short __a, vector short __b)
{
  return (vector short)__a & __b;
}

static vector short __ATTRS_o_ai
vec_vand(vector short __a, vector bool short __b)
{
  return __a & (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vand(vector unsigned short __a, vector unsigned short __b)
{
  return __a & __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vand(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a & __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vand(vector unsigned short __a, vector bool short __b)
{
  return __a & (vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_vand(vector bool short __a, vector bool short __b)
{
  return __a & __b;
}

static vector int __ATTRS_o_ai
vec_vand(vector int __a, vector int __b)
{
  return __a & __b;
}

static vector int __ATTRS_o_ai
vec_vand(vector bool int __a, vector int __b)
{
  return (vector int)__a & __b;
}

static vector int __ATTRS_o_ai
vec_vand(vector int __a, vector bool int __b)
{
  return __a & (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vand(vector unsigned int __a, vector unsigned int __b)
{
  return __a & __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vand(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a & __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vand(vector unsigned int __a, vector bool int __b)
{
  return __a & (vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_vand(vector bool int __a, vector bool int __b)
{
  return __a & __b;
}

static vector float __ATTRS_o_ai
vec_vand(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vand(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vand(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a & (vector unsigned int)__b;
  return (vector float)__res;
}

/* vec_andc */

#define __builtin_altivec_vandc vec_andc

static vector signed char __ATTRS_o_ai
vec_andc(vector signed char __a, vector signed char __b)
{
  return __a & ~__b;
}

static vector signed char __ATTRS_o_ai
vec_andc(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a & ~__b;
}

static vector signed char __ATTRS_o_ai
vec_andc(vector signed char __a, vector bool char __b)
{
  return __a & ~(vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_andc(vector unsigned char __a, vector unsigned char __b)
{
  return __a & ~__b;
}

static vector unsigned char __ATTRS_o_ai
vec_andc(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a & ~__b;
}

static vector unsigned char __ATTRS_o_ai
vec_andc(vector unsigned char __a, vector bool char __b)
{
  return __a & ~(vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_andc(vector bool char __a, vector bool char __b)
{
  return __a & ~__b;
}

static vector short __ATTRS_o_ai
vec_andc(vector short __a, vector short __b)
{
  return __a & ~__b;
}

static vector short __ATTRS_o_ai
vec_andc(vector bool short __a, vector short __b)
{
  return (vector short)__a & ~__b;
}

static vector short __ATTRS_o_ai
vec_andc(vector short __a, vector bool short __b)
{
  return __a & ~(vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_andc(vector unsigned short __a, vector unsigned short __b)
{
  return __a & ~__b;
}

static vector unsigned short __ATTRS_o_ai
vec_andc(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a & ~__b;
}

static vector unsigned short __ATTRS_o_ai
vec_andc(vector unsigned short __a, vector bool short __b)
{
  return __a & ~(vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_andc(vector bool short __a, vector bool short __b)
{
  return __a & ~__b;
}

static vector int __ATTRS_o_ai
vec_andc(vector int __a, vector int __b)
{
  return __a & ~__b;
}

static vector int __ATTRS_o_ai
vec_andc(vector bool int __a, vector int __b)
{
  return (vector int)__a & ~__b;
}

static vector int __ATTRS_o_ai
vec_andc(vector int __a, vector bool int __b)
{
  return __a & ~(vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_andc(vector unsigned int __a, vector unsigned int __b)
{
  return __a & ~__b;
}

static vector unsigned int __ATTRS_o_ai
vec_andc(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a & ~__b;
}

static vector unsigned int __ATTRS_o_ai
vec_andc(vector unsigned int __a, vector bool int __b)
{
  return __a & ~(vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_andc(vector bool int __a, vector bool int __b)
{
  return __a & ~__b;
}

static vector float __ATTRS_o_ai
vec_andc(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & ~(vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_andc(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & ~(vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_andc(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a & ~(vector unsigned int)__b;
  return (vector float)__res;
}

/* vec_vandc */

static vector signed char __ATTRS_o_ai
vec_vandc(vector signed char __a, vector signed char __b)
{
  return __a & ~__b;
}

static vector signed char __ATTRS_o_ai
vec_vandc(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a & ~__b;
}

static vector signed char __ATTRS_o_ai
vec_vandc(vector signed char __a, vector bool char __b)
{
  return __a & ~(vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vandc(vector unsigned char __a, vector unsigned char __b)
{
  return __a & ~__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vandc(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a & ~__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vandc(vector unsigned char __a, vector bool char __b)
{
  return __a & ~(vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_vandc(vector bool char __a, vector bool char __b)
{
  return __a & ~__b;
}

static vector short __ATTRS_o_ai
vec_vandc(vector short __a, vector short __b)
{
  return __a & ~__b;
}

static vector short __ATTRS_o_ai
vec_vandc(vector bool short __a, vector short __b)
{
  return (vector short)__a & ~__b;
}

static vector short __ATTRS_o_ai
vec_vandc(vector short __a, vector bool short __b)
{
  return __a & ~(vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vandc(vector unsigned short __a, vector unsigned short __b)
{
  return __a & ~__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vandc(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a & ~__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vandc(vector unsigned short __a, vector bool short __b)
{
  return __a & ~(vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_vandc(vector bool short __a, vector bool short __b)
{
  return __a & ~__b;
}

static vector int __ATTRS_o_ai
vec_vandc(vector int __a, vector int __b)
{
  return __a & ~__b;
}

static vector int __ATTRS_o_ai
vec_vandc(vector bool int __a, vector int __b)
{
  return (vector int)__a & ~__b;
}

static vector int __ATTRS_o_ai
vec_vandc(vector int __a, vector bool int __b)
{
  return __a & ~(vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vandc(vector unsigned int __a, vector unsigned int __b)
{
  return __a & ~__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vandc(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a & ~__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vandc(vector unsigned int __a, vector bool int __b)
{
  return __a & ~(vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_vandc(vector bool int __a, vector bool int __b)
{
  return __a & ~__b;
}

static vector float __ATTRS_o_ai
vec_vandc(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & ~(vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vandc(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a & ~(vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vandc(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a & ~(vector unsigned int)__b;
  return (vector float)__res;
}

/* vec_avg */

static vector signed char __ATTRS_o_ai
vec_avg(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vavgsb(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_avg(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vavgub(__a, __b);
}

static vector short __ATTRS_o_ai
vec_avg(vector short __a, vector short __b)
{
  return __builtin_altivec_vavgsh(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_avg(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vavguh(__a, __b);
}

static vector int __ATTRS_o_ai
vec_avg(vector int __a, vector int __b)
{
  return __builtin_altivec_vavgsw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_avg(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vavguw(__a, __b);
}

/* vec_vavgsb */

static vector signed char __attribute__((__always_inline__))
vec_vavgsb(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vavgsb(__a, __b);
}

/* vec_vavgub */

static vector unsigned char __attribute__((__always_inline__))
vec_vavgub(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vavgub(__a, __b);
}

/* vec_vavgsh */

static vector short __attribute__((__always_inline__))
vec_vavgsh(vector short __a, vector short __b)
{
  return __builtin_altivec_vavgsh(__a, __b);
}

/* vec_vavguh */

static vector unsigned short __attribute__((__always_inline__))
vec_vavguh(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vavguh(__a, __b);
}

/* vec_vavgsw */

static vector int __attribute__((__always_inline__))
vec_vavgsw(vector int __a, vector int __b)
{
  return __builtin_altivec_vavgsw(__a, __b);
}

/* vec_vavguw */

static vector unsigned int __attribute__((__always_inline__))
vec_vavguw(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vavguw(__a, __b);
}

/* vec_ceil */

static vector float __attribute__((__always_inline__))
vec_ceil(vector float __a)
{
  return __builtin_altivec_vrfip(__a);
}

/* vec_vrfip */

static vector float __attribute__((__always_inline__))
vec_vrfip(vector float __a)
{
  return __builtin_altivec_vrfip(__a);
}

/* vec_cmpb */

static vector int __attribute__((__always_inline__))
vec_cmpb(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpbfp(__a, __b);
}

/* vec_vcmpbfp */

static vector int __attribute__((__always_inline__))
vec_vcmpbfp(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpbfp(__a, __b);
}

/* vec_cmpeq */

static vector bool char __ATTRS_o_ai
vec_cmpeq(vector signed char __a, vector signed char __b)
{
  return (vector bool char)
    __builtin_altivec_vcmpequb((vector char)__a, (vector char)__b);
}

static vector bool char __ATTRS_o_ai
vec_cmpeq(vector unsigned char __a, vector unsigned char __b)
{
  return (vector bool char)
    __builtin_altivec_vcmpequb((vector char)__a, (vector char)__b);
}

static vector bool short __ATTRS_o_ai
vec_cmpeq(vector short __a, vector short __b)
{
  return (vector bool short)__builtin_altivec_vcmpequh(__a, __b);
}

static vector bool short __ATTRS_o_ai
vec_cmpeq(vector unsigned short __a, vector unsigned short __b)
{
  return (vector bool short)
    __builtin_altivec_vcmpequh((vector short)__a, (vector short)__b);
}

static vector bool int __ATTRS_o_ai
vec_cmpeq(vector int __a, vector int __b)
{
  return (vector bool int)__builtin_altivec_vcmpequw(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_cmpeq(vector unsigned int __a, vector unsigned int __b)
{
  return (vector bool int)
    __builtin_altivec_vcmpequw((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_cmpeq(vector float __a, vector float __b)
{
  return (vector bool int)__builtin_altivec_vcmpeqfp(__a, __b);
}

/* vec_cmpge */

static vector bool int __attribute__((__always_inline__))
vec_cmpge(vector float __a, vector float __b)
{
  return (vector bool int)__builtin_altivec_vcmpgefp(__a, __b);
}

/* vec_vcmpgefp */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgefp(vector float __a, vector float __b)
{
  return (vector bool int)__builtin_altivec_vcmpgefp(__a, __b);
}

/* vec_cmpgt */

static vector bool char __ATTRS_o_ai
vec_cmpgt(vector signed char __a, vector signed char __b)
{
  return (vector bool char)__builtin_altivec_vcmpgtsb(__a, __b);
}

static vector bool char __ATTRS_o_ai
vec_cmpgt(vector unsigned char __a, vector unsigned char __b)
{
  return (vector bool char)__builtin_altivec_vcmpgtub(__a, __b);
}

static vector bool short __ATTRS_o_ai
vec_cmpgt(vector short __a, vector short __b)
{
  return (vector bool short)__builtin_altivec_vcmpgtsh(__a, __b);
}

static vector bool short __ATTRS_o_ai
vec_cmpgt(vector unsigned short __a, vector unsigned short __b)
{
  return (vector bool short)__builtin_altivec_vcmpgtuh(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_cmpgt(vector int __a, vector int __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtsw(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_cmpgt(vector unsigned int __a, vector unsigned int __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtuw(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_cmpgt(vector float __a, vector float __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtfp(__a, __b);
}

/* vec_vcmpgtsb */

static vector bool char __attribute__((__always_inline__))
vec_vcmpgtsb(vector signed char __a, vector signed char __b)
{
  return (vector bool char)__builtin_altivec_vcmpgtsb(__a, __b);
}

/* vec_vcmpgtub */

static vector bool char __attribute__((__always_inline__))
vec_vcmpgtub(vector unsigned char __a, vector unsigned char __b)
{
  return (vector bool char)__builtin_altivec_vcmpgtub(__a, __b);
}

/* vec_vcmpgtsh */

static vector bool short __attribute__((__always_inline__))
vec_vcmpgtsh(vector short __a, vector short __b)
{
  return (vector bool short)__builtin_altivec_vcmpgtsh(__a, __b);
}

/* vec_vcmpgtuh */

static vector bool short __attribute__((__always_inline__))
vec_vcmpgtuh(vector unsigned short __a, vector unsigned short __b)
{
  return (vector bool short)__builtin_altivec_vcmpgtuh(__a, __b);
}

/* vec_vcmpgtsw */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgtsw(vector int __a, vector int __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtsw(__a, __b);
}

/* vec_vcmpgtuw */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgtuw(vector unsigned int __a, vector unsigned int __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtuw(__a, __b);
}

/* vec_vcmpgtfp */

static vector bool int __attribute__((__always_inline__))
vec_vcmpgtfp(vector float __a, vector float __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtfp(__a, __b);
}

/* vec_cmple */

static vector bool int __attribute__((__always_inline__))
vec_cmple(vector float __a, vector float __b)
{
  return (vector bool int)__builtin_altivec_vcmpgefp(__b, __a);
}

/* vec_cmplt */

static vector bool char __ATTRS_o_ai
vec_cmplt(vector signed char __a, vector signed char __b)
{
  return (vector bool char)__builtin_altivec_vcmpgtsb(__b, __a);
}

static vector bool char __ATTRS_o_ai
vec_cmplt(vector unsigned char __a, vector unsigned char __b)
{
  return (vector bool char)__builtin_altivec_vcmpgtub(__b, __a);
}

static vector bool short __ATTRS_o_ai
vec_cmplt(vector short __a, vector short __b)
{
  return (vector bool short)__builtin_altivec_vcmpgtsh(__b, __a);
}

static vector bool short __ATTRS_o_ai
vec_cmplt(vector unsigned short __a, vector unsigned short __b)
{
  return (vector bool short)__builtin_altivec_vcmpgtuh(__b, __a);
}

static vector bool int __ATTRS_o_ai
vec_cmplt(vector int __a, vector int __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtsw(__b, __a);
}

static vector bool int __ATTRS_o_ai
vec_cmplt(vector unsigned int __a, vector unsigned int __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtuw(__b, __a);
}

static vector bool int __ATTRS_o_ai
vec_cmplt(vector float __a, vector float __b)
{
  return (vector bool int)__builtin_altivec_vcmpgtfp(__b, __a);
}

/* vec_ctf */

static vector float __ATTRS_o_ai
vec_ctf(vector int __a, int __b)
{
  return __builtin_altivec_vcfsx(__a, __b);
}

static vector float __ATTRS_o_ai
vec_ctf(vector unsigned int __a, int __b)
{
  return __builtin_altivec_vcfux((vector int)__a, __b);
}

/* vec_vcfsx */

static vector float __attribute__((__always_inline__))
vec_vcfsx(vector int __a, int __b)
{
  return __builtin_altivec_vcfsx(__a, __b);
}

/* vec_vcfux */

static vector float __attribute__((__always_inline__))
vec_vcfux(vector unsigned int __a, int __b)
{
  return __builtin_altivec_vcfux((vector int)__a, __b);
}

/* vec_cts */

static vector int __attribute__((__always_inline__))
vec_cts(vector float __a, int __b)
{
  return __builtin_altivec_vctsxs(__a, __b);
}

/* vec_vctsxs */

static vector int __attribute__((__always_inline__))
vec_vctsxs(vector float __a, int __b)
{
  return __builtin_altivec_vctsxs(__a, __b);
}

/* vec_ctu */

static vector unsigned int __attribute__((__always_inline__))
vec_ctu(vector float __a, int __b)
{
  return __builtin_altivec_vctuxs(__a, __b);
}

/* vec_vctuxs */

static vector unsigned int __attribute__((__always_inline__))
vec_vctuxs(vector float __a, int __b)
{
  return __builtin_altivec_vctuxs(__a, __b);
}

/* vec_dss */

static void __attribute__((__always_inline__))
vec_dss(int __a)
{
  __builtin_altivec_dss(__a);
}

/* vec_dssall */

static void __attribute__((__always_inline__))
vec_dssall(void)
{
  __builtin_altivec_dssall();
}

/* vec_dst */

static void __attribute__((__always_inline__))
vec_dst(const void *__a, int __b, int __c)
{
  __builtin_altivec_dst(__a, __b, __c);
}

/* vec_dstst */

static void __attribute__((__always_inline__))
vec_dstst(const void *__a, int __b, int __c)
{
  __builtin_altivec_dstst(__a, __b, __c);
}

/* vec_dststt */

static void __attribute__((__always_inline__))
vec_dststt(const void *__a, int __b, int __c)
{
  __builtin_altivec_dststt(__a, __b, __c);
}

/* vec_dstt */

static void __attribute__((__always_inline__))
vec_dstt(const void *__a, int __b, int __c)
{
  __builtin_altivec_dstt(__a, __b, __c);
}

/* vec_expte */

static vector float __attribute__((__always_inline__))
vec_expte(vector float __a)
{
  return __builtin_altivec_vexptefp(__a);
}

/* vec_vexptefp */

static vector float __attribute__((__always_inline__))
vec_vexptefp(vector float __a)
{
  return __builtin_altivec_vexptefp(__a);
}

/* vec_floor */

static vector float __attribute__((__always_inline__))
vec_floor(vector float __a)
{
  return __builtin_altivec_vrfim(__a);
}

/* vec_vrfim */

static vector float __attribute__((__always_inline__))
vec_vrfim(vector float __a)
{
  return __builtin_altivec_vrfim(__a);
}

/* vec_ld */

static vector signed char __ATTRS_o_ai
vec_ld(int __a, const vector signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvx(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_ld(int __a, const signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_ld(int __a, const vector unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_ld(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvx(__a, __b);
}

static vector bool char __ATTRS_o_ai
vec_ld(int __a, const vector bool char *__b)
{
  return (vector bool char)__builtin_altivec_lvx(__a, __b);
}

static vector short __ATTRS_o_ai
vec_ld(int __a, const vector short *__b)
{
  return (vector short)__builtin_altivec_lvx(__a, __b);
}

static vector short __ATTRS_o_ai
vec_ld(int __a, const short *__b)
{
  return (vector short)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_ld(int __a, const vector unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_ld(int __a, const unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvx(__a, __b);
}

static vector bool short __ATTRS_o_ai
vec_ld(int __a, const vector bool short *__b)
{
  return (vector bool short)__builtin_altivec_lvx(__a, __b);
}

static vector pixel __ATTRS_o_ai
vec_ld(int __a, const vector pixel *__b)
{
  return (vector pixel)__builtin_altivec_lvx(__a, __b);
}

static vector int __ATTRS_o_ai
vec_ld(int __a, const vector int *__b)
{
  return (vector int)__builtin_altivec_lvx(__a, __b);
}

static vector int __ATTRS_o_ai
vec_ld(int __a, const int *__b)
{
  return (vector int)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_ld(int __a, const vector unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_ld(int __a, const unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvx(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_ld(int __a, const vector bool int *__b)
{
  return (vector bool int)__builtin_altivec_lvx(__a, __b);
}

static vector float __ATTRS_o_ai
vec_ld(int __a, const vector float *__b)
{
  return (vector float)__builtin_altivec_lvx(__a, __b);
}

static vector float __ATTRS_o_ai
vec_ld(int __a, const float *__b)
{
  return (vector float)__builtin_altivec_lvx(__a, __b);
}

/* vec_lvx */

static vector signed char __ATTRS_o_ai
vec_lvx(int __a, const vector signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvx(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_lvx(int __a, const signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvx(int __a, const vector unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvx(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvx(__a, __b);
}

static vector bool char __ATTRS_o_ai
vec_lvx(int __a, const vector bool char *__b)
{
  return (vector bool char)__builtin_altivec_lvx(__a, __b);
}

static vector short __ATTRS_o_ai
vec_lvx(int __a, const vector short *__b)
{
  return (vector short)__builtin_altivec_lvx(__a, __b);
}

static vector short __ATTRS_o_ai
vec_lvx(int __a, const short *__b)
{
  return (vector short)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvx(int __a, const vector unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvx(int __a, const unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvx(__a, __b);
}

static vector bool short __ATTRS_o_ai
vec_lvx(int __a, const vector bool short *__b)
{
  return (vector bool short)__builtin_altivec_lvx(__a, __b);
}

static vector pixel __ATTRS_o_ai
vec_lvx(int __a, const vector pixel *__b)
{
  return (vector pixel)__builtin_altivec_lvx(__a, __b);
}

static vector int __ATTRS_o_ai
vec_lvx(int __a, const vector int *__b)
{
  return (vector int)__builtin_altivec_lvx(__a, __b);
}

static vector int __ATTRS_o_ai
vec_lvx(int __a, const int *__b)
{
  return (vector int)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvx(int __a, const vector unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvx(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvx(int __a, const unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvx(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_lvx(int __a, const vector bool int *__b)
{
  return (vector bool int)__builtin_altivec_lvx(__a, __b);
}

static vector float __ATTRS_o_ai
vec_lvx(int __a, const vector float *__b)
{
  return (vector float)__builtin_altivec_lvx(__a, __b);
}

static vector float __ATTRS_o_ai
vec_lvx(int __a, const float *__b)
{
  return (vector float)__builtin_altivec_lvx(__a, __b);
}

/* vec_lde */

static vector signed char __ATTRS_o_ai
vec_lde(int __a, const signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvebx(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lde(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvebx(__a, __b);
}

static vector short __ATTRS_o_ai
vec_lde(int __a, const short *__b)
{
  return (vector short)__builtin_altivec_lvehx(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_lde(int __a, const unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvehx(__a, __b);
}

static vector int __ATTRS_o_ai
vec_lde(int __a, const int *__b)
{
  return (vector int)__builtin_altivec_lvewx(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_lde(int __a, const unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvewx(__a, __b);
}

static vector float __ATTRS_o_ai
vec_lde(int __a, const float *__b)
{
  return (vector float)__builtin_altivec_lvewx(__a, __b);
}

/* vec_lvebx */

static vector signed char __ATTRS_o_ai
vec_lvebx(int __a, const signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvebx(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvebx(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvebx(__a, __b);
}

/* vec_lvehx */

static vector short __ATTRS_o_ai
vec_lvehx(int __a, const short *__b)
{
  return (vector short)__builtin_altivec_lvehx(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvehx(int __a, const unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvehx(__a, __b);
}

/* vec_lvewx */

static vector int __ATTRS_o_ai
vec_lvewx(int __a, const int *__b)
{
  return (vector int)__builtin_altivec_lvewx(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvewx(int __a, const unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvewx(__a, __b);
}

static vector float __ATTRS_o_ai
vec_lvewx(int __a, const float *__b)
{
  return (vector float)__builtin_altivec_lvewx(__a, __b);
}

/* vec_ldl */

static vector signed char __ATTRS_o_ai
vec_ldl(int __a, const vector signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvxl(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_ldl(int __a, const signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_ldl(int __a, const vector unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_ldl(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(__a, __b);
}

static vector bool char __ATTRS_o_ai
vec_ldl(int __a, const vector bool char *__b)
{
  return (vector bool char)__builtin_altivec_lvxl(__a, __b);
}

static vector short __ATTRS_o_ai
vec_ldl(int __a, const vector short *__b)
{
  return (vector short)__builtin_altivec_lvxl(__a, __b);
}

static vector short __ATTRS_o_ai
vec_ldl(int __a, const short *__b)
{
  return (vector short)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_ldl(int __a, const vector unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_ldl(int __a, const unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(__a, __b);
}

static vector bool short __ATTRS_o_ai
vec_ldl(int __a, const vector bool short *__b)
{
  return (vector bool short)__builtin_altivec_lvxl(__a, __b);
}

static vector pixel __ATTRS_o_ai
vec_ldl(int __a, const vector pixel *__b)
{
  return (vector pixel short)__builtin_altivec_lvxl(__a, __b);
}

static vector int __ATTRS_o_ai
vec_ldl(int __a, const vector int *__b)
{
  return (vector int)__builtin_altivec_lvxl(__a, __b);
}

static vector int __ATTRS_o_ai
vec_ldl(int __a, const int *__b)
{
  return (vector int)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_ldl(int __a, const vector unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_ldl(int __a, const unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_ldl(int __a, const vector bool int *__b)
{
  return (vector bool int)__builtin_altivec_lvxl(__a, __b);
}

static vector float __ATTRS_o_ai
vec_ldl(int __a, const vector float *__b)
{
  return (vector float)__builtin_altivec_lvxl(__a, __b);
}

static vector float __ATTRS_o_ai
vec_ldl(int __a, const float *__b)
{
  return (vector float)__builtin_altivec_lvxl(__a, __b);
}

/* vec_lvxl */

static vector signed char __ATTRS_o_ai
vec_lvxl(int __a, const vector signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvxl(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_lvxl(int __a, const signed char *__b)
{
  return (vector signed char)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvxl(int __a, const vector unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvxl(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvxl(__a, __b);
}

static vector bool char __ATTRS_o_ai
vec_lvxl(int __a, const vector bool char *__b)
{
  return (vector bool char)__builtin_altivec_lvxl(__a, __b);
}

static vector short __ATTRS_o_ai
vec_lvxl(int __a, const vector short *__b)
{
  return (vector short)__builtin_altivec_lvxl(__a, __b);
}

static vector short __ATTRS_o_ai
vec_lvxl(int __a, const short *__b)
{
  return (vector short)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvxl(int __a, const vector unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_lvxl(int __a, const unsigned short *__b)
{
  return (vector unsigned short)__builtin_altivec_lvxl(__a, __b);
}

static vector bool short __ATTRS_o_ai
vec_lvxl(int __a, const vector bool short *__b)
{
  return (vector bool short)__builtin_altivec_lvxl(__a, __b);
}

static vector pixel __ATTRS_o_ai
vec_lvxl(int __a, const vector pixel *__b)
{
  return (vector pixel)__builtin_altivec_lvxl(__a, __b);
}

static vector int __ATTRS_o_ai
vec_lvxl(int __a, const vector int *__b)
{
  return (vector int)__builtin_altivec_lvxl(__a, __b);
}

static vector int __ATTRS_o_ai
vec_lvxl(int __a, const int *__b)
{
  return (vector int)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvxl(int __a, const vector unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_lvxl(int __a, const unsigned int *__b)
{
  return (vector unsigned int)__builtin_altivec_lvxl(__a, __b);
}

static vector bool int __ATTRS_o_ai
vec_lvxl(int __a, const vector bool int *__b)
{
  return (vector bool int)__builtin_altivec_lvxl(__a, __b);
}

static vector float __ATTRS_o_ai
vec_lvxl(int __a, const vector float *__b)
{
  return (vector float)__builtin_altivec_lvxl(__a, __b);
}

static vector float __ATTRS_o_ai
vec_lvxl(int __a, const float *__b)
{
  return (vector float)__builtin_altivec_lvxl(__a, __b);
}

/* vec_loge */

static vector float __attribute__((__always_inline__))
vec_loge(vector float __a)
{
  return __builtin_altivec_vlogefp(__a);
}

/* vec_vlogefp */

static vector float __attribute__((__always_inline__))
vec_vlogefp(vector float __a)
{
  return __builtin_altivec_vlogefp(__a);
}

/* vec_lvsl */

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int __a, const signed char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int __a, const short *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int __a, const unsigned short *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int __a, const int *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int __a, const unsigned int *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsl(int __a, const float *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsl(__a, __b);
}

/* vec_lvsr */

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int __a, const signed char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int __a, const unsigned char *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int __a, const short *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int __a, const unsigned short *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int __a, const int *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int __a, const unsigned int *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_lvsr(int __a, const float *__b)
{
  return (vector unsigned char)__builtin_altivec_lvsr(__a, __b);
}

/* vec_madd */

static vector float __attribute__((__always_inline__))
vec_madd(vector float __a, vector float __b, vector float __c)
{
  return __builtin_altivec_vmaddfp(__a, __b, __c);
}

/* vec_vmaddfp */

static vector float __attribute__((__always_inline__))
vec_vmaddfp(vector float __a, vector float __b, vector float __c)
{
  return __builtin_altivec_vmaddfp(__a, __b, __c);
}

/* vec_madds */

static vector signed short __attribute__((__always_inline__))
vec_madds(vector signed short __a, vector signed short __b, vector signed short __c)
{
  return __builtin_altivec_vmhaddshs(__a, __b, __c);
}

/* vec_vmhaddshs */
static vector signed short __attribute__((__always_inline__))
vec_vmhaddshs(vector signed short __a,
              vector signed short __b,
              vector signed short __c)
{
  return __builtin_altivec_vmhaddshs(__a, __b, __c);
}

/* vec_max */

static vector signed char __ATTRS_o_ai
vec_max(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vmaxsb(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_max(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vmaxsb((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_max(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vmaxsb(__a, (vector signed char)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_max(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vmaxub(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_max(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vmaxub((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_max(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vmaxub(__a, (vector unsigned char)__b);
}

static vector short __ATTRS_o_ai
vec_max(vector short __a, vector short __b)
{
  return __builtin_altivec_vmaxsh(__a, __b);
}

static vector short __ATTRS_o_ai
vec_max(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vmaxsh((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_max(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vmaxsh(__a, (vector short)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_max(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vmaxuh(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_max(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vmaxuh((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_max(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vmaxuh(__a, (vector unsigned short)__b);
}

static vector int __ATTRS_o_ai
vec_max(vector int __a, vector int __b)
{
  return __builtin_altivec_vmaxsw(__a, __b);
}

static vector int __ATTRS_o_ai
vec_max(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vmaxsw((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_max(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vmaxsw(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_max(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vmaxuw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_max(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vmaxuw((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_max(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vmaxuw(__a, (vector unsigned int)__b);
}

static vector float __ATTRS_o_ai
vec_max(vector float __a, vector float __b)
{
  return __builtin_altivec_vmaxfp(__a, __b);
}

/* vec_vmaxsb */

static vector signed char __ATTRS_o_ai
vec_vmaxsb(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vmaxsb(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vmaxsb(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vmaxsb((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vmaxsb(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vmaxsb(__a, (vector signed char)__b);
}

/* vec_vmaxub */

static vector unsigned char __ATTRS_o_ai
vec_vmaxub(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vmaxub(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vmaxub(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vmaxub((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vmaxub(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vmaxub(__a, (vector unsigned char)__b);
}

/* vec_vmaxsh */

static vector short __ATTRS_o_ai
vec_vmaxsh(vector short __a, vector short __b)
{
  return __builtin_altivec_vmaxsh(__a, __b);
}

static vector short __ATTRS_o_ai
vec_vmaxsh(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vmaxsh((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_vmaxsh(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vmaxsh(__a, (vector short)__b);
}

/* vec_vmaxuh */

static vector unsigned short __ATTRS_o_ai
vec_vmaxuh(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vmaxuh(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vmaxuh(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vmaxuh((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vmaxuh(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vmaxuh(__a, (vector unsigned short)__b);
}

/* vec_vmaxsw */

static vector int __ATTRS_o_ai
vec_vmaxsw(vector int __a, vector int __b)
{
  return __builtin_altivec_vmaxsw(__a, __b);
}

static vector int __ATTRS_o_ai
vec_vmaxsw(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vmaxsw((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_vmaxsw(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vmaxsw(__a, (vector int)__b);
}

/* vec_vmaxuw */

static vector unsigned int __ATTRS_o_ai
vec_vmaxuw(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vmaxuw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vmaxuw(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vmaxuw((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vmaxuw(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vmaxuw(__a, (vector unsigned int)__b);
}

/* vec_vmaxfp */

static vector float __attribute__((__always_inline__))
vec_vmaxfp(vector float __a, vector float __b)
{
  return __builtin_altivec_vmaxfp(__a, __b);
}

/* vec_mergeh */

static vector signed char __ATTRS_o_ai
vec_mergeh(vector signed char __a, vector signed char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector unsigned char __ATTRS_o_ai
vec_mergeh(vector unsigned char __a, vector unsigned char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector bool char __ATTRS_o_ai
vec_mergeh(vector bool char __a, vector bool char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector short __ATTRS_o_ai
vec_mergeh(vector short __a, vector short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector unsigned short __ATTRS_o_ai
vec_mergeh(vector unsigned short __a, vector unsigned short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector bool short __ATTRS_o_ai
vec_mergeh(vector bool short __a, vector bool short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector pixel __ATTRS_o_ai
vec_mergeh(vector pixel __a, vector pixel __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector int __ATTRS_o_ai
vec_mergeh(vector int __a, vector int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector unsigned int __ATTRS_o_ai
vec_mergeh(vector unsigned int __a, vector unsigned int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector bool int __ATTRS_o_ai
vec_mergeh(vector bool int __a, vector bool int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector float __ATTRS_o_ai
vec_mergeh(vector float __a, vector float __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

/* vec_vmrghb */

#define __builtin_altivec_vmrghb vec_vmrghb

static vector signed char __ATTRS_o_ai
vec_vmrghb(vector signed char __a, vector signed char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector unsigned char __ATTRS_o_ai
vec_vmrghb(vector unsigned char __a, vector unsigned char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

static vector bool char __ATTRS_o_ai
vec_vmrghb(vector bool char __a, vector bool char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 
     0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17));
}

/* vec_vmrghh */

#define __builtin_altivec_vmrghh vec_vmrghh

static vector short __ATTRS_o_ai
vec_vmrghh(vector short __a, vector short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector unsigned short __ATTRS_o_ai
vec_vmrghh(vector unsigned short __a, vector unsigned short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector bool short __ATTRS_o_ai
vec_vmrghh(vector bool short __a, vector bool short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

static vector pixel __ATTRS_o_ai
vec_vmrghh(vector pixel __a, vector pixel __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17));
}

/* vec_vmrghw */

#define __builtin_altivec_vmrghw vec_vmrghw

static vector int __ATTRS_o_ai
vec_vmrghw(vector int __a, vector int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector unsigned int __ATTRS_o_ai
vec_vmrghw(vector unsigned int __a, vector unsigned int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector bool int __ATTRS_o_ai
vec_vmrghw(vector bool int __a, vector bool int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

static vector float __ATTRS_o_ai
vec_vmrghw(vector float __a, vector float __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17));
}

/* vec_mergel */

static vector signed char __ATTRS_o_ai
vec_mergel(vector signed char __a, vector signed char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector unsigned char __ATTRS_o_ai
vec_mergel(vector unsigned char __a, vector unsigned char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector bool char __ATTRS_o_ai
vec_mergel(vector bool char __a, vector bool char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector short __ATTRS_o_ai
vec_mergel(vector short __a, vector short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector unsigned short __ATTRS_o_ai
vec_mergel(vector unsigned short __a, vector unsigned short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector bool short __ATTRS_o_ai
vec_mergel(vector bool short __a, vector bool short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector pixel __ATTRS_o_ai
vec_mergel(vector pixel __a, vector pixel __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector int __ATTRS_o_ai
vec_mergel(vector int __a, vector int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector unsigned int __ATTRS_o_ai
vec_mergel(vector unsigned int __a, vector unsigned int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector bool int __ATTRS_o_ai
vec_mergel(vector bool int __a, vector bool int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector float __ATTRS_o_ai
vec_mergel(vector float __a, vector float __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

/* vec_vmrglb */

#define __builtin_altivec_vmrglb vec_vmrglb

static vector signed char __ATTRS_o_ai
vec_vmrglb(vector signed char __a, vector signed char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector unsigned char __ATTRS_o_ai
vec_vmrglb(vector unsigned char __a, vector unsigned char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

static vector bool char __ATTRS_o_ai
vec_vmrglb(vector bool char __a, vector bool char __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 
     0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F));
}

/* vec_vmrglh */

#define __builtin_altivec_vmrglh vec_vmrglh

static vector short __ATTRS_o_ai
vec_vmrglh(vector short __a, vector short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector unsigned short __ATTRS_o_ai
vec_vmrglh(vector unsigned short __a, vector unsigned short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector bool short __ATTRS_o_ai
vec_vmrglh(vector bool short __a, vector bool short __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

static vector pixel __ATTRS_o_ai
vec_vmrglh(vector pixel __a, vector pixel __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x18, 0x19, 0x0A, 0x0B, 0x1A, 0x1B,
     0x0C, 0x0D, 0x1C, 0x1D, 0x0E, 0x0F, 0x1E, 0x1F));
}

/* vec_vmrglw */

#define __builtin_altivec_vmrglw vec_vmrglw

static vector int __ATTRS_o_ai
vec_vmrglw(vector int __a, vector int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector unsigned int __ATTRS_o_ai
vec_vmrglw(vector unsigned int __a, vector unsigned int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector bool int __ATTRS_o_ai
vec_vmrglw(vector bool int __a, vector bool int __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

static vector float __ATTRS_o_ai
vec_vmrglw(vector float __a, vector float __b)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B,
     0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F));
}

/* vec_mfvscr */

static vector unsigned short __attribute__((__always_inline__))
vec_mfvscr(void)
{
  return __builtin_altivec_mfvscr();
}

/* vec_min */

static vector signed char __ATTRS_o_ai
vec_min(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vminsb(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_min(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vminsb((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_min(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vminsb(__a, (vector signed char)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_min(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vminub(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_min(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vminub((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_min(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vminub(__a, (vector unsigned char)__b);
}

static vector short __ATTRS_o_ai
vec_min(vector short __a, vector short __b)
{
  return __builtin_altivec_vminsh(__a, __b);
}

static vector short __ATTRS_o_ai
vec_min(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vminsh((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_min(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vminsh(__a, (vector short)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_min(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vminuh(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_min(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vminuh((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_min(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vminuh(__a, (vector unsigned short)__b);
}

static vector int __ATTRS_o_ai
vec_min(vector int __a, vector int __b)
{
  return __builtin_altivec_vminsw(__a, __b);
}

static vector int __ATTRS_o_ai
vec_min(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vminsw((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_min(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vminsw(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_min(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vminuw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_min(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vminuw((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_min(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vminuw(__a, (vector unsigned int)__b);
}

static vector float __ATTRS_o_ai
vec_min(vector float __a, vector float __b)
{
  return __builtin_altivec_vminfp(__a, __b);
}

/* vec_vminsb */

static vector signed char __ATTRS_o_ai
vec_vminsb(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vminsb(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vminsb(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vminsb((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vminsb(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vminsb(__a, (vector signed char)__b);
}

/* vec_vminub */

static vector unsigned char __ATTRS_o_ai
vec_vminub(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vminub(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vminub(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vminub((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vminub(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vminub(__a, (vector unsigned char)__b);
}

/* vec_vminsh */

static vector short __ATTRS_o_ai
vec_vminsh(vector short __a, vector short __b)
{
  return __builtin_altivec_vminsh(__a, __b);
}

static vector short __ATTRS_o_ai
vec_vminsh(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vminsh((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_vminsh(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vminsh(__a, (vector short)__b);
}

/* vec_vminuh */

static vector unsigned short __ATTRS_o_ai
vec_vminuh(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vminuh(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vminuh(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vminuh((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vminuh(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vminuh(__a, (vector unsigned short)__b);
}

/* vec_vminsw */

static vector int __ATTRS_o_ai
vec_vminsw(vector int __a, vector int __b)
{
  return __builtin_altivec_vminsw(__a, __b);
}

static vector int __ATTRS_o_ai
vec_vminsw(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vminsw((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_vminsw(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vminsw(__a, (vector int)__b);
}

/* vec_vminuw */

static vector unsigned int __ATTRS_o_ai
vec_vminuw(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vminuw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vminuw(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vminuw((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vminuw(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vminuw(__a, (vector unsigned int)__b);
}

/* vec_vminfp */

static vector float __attribute__((__always_inline__))
vec_vminfp(vector float __a, vector float __b)
{
  return __builtin_altivec_vminfp(__a, __b);
}

/* vec_mladd */

#define __builtin_altivec_vmladduhm vec_mladd

static vector short __ATTRS_o_ai
vec_mladd(vector short __a, vector short __b, vector short __c)
{
  return __a * __b + __c;
}

static vector short __ATTRS_o_ai
vec_mladd(vector short __a, vector unsigned short __b, vector unsigned short __c)
{
  return __a * (vector short)__b + (vector short)__c;
}

static vector short __ATTRS_o_ai
vec_mladd(vector unsigned short __a, vector short __b, vector short __c)
{
  return (vector short)__a * __b + __c;
}

static vector unsigned short __ATTRS_o_ai
vec_mladd(vector unsigned short __a,
          vector unsigned short __b,
          vector unsigned short __c)
{
  return __a * __b + __c;
}

/* vec_vmladduhm */

static vector short __ATTRS_o_ai
vec_vmladduhm(vector short __a, vector short __b, vector short __c)
{
  return __a * __b + __c;
}

static vector short __ATTRS_o_ai
vec_vmladduhm(vector short __a, vector unsigned short __b, vector unsigned short __c)
{
  return __a * (vector short)__b + (vector short)__c;
}

static vector short __ATTRS_o_ai
vec_vmladduhm(vector unsigned short __a, vector short __b, vector short __c)
{
  return (vector short)__a * __b + __c;
}

static vector unsigned short __ATTRS_o_ai
vec_vmladduhm(vector unsigned short __a,
              vector unsigned short __b,
              vector unsigned short __c)
{
  return __a * __b + __c;
}

/* vec_mradds */

static vector short __attribute__((__always_inline__))
vec_mradds(vector short __a, vector short __b, vector short __c)
{
  return __builtin_altivec_vmhraddshs(__a, __b, __c);
}

/* vec_vmhraddshs */

static vector short __attribute__((__always_inline__))
vec_vmhraddshs(vector short __a, vector short __b, vector short __c)
{
  return __builtin_altivec_vmhraddshs(__a, __b, __c);
}

/* vec_msum */

static vector int __ATTRS_o_ai
vec_msum(vector signed char __a, vector unsigned char __b, vector int __c)
{
  return __builtin_altivec_vmsummbm(__a, __b, __c);
}

static vector unsigned int __ATTRS_o_ai
vec_msum(vector unsigned char __a, vector unsigned char __b, vector unsigned int __c)
{
  return __builtin_altivec_vmsumubm(__a, __b, __c);
}

static vector int __ATTRS_o_ai
vec_msum(vector short __a, vector short __b, vector int __c)
{
  return __builtin_altivec_vmsumshm(__a, __b, __c);
}

static vector unsigned int __ATTRS_o_ai
vec_msum(vector unsigned short __a,
         vector unsigned short __b,
         vector unsigned int __c)
{
  return __builtin_altivec_vmsumuhm(__a, __b, __c);
}

/* vec_vmsummbm */

static vector int __attribute__((__always_inline__))
vec_vmsummbm(vector signed char __a, vector unsigned char __b, vector int __c)
{
  return __builtin_altivec_vmsummbm(__a, __b, __c);
}

/* vec_vmsumubm */

static vector unsigned int __attribute__((__always_inline__))
vec_vmsumubm(vector unsigned char __a,
             vector unsigned char __b,
             vector unsigned int __c)
{
  return __builtin_altivec_vmsumubm(__a, __b, __c);
}

/* vec_vmsumshm */

static vector int __attribute__((__always_inline__))
vec_vmsumshm(vector short __a, vector short __b, vector int __c)
{
  return __builtin_altivec_vmsumshm(__a, __b, __c);
}

/* vec_vmsumuhm */

static vector unsigned int __attribute__((__always_inline__))
vec_vmsumuhm(vector unsigned short __a,
             vector unsigned short __b,
             vector unsigned int __c)
{
  return __builtin_altivec_vmsumuhm(__a, __b, __c);
}

/* vec_msums */

static vector int __ATTRS_o_ai
vec_msums(vector short __a, vector short __b, vector int __c)
{
  return __builtin_altivec_vmsumshs(__a, __b, __c);
}

static vector unsigned int __ATTRS_o_ai
vec_msums(vector unsigned short __a,
          vector unsigned short __b,
          vector unsigned int __c)
{
  return __builtin_altivec_vmsumuhs(__a, __b, __c);
}

/* vec_vmsumshs */

static vector int __attribute__((__always_inline__))
vec_vmsumshs(vector short __a, vector short __b, vector int __c)
{
  return __builtin_altivec_vmsumshs(__a, __b, __c);
}

/* vec_vmsumuhs */

static vector unsigned int __attribute__((__always_inline__))
vec_vmsumuhs(vector unsigned short __a,
             vector unsigned short __b,
             vector unsigned int __c)
{
  return __builtin_altivec_vmsumuhs(__a, __b, __c);
}

/* vec_mtvscr */

static void __ATTRS_o_ai
vec_mtvscr(vector signed char __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector unsigned char __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector bool char __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector short __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector unsigned short __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector bool short __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector pixel __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector int __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector unsigned int __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector bool int __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

static void __ATTRS_o_ai
vec_mtvscr(vector float __a)
{
  __builtin_altivec_mtvscr((vector int)__a);
}

/* The vmulos* and vmules* instructions have a big endian bias, so
   we must reverse the meaning of "even" and "odd" for little endian.  */

/* vec_mule */

static vector short __ATTRS_o_ai
vec_mule(vector signed char __a, vector signed char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulosb(__a, __b);
#else
  return __builtin_altivec_vmulesb(__a, __b);
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_mule(vector unsigned char __a, vector unsigned char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmuloub(__a, __b);
#else
  return __builtin_altivec_vmuleub(__a, __b);
#endif
}

static vector int __ATTRS_o_ai
vec_mule(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulosh(__a, __b);
#else
  return __builtin_altivec_vmulesh(__a, __b);
#endif
}

static vector unsigned int __ATTRS_o_ai
vec_mule(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulouh(__a, __b);
#else
  return __builtin_altivec_vmuleuh(__a, __b);
#endif
}

/* vec_vmulesb */

static vector short __attribute__((__always_inline__))
vec_vmulesb(vector signed char __a, vector signed char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulosb(__a, __b);
#else
  return __builtin_altivec_vmulesb(__a, __b);
#endif
}

/* vec_vmuleub */

static vector unsigned short __attribute__((__always_inline__))
vec_vmuleub(vector unsigned char __a, vector unsigned char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmuloub(__a, __b);
#else
  return __builtin_altivec_vmuleub(__a, __b);
#endif
}

/* vec_vmulesh */

static vector int __attribute__((__always_inline__))
vec_vmulesh(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulosh(__a, __b);
#else
  return __builtin_altivec_vmulesh(__a, __b);
#endif
}

/* vec_vmuleuh */

static vector unsigned int __attribute__((__always_inline__))
vec_vmuleuh(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulouh(__a, __b);
#else
  return __builtin_altivec_vmuleuh(__a, __b);
#endif
}

/* vec_mulo */

static vector short __ATTRS_o_ai
vec_mulo(vector signed char __a, vector signed char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulesb(__a, __b);
#else
  return __builtin_altivec_vmulosb(__a, __b);
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_mulo(vector unsigned char __a, vector unsigned char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmuleub(__a, __b);
#else
  return __builtin_altivec_vmuloub(__a, __b);
#endif
}

static vector int __ATTRS_o_ai
vec_mulo(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulesh(__a, __b);
#else
  return __builtin_altivec_vmulosh(__a, __b);
#endif
}

static vector unsigned int __ATTRS_o_ai
vec_mulo(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmuleuh(__a, __b);
#else
  return __builtin_altivec_vmulouh(__a, __b);
#endif
}

/* vec_vmulosb */

static vector short __attribute__((__always_inline__))
vec_vmulosb(vector signed char __a, vector signed char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulesb(__a, __b);
#else
  return __builtin_altivec_vmulosb(__a, __b);
#endif
}

/* vec_vmuloub */

static vector unsigned short __attribute__((__always_inline__))
vec_vmuloub(vector unsigned char __a, vector unsigned char __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmuleub(__a, __b);
#else
  return __builtin_altivec_vmuloub(__a, __b);
#endif
}

/* vec_vmulosh */

static vector int __attribute__((__always_inline__))
vec_vmulosh(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmulesh(__a, __b);
#else
  return __builtin_altivec_vmulosh(__a, __b);
#endif
}

/* vec_vmulouh */

static vector unsigned int __attribute__((__always_inline__))
vec_vmulouh(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vmuleuh(__a, __b);
#else
  return __builtin_altivec_vmulouh(__a, __b);
#endif
}

/* vec_nmsub */

static vector float __attribute__((__always_inline__))
vec_nmsub(vector float __a, vector float __b, vector float __c)
{
  return __builtin_altivec_vnmsubfp(__a, __b, __c);
}

/* vec_vnmsubfp */

static vector float __attribute__((__always_inline__))
vec_vnmsubfp(vector float __a, vector float __b, vector float __c)
{
  return __builtin_altivec_vnmsubfp(__a, __b, __c);
}

/* vec_nor */

#define __builtin_altivec_vnor vec_nor

static vector signed char __ATTRS_o_ai
vec_nor(vector signed char __a, vector signed char __b)
{
  return ~(__a | __b);
}

static vector unsigned char __ATTRS_o_ai
vec_nor(vector unsigned char __a, vector unsigned char __b)
{
  return ~(__a | __b);
}

static vector bool char __ATTRS_o_ai
vec_nor(vector bool char __a, vector bool char __b)
{
  return ~(__a | __b);
}

static vector short __ATTRS_o_ai
vec_nor(vector short __a, vector short __b)
{
  return ~(__a | __b);
}

static vector unsigned short __ATTRS_o_ai
vec_nor(vector unsigned short __a, vector unsigned short __b)
{
  return ~(__a | __b);
}

static vector bool short __ATTRS_o_ai
vec_nor(vector bool short __a, vector bool short __b)
{
  return ~(__a | __b);
}

static vector int __ATTRS_o_ai
vec_nor(vector int __a, vector int __b)
{
  return ~(__a | __b);
}

static vector unsigned int __ATTRS_o_ai
vec_nor(vector unsigned int __a, vector unsigned int __b)
{
  return ~(__a | __b);
}

static vector bool int __ATTRS_o_ai
vec_nor(vector bool int __a, vector bool int __b)
{
  return ~(__a | __b);
}

static vector float __ATTRS_o_ai
vec_nor(vector float __a, vector float __b)
{
  vector unsigned int __res = ~((vector unsigned int)__a | (vector unsigned int)__b);
  return (vector float)__res;
}

/* vec_vnor */

static vector signed char __ATTRS_o_ai
vec_vnor(vector signed char __a, vector signed char __b)
{
  return ~(__a | __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vnor(vector unsigned char __a, vector unsigned char __b)
{
  return ~(__a | __b);
}

static vector bool char __ATTRS_o_ai
vec_vnor(vector bool char __a, vector bool char __b)
{
  return ~(__a | __b);
}

static vector short __ATTRS_o_ai
vec_vnor(vector short __a, vector short __b)
{
  return ~(__a | __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vnor(vector unsigned short __a, vector unsigned short __b)
{
  return ~(__a | __b);
}

static vector bool short __ATTRS_o_ai
vec_vnor(vector bool short __a, vector bool short __b)
{
  return ~(__a | __b);
}

static vector int __ATTRS_o_ai
vec_vnor(vector int __a, vector int __b)
{
  return ~(__a | __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vnor(vector unsigned int __a, vector unsigned int __b)
{
  return ~(__a | __b);
}

static vector bool int __ATTRS_o_ai
vec_vnor(vector bool int __a, vector bool int __b)
{
  return ~(__a | __b);
}

static vector float __ATTRS_o_ai
vec_vnor(vector float __a, vector float __b)
{
  vector unsigned int __res = ~((vector unsigned int)__a | (vector unsigned int)__b);
  return (vector float)__res;
}

/* vec_or */

#define __builtin_altivec_vor vec_or

static vector signed char __ATTRS_o_ai
vec_or(vector signed char __a, vector signed char __b)
{
  return __a | __b;
}

static vector signed char __ATTRS_o_ai
vec_or(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a | __b;
}

static vector signed char __ATTRS_o_ai
vec_or(vector signed char __a, vector bool char __b)
{
  return __a | (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_or(vector unsigned char __a, vector unsigned char __b)
{
  return __a | __b;
}

static vector unsigned char __ATTRS_o_ai
vec_or(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a | __b;
}

static vector unsigned char __ATTRS_o_ai
vec_or(vector unsigned char __a, vector bool char __b)
{
  return __a | (vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_or(vector bool char __a, vector bool char __b)
{
  return __a | __b;
}

static vector short __ATTRS_o_ai
vec_or(vector short __a, vector short __b)
{
  return __a | __b;
}

static vector short __ATTRS_o_ai
vec_or(vector bool short __a, vector short __b)
{
  return (vector short)__a | __b;
}

static vector short __ATTRS_o_ai
vec_or(vector short __a, vector bool short __b)
{
  return __a | (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_or(vector unsigned short __a, vector unsigned short __b)
{
  return __a | __b;
}

static vector unsigned short __ATTRS_o_ai
vec_or(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a | __b;
}

static vector unsigned short __ATTRS_o_ai
vec_or(vector unsigned short __a, vector bool short __b)
{
  return __a | (vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_or(vector bool short __a, vector bool short __b)
{
  return __a | __b;
}

static vector int __ATTRS_o_ai
vec_or(vector int __a, vector int __b)
{
  return __a | __b;
}

static vector int __ATTRS_o_ai
vec_or(vector bool int __a, vector int __b)
{
  return (vector int)__a | __b;
}

static vector int __ATTRS_o_ai
vec_or(vector int __a, vector bool int __b)
{
  return __a | (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_or(vector unsigned int __a, vector unsigned int __b)
{
  return __a | __b;
}

static vector unsigned int __ATTRS_o_ai
vec_or(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a | __b;
}

static vector unsigned int __ATTRS_o_ai
vec_or(vector unsigned int __a, vector bool int __b)
{
  return __a | (vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_or(vector bool int __a, vector bool int __b)
{
  return __a | __b;
}

static vector float __ATTRS_o_ai
vec_or(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a | (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_or(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a | (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_or(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a | (vector unsigned int)__b;
  return (vector float)__res;
}

/* vec_vor */

static vector signed char __ATTRS_o_ai
vec_vor(vector signed char __a, vector signed char __b)
{
  return __a | __b;
}

static vector signed char __ATTRS_o_ai
vec_vor(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a | __b;
}

static vector signed char __ATTRS_o_ai
vec_vor(vector signed char __a, vector bool char __b)
{
  return __a | (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vor(vector unsigned char __a, vector unsigned char __b)
{
  return __a | __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vor(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a | __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vor(vector unsigned char __a, vector bool char __b)
{
  return __a | (vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_vor(vector bool char __a, vector bool char __b)
{
  return __a | __b;
}

static vector short __ATTRS_o_ai
vec_vor(vector short __a, vector short __b)
{
  return __a | __b;
}

static vector short __ATTRS_o_ai
vec_vor(vector bool short __a, vector short __b)
{
  return (vector short)__a | __b;
}

static vector short __ATTRS_o_ai
vec_vor(vector short __a, vector bool short __b)
{
  return __a | (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vor(vector unsigned short __a, vector unsigned short __b)
{
  return __a | __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vor(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a | __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vor(vector unsigned short __a, vector bool short __b)
{
  return __a | (vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_vor(vector bool short __a, vector bool short __b)
{
  return __a | __b;
}

static vector int __ATTRS_o_ai
vec_vor(vector int __a, vector int __b)
{
  return __a | __b;
}

static vector int __ATTRS_o_ai
vec_vor(vector bool int __a, vector int __b)
{
  return (vector int)__a | __b;
}

static vector int __ATTRS_o_ai
vec_vor(vector int __a, vector bool int __b)
{
  return __a | (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vor(vector unsigned int __a, vector unsigned int __b)
{
  return __a | __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vor(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a | __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vor(vector unsigned int __a, vector bool int __b)
{
  return __a | (vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_vor(vector bool int __a, vector bool int __b)
{
  return __a | __b;
}

static vector float __ATTRS_o_ai
vec_vor(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a | (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vor(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a | (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vor(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a | (vector unsigned int)__b;
  return (vector float)__res;
}

/* vec_pack */

/* The various vector pack instructions have a big-endian bias, so for
   little endian we must handle reversed element numbering.  */

static vector signed char __ATTRS_o_ai
vec_pack(vector signed short __a, vector signed short __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector signed char)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
     0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E));
#else
  return (vector signed char)vec_perm(__a, __b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
#endif
}

static vector unsigned char __ATTRS_o_ai
vec_pack(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned char)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
     0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E));
#else
  return (vector unsigned char)vec_perm(__a, __b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
#endif
}

static vector bool char __ATTRS_o_ai
vec_pack(vector bool short __a, vector bool short __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool char)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
     0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E));
#else
  return (vector bool char)vec_perm(__a, __b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
#endif
}

static vector short __ATTRS_o_ai
vec_pack(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector short)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0C, 0x0D,
     0x10, 0x11, 0x14, 0x15, 0x18, 0x19, 0x1C, 0x1D));
#else
  return (vector short)vec_perm(__a, __b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_pack(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned short)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0C, 0x0D,
     0x10, 0x11, 0x14, 0x15, 0x18, 0x19, 0x1C, 0x1D));
#else
  return (vector unsigned short)vec_perm(__a, __b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
#endif
}

static vector bool short __ATTRS_o_ai
vec_pack(vector bool int __a, vector bool int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool short)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0C, 0x0D,
     0x10, 0x11, 0x14, 0x15, 0x18, 0x19, 0x1C, 0x1D));
#else
  return (vector bool short)vec_perm(__a, __b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
#endif
}

/* vec_vpkuhum */

#define __builtin_altivec_vpkuhum vec_vpkuhum

static vector signed char __ATTRS_o_ai
vec_vpkuhum(vector signed short __a, vector signed short __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector signed char)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
     0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E));
#else
  return (vector signed char)vec_perm(__a, __b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
#endif
}

static vector unsigned char __ATTRS_o_ai
vec_vpkuhum(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned char)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
     0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E));
#else
  return (vector unsigned char)vec_perm(__a, __b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
#endif
}

static vector bool char __ATTRS_o_ai
vec_vpkuhum(vector bool short __a, vector bool short __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool char)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E,
     0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E));
#else
  return (vector bool char)vec_perm(__a, __b, (vector unsigned char)
    (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F,
     0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F));
#endif
}

/* vec_vpkuwum */

#define __builtin_altivec_vpkuwum vec_vpkuwum

static vector short __ATTRS_o_ai
vec_vpkuwum(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector short)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0C, 0x0D,
     0x10, 0x11, 0x14, 0x15, 0x18, 0x19, 0x1C, 0x1D));
#else
  return (vector short)vec_perm(__a, __b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_vpkuwum(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned short)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0C, 0x0D,
     0x10, 0x11, 0x14, 0x15, 0x18, 0x19, 0x1C, 0x1D));
#else
  return (vector unsigned short)vec_perm(__a, __b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
#endif
}

static vector bool short __ATTRS_o_ai
vec_vpkuwum(vector bool int __a, vector bool int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool short)vec_perm(__a, __b, (vector unsigned char)
    (0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0C, 0x0D,
     0x10, 0x11, 0x14, 0x15, 0x18, 0x19, 0x1C, 0x1D));
#else
  return (vector bool short)vec_perm(__a, __b, (vector unsigned char)
    (0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F,
     0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F));
#endif
}

/* vec_packpx */

static vector pixel __attribute__((__always_inline__))
vec_packpx(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector pixel)__builtin_altivec_vpkpx(__b, __a);
#else
  return (vector pixel)__builtin_altivec_vpkpx(__a, __b);
#endif
}

/* vec_vpkpx */

static vector pixel __attribute__((__always_inline__))
vec_vpkpx(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return (vector pixel)__builtin_altivec_vpkpx(__b, __a);
#else
  return (vector pixel)__builtin_altivec_vpkpx(__a, __b);
#endif
}

/* vec_packs */

static vector signed char __ATTRS_o_ai
vec_packs(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkshss(__b, __a);
#else
  return __builtin_altivec_vpkshss(__a, __b);
#endif
}

static vector unsigned char __ATTRS_o_ai
vec_packs(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuhus(__b, __a);
#else
  return __builtin_altivec_vpkuhus(__a, __b);
#endif
}

static vector signed short __ATTRS_o_ai
vec_packs(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkswss(__b, __a);
#else
  return __builtin_altivec_vpkswss(__a, __b);
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_packs(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuwus(__b, __a);
#else
  return __builtin_altivec_vpkuwus(__a, __b);
#endif
}

/* vec_vpkshss */

static vector signed char __attribute__((__always_inline__))
vec_vpkshss(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkshss(__b, __a);
#else
  return __builtin_altivec_vpkshss(__a, __b);
#endif
}

/* vec_vpkuhus */

static vector unsigned char __attribute__((__always_inline__))
vec_vpkuhus(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuhus(__b, __a);
#else
  return __builtin_altivec_vpkuhus(__a, __b);
#endif
}

/* vec_vpkswss */

static vector signed short __attribute__((__always_inline__))
vec_vpkswss(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkswss(__b, __a);
#else
  return __builtin_altivec_vpkswss(__a, __b);
#endif
}

/* vec_vpkuwus */

static vector unsigned short __attribute__((__always_inline__))
vec_vpkuwus(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuwus(__b, __a);
#else
  return __builtin_altivec_vpkuwus(__a, __b);
#endif
}

/* vec_packsu */

static vector unsigned char __ATTRS_o_ai
vec_packsu(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkshus(__b, __a);
#else
  return __builtin_altivec_vpkshus(__a, __b);
#endif
}

static vector unsigned char __ATTRS_o_ai
vec_packsu(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuhus(__b, __a);
#else
  return __builtin_altivec_vpkuhus(__a, __b);
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_packsu(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkswus(__b, __a);
#else
  return __builtin_altivec_vpkswus(__a, __b);
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_packsu(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuwus(__b, __a);
#else
  return __builtin_altivec_vpkuwus(__a, __b);
#endif
}

/* vec_vpkshus */

static vector unsigned char __ATTRS_o_ai
vec_vpkshus(vector short __a, vector short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkshus(__b, __a);
#else
  return __builtin_altivec_vpkshus(__a, __b);
#endif
}

static vector unsigned char __ATTRS_o_ai
vec_vpkshus(vector unsigned short __a, vector unsigned short __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuhus(__b, __a);
#else
  return __builtin_altivec_vpkuhus(__a, __b);
#endif
}

/* vec_vpkswus */

static vector unsigned short __ATTRS_o_ai
vec_vpkswus(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkswus(__b, __a);
#else
  return __builtin_altivec_vpkswus(__a, __b);
#endif
}

static vector unsigned short __ATTRS_o_ai
vec_vpkswus(vector unsigned int __a, vector unsigned int __b)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vpkuwus(__b, __a);
#else
  return __builtin_altivec_vpkuwus(__a, __b);
#endif
}

/* vec_perm */

// The vperm instruction is defined architecturally with a big-endian bias.
// For little endian, we swap the input operands and invert the permute
// control vector.  Only the rightmost 5 bits matter, so we could use
// a vector of all 31s instead of all 255s to perform the inversion.
// However, when the PCV is not a constant, using 255 has an advantage
// in that the vec_xor can be recognized as a vec_nor (and for P8 and
// later, possibly a vec_nand).

vector signed char __ATTRS_o_ai
vec_perm(vector signed char __a, vector signed char __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector signed char)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector signed char)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector unsigned char __ATTRS_o_ai
vec_perm(vector unsigned char __a,
         vector unsigned char __b,
         vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector unsigned char)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector unsigned char)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector bool char __ATTRS_o_ai
vec_perm(vector bool char __a, vector bool char __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector bool char)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector bool char)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector short __ATTRS_o_ai
vec_perm(vector short __a, vector short __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector short)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector short)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector unsigned short __ATTRS_o_ai
vec_perm(vector unsigned short __a,
         vector unsigned short __b,
         vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector unsigned short)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector unsigned short)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector bool short __ATTRS_o_ai
vec_perm(vector bool short __a, vector bool short __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector bool short)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector bool short)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector pixel __ATTRS_o_ai
vec_perm(vector pixel __a, vector pixel __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector pixel)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector pixel)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector int __ATTRS_o_ai
vec_perm(vector int __a, vector int __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector int)__builtin_altivec_vperm_4si(__b, __a, __d);
#else
  return (vector int)__builtin_altivec_vperm_4si(__a, __b, __c);
#endif
}

vector unsigned int __ATTRS_o_ai
vec_perm(vector unsigned int __a, vector unsigned int __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector unsigned int)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector unsigned int)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector bool int __ATTRS_o_ai
vec_perm(vector bool int __a, vector bool int __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector bool int)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector bool int)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

vector float __ATTRS_o_ai
vec_perm(vector float __a, vector float __b, vector unsigned char __c)
{
#ifdef __LITTLE_ENDIAN__
  vector unsigned char __d = {255,255,255,255,255,255,255,255,
                              255,255,255,255,255,255,255,255};
  __d = vec_xor(__c, __d);
  return (vector float)
           __builtin_altivec_vperm_4si((vector int)__b, (vector int)__a, __d);
#else
  return (vector float)
           __builtin_altivec_vperm_4si((vector int)__a, (vector int)__b, __c);
#endif
}

/* vec_vperm */

static vector signed char __ATTRS_o_ai
vec_vperm(vector signed char __a, vector signed char __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector unsigned char __ATTRS_o_ai
vec_vperm(vector unsigned char __a,
          vector unsigned char __b,
          vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector bool char __ATTRS_o_ai
vec_vperm(vector bool char __a, vector bool char __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector short __ATTRS_o_ai
vec_vperm(vector short __a, vector short __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector unsigned short __ATTRS_o_ai
vec_vperm(vector unsigned short __a,
          vector unsigned short __b,
          vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector bool short __ATTRS_o_ai
vec_vperm(vector bool short __a, vector bool short __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector pixel __ATTRS_o_ai
vec_vperm(vector pixel __a, vector pixel __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector int __ATTRS_o_ai
vec_vperm(vector int __a, vector int __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector unsigned int __ATTRS_o_ai
vec_vperm(vector unsigned int __a, vector unsigned int __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector bool int __ATTRS_o_ai
vec_vperm(vector bool int __a, vector bool int __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

static vector float __ATTRS_o_ai
vec_vperm(vector float __a, vector float __b, vector unsigned char __c)
{
  return vec_perm(__a, __b, __c);
}

/* vec_re */

static vector float __attribute__((__always_inline__))
vec_re(vector float __a)
{
  return __builtin_altivec_vrefp(__a);
}

/* vec_vrefp */

static vector float __attribute__((__always_inline__))
vec_vrefp(vector float __a)
{
  return __builtin_altivec_vrefp(__a);
}

/* vec_rl */

static vector signed char __ATTRS_o_ai
vec_rl(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)__builtin_altivec_vrlb((vector char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_rl(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)__builtin_altivec_vrlb((vector char)__a, __b);
}

static vector short __ATTRS_o_ai
vec_rl(vector short __a, vector unsigned short __b)
{
  return __builtin_altivec_vrlh(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_rl(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)__builtin_altivec_vrlh((vector short)__a, __b);
}

static vector int __ATTRS_o_ai
vec_rl(vector int __a, vector unsigned int __b)
{
  return __builtin_altivec_vrlw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_rl(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)__builtin_altivec_vrlw((vector int)__a, __b);
}

/* vec_vrlb */

static vector signed char __ATTRS_o_ai
vec_vrlb(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)__builtin_altivec_vrlb((vector char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vrlb(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)__builtin_altivec_vrlb((vector char)__a, __b);
}

/* vec_vrlh */

static vector short __ATTRS_o_ai
vec_vrlh(vector short __a, vector unsigned short __b)
{
  return __builtin_altivec_vrlh(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vrlh(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)__builtin_altivec_vrlh((vector short)__a, __b);
}

/* vec_vrlw */

static vector int __ATTRS_o_ai
vec_vrlw(vector int __a, vector unsigned int __b)
{
  return __builtin_altivec_vrlw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vrlw(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)__builtin_altivec_vrlw((vector int)__a, __b);
}

/* vec_round */

static vector float __attribute__((__always_inline__))
vec_round(vector float __a)
{
  return __builtin_altivec_vrfin(__a);
}

/* vec_vrfin */

static vector float __attribute__((__always_inline__))
vec_vrfin(vector float __a)
{
  return __builtin_altivec_vrfin(__a);
}

/* vec_rsqrte */

static __vector float __attribute__((__always_inline__))
vec_rsqrte(vector float __a)
{
  return __builtin_altivec_vrsqrtefp(__a);
}

/* vec_vrsqrtefp */

static __vector float __attribute__((__always_inline__))
vec_vrsqrtefp(vector float __a)
{
  return __builtin_altivec_vrsqrtefp(__a);
}

/* vec_sel */

#define __builtin_altivec_vsel_4si vec_sel

static vector signed char __ATTRS_o_ai
vec_sel(vector signed char __a, vector signed char __b, vector unsigned char __c)
{
  return (__a & ~(vector signed char)__c) | (__b & (vector signed char)__c);
}

static vector signed char __ATTRS_o_ai
vec_sel(vector signed char __a, vector signed char __b, vector bool char __c)
{
  return (__a & ~(vector signed char)__c) | (__b & (vector signed char)__c);
}

static vector unsigned char __ATTRS_o_ai
vec_sel(vector unsigned char __a, vector unsigned char __b, vector unsigned char __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector unsigned char __ATTRS_o_ai
vec_sel(vector unsigned char __a, vector unsigned char __b, vector bool char __c)
{
  return (__a & ~(vector unsigned char)__c) | (__b & (vector unsigned char)__c);
}

static vector bool char __ATTRS_o_ai
vec_sel(vector bool char __a, vector bool char __b, vector unsigned char __c)
{
  return (__a & ~(vector bool char)__c) | (__b & (vector bool char)__c);
}

static vector bool char __ATTRS_o_ai
vec_sel(vector bool char __a, vector bool char __b, vector bool char __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector short __ATTRS_o_ai
vec_sel(vector short __a, vector short __b, vector unsigned short __c)
{
  return (__a & ~(vector short)__c) | (__b & (vector short)__c);
}

static vector short __ATTRS_o_ai
vec_sel(vector short __a, vector short __b, vector bool short __c)
{
  return (__a & ~(vector short)__c) | (__b & (vector short)__c);
}

static vector unsigned short __ATTRS_o_ai
vec_sel(vector unsigned short __a,
        vector unsigned short __b,
        vector unsigned short __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector unsigned short __ATTRS_o_ai
vec_sel(vector unsigned short __a, vector unsigned short __b, vector bool short __c)
{
  return (__a & ~(vector unsigned short)__c) | (__b & (vector unsigned short)__c);
}

static vector bool short __ATTRS_o_ai
vec_sel(vector bool short __a, vector bool short __b, vector unsigned short __c)
{
  return (__a & ~(vector bool short)__c) | (__b & (vector bool short)__c);
}

static vector bool short __ATTRS_o_ai
vec_sel(vector bool short __a, vector bool short __b, vector bool short __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector int __ATTRS_o_ai
vec_sel(vector int __a, vector int __b, vector unsigned int __c)
{
  return (__a & ~(vector int)__c) | (__b & (vector int)__c);
}

static vector int __ATTRS_o_ai
vec_sel(vector int __a, vector int __b, vector bool int __c)
{
  return (__a & ~(vector int)__c) | (__b & (vector int)__c);
}

static vector unsigned int __ATTRS_o_ai
vec_sel(vector unsigned int __a, vector unsigned int __b, vector unsigned int __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector unsigned int __ATTRS_o_ai
vec_sel(vector unsigned int __a, vector unsigned int __b, vector bool int __c)
{
  return (__a & ~(vector unsigned int)__c) | (__b & (vector unsigned int)__c);
}

static vector bool int __ATTRS_o_ai
vec_sel(vector bool int __a, vector bool int __b, vector unsigned int __c)
{
  return (__a & ~(vector bool int)__c) | (__b & (vector bool int)__c);
}

static vector bool int __ATTRS_o_ai
vec_sel(vector bool int __a, vector bool int __b, vector bool int __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector float __ATTRS_o_ai
vec_sel(vector float __a, vector float __b, vector unsigned int __c)
{
  vector int __res = ((vector int)__a & ~(vector int)__c)
                   | ((vector int)__b & (vector int)__c);
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_sel(vector float __a, vector float __b, vector bool int __c)
{
  vector int __res = ((vector int)__a & ~(vector int)__c)
                   | ((vector int)__b & (vector int)__c);
  return (vector float)__res;
}

/* vec_vsel */

static vector signed char __ATTRS_o_ai
vec_vsel(vector signed char __a, vector signed char __b, vector unsigned char __c)
{
  return (__a & ~(vector signed char)__c) | (__b & (vector signed char)__c);
}

static vector signed char __ATTRS_o_ai
vec_vsel(vector signed char __a, vector signed char __b, vector bool char __c)
{
  return (__a & ~(vector signed char)__c) | (__b & (vector signed char)__c);
}

static vector unsigned char __ATTRS_o_ai
vec_vsel(vector unsigned char __a, vector unsigned char __b, vector unsigned char __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector unsigned char __ATTRS_o_ai
vec_vsel(vector unsigned char __a, vector unsigned char __b, vector bool char __c)
{
  return (__a & ~(vector unsigned char)__c) | (__b & (vector unsigned char)__c);
}

static vector bool char __ATTRS_o_ai
vec_vsel(vector bool char __a, vector bool char __b, vector unsigned char __c)
{
  return (__a & ~(vector bool char)__c) | (__b & (vector bool char)__c);
}

static vector bool char __ATTRS_o_ai
vec_vsel(vector bool char __a, vector bool char __b, vector bool char __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector short __ATTRS_o_ai
vec_vsel(vector short __a, vector short __b, vector unsigned short __c)
{
  return (__a & ~(vector short)__c) | (__b & (vector short)__c);
}

static vector short __ATTRS_o_ai
vec_vsel(vector short __a, vector short __b, vector bool short __c)
{
  return (__a & ~(vector short)__c) | (__b & (vector short)__c);
}

static vector unsigned short __ATTRS_o_ai
vec_vsel(vector unsigned short __a,
         vector unsigned short __b,
         vector unsigned short __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector unsigned short __ATTRS_o_ai
vec_vsel(vector unsigned short __a, vector unsigned short __b, vector bool short __c)
{
  return (__a & ~(vector unsigned short)__c) | (__b & (vector unsigned short)__c);
}

static vector bool short __ATTRS_o_ai
vec_vsel(vector bool short __a, vector bool short __b, vector unsigned short __c)
{
  return (__a & ~(vector bool short)__c) | (__b & (vector bool short)__c);
}

static vector bool short __ATTRS_o_ai
vec_vsel(vector bool short __a, vector bool short __b, vector bool short __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector int __ATTRS_o_ai
vec_vsel(vector int __a, vector int __b, vector unsigned int __c)
{
  return (__a & ~(vector int)__c) | (__b & (vector int)__c);
}

static vector int __ATTRS_o_ai
vec_vsel(vector int __a, vector int __b, vector bool int __c)
{
  return (__a & ~(vector int)__c) | (__b & (vector int)__c);
}

static vector unsigned int __ATTRS_o_ai
vec_vsel(vector unsigned int __a, vector unsigned int __b, vector unsigned int __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector unsigned int __ATTRS_o_ai
vec_vsel(vector unsigned int __a, vector unsigned int __b, vector bool int __c)
{
  return (__a & ~(vector unsigned int)__c) | (__b & (vector unsigned int)__c);
}

static vector bool int __ATTRS_o_ai
vec_vsel(vector bool int __a, vector bool int __b, vector unsigned int __c)
{
  return (__a & ~(vector bool int)__c) | (__b & (vector bool int)__c);
}

static vector bool int __ATTRS_o_ai
vec_vsel(vector bool int __a, vector bool int __b, vector bool int __c)
{
  return (__a & ~__c) | (__b & __c);
}

static vector float __ATTRS_o_ai
vec_vsel(vector float __a, vector float __b, vector unsigned int __c)
{
  vector int __res = ((vector int)__a & ~(vector int)__c)
                   | ((vector int)__b & (vector int)__c);
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vsel(vector float __a, vector float __b, vector bool int __c)
{
  vector int __res = ((vector int)__a & ~(vector int)__c)
                   | ((vector int)__b & (vector int)__c);
  return (vector float)__res;
}

/* vec_sl */

static vector signed char __ATTRS_o_ai
vec_sl(vector signed char __a, vector unsigned char __b)
{
  return __a << (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_sl(vector unsigned char __a, vector unsigned char __b)
{
  return __a << __b;
}

static vector short __ATTRS_o_ai
vec_sl(vector short __a, vector unsigned short __b)
{
  return __a << (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_sl(vector unsigned short __a, vector unsigned short __b)
{
  return __a << __b;
}

static vector int __ATTRS_o_ai
vec_sl(vector int __a, vector unsigned int __b)
{
  return __a << (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_sl(vector unsigned int __a, vector unsigned int __b)
{
  return __a << __b;
}

/* vec_vslb */

#define __builtin_altivec_vslb vec_vslb

static vector signed char __ATTRS_o_ai
vec_vslb(vector signed char __a, vector unsigned char __b)
{
  return vec_sl(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vslb(vector unsigned char __a, vector unsigned char __b)
{
  return vec_sl(__a, __b);
}

/* vec_vslh */

#define __builtin_altivec_vslh vec_vslh

static vector short __ATTRS_o_ai
vec_vslh(vector short __a, vector unsigned short __b)
{
  return vec_sl(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vslh(vector unsigned short __a, vector unsigned short __b)
{
  return vec_sl(__a, __b);
}

/* vec_vslw */

#define __builtin_altivec_vslw vec_vslw

static vector int __ATTRS_o_ai
vec_vslw(vector int __a, vector unsigned int __b)
{
  return vec_sl(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vslw(vector unsigned int __a, vector unsigned int __b)
{
  return vec_sl(__a, __b);
}

/* vec_sld */

#define __builtin_altivec_vsldoi_4si vec_sld

static vector signed char __ATTRS_o_ai
vec_sld(vector signed char __a, vector signed char __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector unsigned char __ATTRS_o_ai
vec_sld(vector unsigned char __a, vector unsigned char __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector short __ATTRS_o_ai
vec_sld(vector short __a, vector short __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector unsigned short __ATTRS_o_ai
vec_sld(vector unsigned short __a, vector unsigned short __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector pixel __ATTRS_o_ai
vec_sld(vector pixel __a, vector pixel __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector int __ATTRS_o_ai
vec_sld(vector int __a, vector int __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector unsigned int __ATTRS_o_ai
vec_sld(vector unsigned int __a, vector unsigned int __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector float __ATTRS_o_ai
vec_sld(vector float __a, vector float __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

/* vec_vsldoi */

static vector signed char __ATTRS_o_ai
vec_vsldoi(vector signed char __a, vector signed char __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector unsigned char __ATTRS_o_ai
vec_vsldoi(vector unsigned char __a, vector unsigned char __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector short __ATTRS_o_ai
vec_vsldoi(vector short __a, vector short __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector unsigned short __ATTRS_o_ai
vec_vsldoi(vector unsigned short __a, vector unsigned short __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector pixel __ATTRS_o_ai
vec_vsldoi(vector pixel __a, vector pixel __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector int __ATTRS_o_ai
vec_vsldoi(vector int __a, vector int __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector unsigned int __ATTRS_o_ai
vec_vsldoi(vector unsigned int __a, vector unsigned int __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

static vector float __ATTRS_o_ai
vec_vsldoi(vector float __a, vector float __b, unsigned char __c)
{
  return vec_perm(__a, __b, (vector unsigned char)
    (__c,   __c+1, __c+2,  __c+3,  __c+4,  __c+5,  __c+6,  __c+7,
     __c+8, __c+9, __c+10, __c+11, __c+12, __c+13, __c+14, __c+15));
}

/* vec_sll */

static vector signed char __ATTRS_o_ai
vec_sll(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_sll(vector signed char __a, vector unsigned short __b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_sll(vector signed char __a, vector unsigned int __b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_sll(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_sll(vector unsigned char __a, vector unsigned short __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_sll(vector unsigned char __a, vector unsigned int __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_sll(vector bool char __a, vector unsigned char __b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_sll(vector bool char __a, vector unsigned short __b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_sll(vector bool char __a, vector unsigned int __b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_sll(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_sll(vector short __a, vector unsigned short __b)
{
  return (vector short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_sll(vector short __a, vector unsigned int __b)
{
  return (vector short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_sll(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_sll(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_sll(vector unsigned short __a, vector unsigned int __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_sll(vector bool short __a, vector unsigned char __b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_sll(vector bool short __a, vector unsigned short __b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_sll(vector bool short __a, vector unsigned int __b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_sll(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_sll(vector pixel __a, vector unsigned short __b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_sll(vector pixel __a, vector unsigned int __b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_sll(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vsl(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_sll(vector int __a, vector unsigned short __b)
{
  return (vector int)__builtin_altivec_vsl(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_sll(vector int __a, vector unsigned int __b)
{
  return (vector int)__builtin_altivec_vsl(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_sll(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_sll(vector unsigned int __a, vector unsigned short __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_sll(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_sll(vector bool int __a, vector unsigned char __b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_sll(vector bool int __a, vector unsigned short __b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_sll(vector bool int __a, vector unsigned int __b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

/* vec_vsl */

static vector signed char __ATTRS_o_ai
vec_vsl(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_vsl(vector signed char __a, vector unsigned short __b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_vsl(vector signed char __a, vector unsigned int __b)
{
  return (vector signed char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsl(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsl(vector unsigned char __a, vector unsigned short __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsl(vector unsigned char __a, vector unsigned int __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_vsl(vector bool char __a, vector unsigned char __b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_vsl(vector bool char __a, vector unsigned short __b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_vsl(vector bool char __a, vector unsigned int __b)
{
  return (vector bool char)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsl(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsl(vector short __a, vector unsigned short __b)
{
  return (vector short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsl(vector short __a, vector unsigned int __b)
{
  return (vector short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsl(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsl(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsl(vector unsigned short __a, vector unsigned int __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_vsl(vector bool short __a, vector unsigned char __b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_vsl(vector bool short __a, vector unsigned short __b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_vsl(vector bool short __a, vector unsigned int __b)
{
  return (vector bool short)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsl(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsl(vector pixel __a, vector unsigned short __b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsl(vector pixel __a, vector unsigned int __b)
{
  return (vector pixel)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsl(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vsl(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsl(vector int __a, vector unsigned short __b)
{
  return (vector int)__builtin_altivec_vsl(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsl(vector int __a, vector unsigned int __b)
{
  return (vector int)__builtin_altivec_vsl(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsl(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsl(vector unsigned int __a, vector unsigned short __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsl(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_vsl(vector bool int __a, vector unsigned char __b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_vsl(vector bool int __a, vector unsigned short __b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_vsl(vector bool int __a, vector unsigned int __b)
{
  return (vector bool int)__builtin_altivec_vsl((vector int)__a, (vector int)__b);
}

/* vec_slo */

static vector signed char __ATTRS_o_ai
vec_slo(vector signed char __a, vector signed char __b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_slo(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_slo(vector unsigned char __a, vector signed char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_slo(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_slo(vector short __a, vector signed char __b)
{
  return (vector short)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_slo(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_slo(vector unsigned short __a, vector signed char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_slo(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_slo(vector pixel __a, vector signed char __b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_slo(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_slo(vector int __a, vector signed char __b)
{
  return (vector int)__builtin_altivec_vslo(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_slo(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vslo(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_slo(vector unsigned int __a, vector signed char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_slo(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_slo(vector float __a, vector signed char __b)
{
  return (vector float)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_slo(vector float __a, vector unsigned char __b)
{
  return (vector float)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

/* vec_vslo */

static vector signed char __ATTRS_o_ai
vec_vslo(vector signed char __a, vector signed char __b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_vslo(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vslo(vector unsigned char __a, vector signed char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vslo(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vslo(vector short __a, vector signed char __b)
{
  return (vector short)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vslo(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vslo(vector unsigned short __a, vector signed char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vslo(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vslo(vector pixel __a, vector signed char __b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vslo(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vslo(vector int __a, vector signed char __b)
{
  return (vector int)__builtin_altivec_vslo(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vslo(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vslo(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vslo(vector unsigned int __a, vector signed char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vslo(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_vslo(vector float __a, vector signed char __b)
{
  return (vector float)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_vslo(vector float __a, vector unsigned char __b)
{
  return (vector float)__builtin_altivec_vslo((vector int)__a, (vector int)__b);
}

/* vec_splat */

static vector signed char __ATTRS_o_ai
vec_splat(vector signed char __a, unsigned char __b)
{
  return vec_perm(__a, __a, (vector unsigned char)(__b));
}

static vector unsigned char __ATTRS_o_ai
vec_splat(vector unsigned char __a, unsigned char __b)
{
  return vec_perm(__a, __a, (vector unsigned char)(__b));
}

static vector bool char __ATTRS_o_ai
vec_splat(vector bool char __a, unsigned char __b)
{
  return vec_perm(__a, __a, (vector unsigned char)(__b));
}

static vector short __ATTRS_o_ai
vec_splat(vector short __a, unsigned char __b)
{ 
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

static vector unsigned short __ATTRS_o_ai
vec_splat(vector unsigned short __a, unsigned char __b)
{ 
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

static vector bool short __ATTRS_o_ai
vec_splat(vector bool short __a, unsigned char __b)
{ 
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

static vector pixel __ATTRS_o_ai
vec_splat(vector pixel __a, unsigned char __b)
{ 
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

static vector int __ATTRS_o_ai
vec_splat(vector int __a, unsigned char __b)
{ 
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

static vector unsigned int __ATTRS_o_ai
vec_splat(vector unsigned int __a, unsigned char __b)
{ 
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

static vector bool int __ATTRS_o_ai
vec_splat(vector bool int __a, unsigned char __b)
{ 
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

static vector float __ATTRS_o_ai
vec_splat(vector float __a, unsigned char __b)
{ 
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

/* vec_vspltb */

#define __builtin_altivec_vspltb vec_vspltb

static vector signed char __ATTRS_o_ai
vec_vspltb(vector signed char __a, unsigned char __b)
{
  return vec_perm(__a, __a, (vector unsigned char)(__b));
}

static vector unsigned char __ATTRS_o_ai
vec_vspltb(vector unsigned char __a, unsigned char __b)
{
  return vec_perm(__a, __a, (vector unsigned char)(__b));
}

static vector bool char __ATTRS_o_ai
vec_vspltb(vector bool char __a, unsigned char __b)
{
  return vec_perm(__a, __a, (vector unsigned char)(__b));
}

/* vec_vsplth */

#define __builtin_altivec_vsplth vec_vsplth

static vector short __ATTRS_o_ai
vec_vsplth(vector short __a, unsigned char __b)
{
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

static vector unsigned short __ATTRS_o_ai
vec_vsplth(vector unsigned short __a, unsigned char __b)
{
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

static vector bool short __ATTRS_o_ai
vec_vsplth(vector bool short __a, unsigned char __b)
{
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

static vector pixel __ATTRS_o_ai
vec_vsplth(vector pixel __a, unsigned char __b)
{
  __b *= 2;
  unsigned char b1=__b+1;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1, __b, b1));
}

/* vec_vspltw */

#define __builtin_altivec_vspltw vec_vspltw

static vector int __ATTRS_o_ai
vec_vspltw(vector int __a, unsigned char __b)
{
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

static vector unsigned int __ATTRS_o_ai
vec_vspltw(vector unsigned int __a, unsigned char __b)
{
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

static vector bool int __ATTRS_o_ai
vec_vspltw(vector bool int __a, unsigned char __b)
{
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

static vector float __ATTRS_o_ai
vec_vspltw(vector float __a, unsigned char __b)
{
  __b *= 4;
  unsigned char b1=__b+1, b2=__b+2, b3=__b+3;
  return vec_perm(__a, __a, (vector unsigned char)
    (__b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3, __b, b1, b2, b3));
}

/* vec_splat_s8 */

#define __builtin_altivec_vspltisb vec_splat_s8

// FIXME: parameter should be treated as 5-bit signed literal
static vector signed char __ATTRS_o_ai
vec_splat_s8(signed char __a)
{
  return (vector signed char)(__a);
}

/* vec_vspltisb */

// FIXME: parameter should be treated as 5-bit signed literal
static vector signed char __ATTRS_o_ai
vec_vspltisb(signed char __a)
{
  return (vector signed char)(__a);
}

/* vec_splat_s16 */

#define __builtin_altivec_vspltish vec_splat_s16

// FIXME: parameter should be treated as 5-bit signed literal
static vector short __ATTRS_o_ai
vec_splat_s16(signed char __a)
{
  return (vector short)(__a);
}

/* vec_vspltish */

// FIXME: parameter should be treated as 5-bit signed literal
static vector short __ATTRS_o_ai
vec_vspltish(signed char __a)
{
  return (vector short)(__a);
}

/* vec_splat_s32 */

#define __builtin_altivec_vspltisw vec_splat_s32

// FIXME: parameter should be treated as 5-bit signed literal
static vector int __ATTRS_o_ai
vec_splat_s32(signed char __a)
{
  return (vector int)(__a);
}

/* vec_vspltisw */

// FIXME: parameter should be treated as 5-bit signed literal
static vector int __ATTRS_o_ai
vec_vspltisw(signed char __a)
{
  return (vector int)(__a);
}

/* vec_splat_u8 */

// FIXME: parameter should be treated as 5-bit signed literal
static vector unsigned char __ATTRS_o_ai
vec_splat_u8(unsigned char __a)
{
  return (vector unsigned char)(__a);
}

/* vec_splat_u16 */

// FIXME: parameter should be treated as 5-bit signed literal
static vector unsigned short __ATTRS_o_ai
vec_splat_u16(signed char __a)
{
  return (vector unsigned short)(__a);
}

/* vec_splat_u32 */

// FIXME: parameter should be treated as 5-bit signed literal
static vector unsigned int __ATTRS_o_ai
vec_splat_u32(signed char __a)
{
  return (vector unsigned int)(__a);
}

/* vec_sr */

static vector signed char __ATTRS_o_ai
vec_sr(vector signed char __a, vector unsigned char __b)
{
  return __a >> (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_sr(vector unsigned char __a, vector unsigned char __b)
{
  return __a >> __b;
}

static vector short __ATTRS_o_ai
vec_sr(vector short __a, vector unsigned short __b)
{
  return __a >> (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_sr(vector unsigned short __a, vector unsigned short __b)
{
  return __a >> __b;
}

static vector int __ATTRS_o_ai
vec_sr(vector int __a, vector unsigned int __b)
{
  return __a >> (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_sr(vector unsigned int __a, vector unsigned int __b)
{
  return __a >> __b;
}

/* vec_vsrb */

#define __builtin_altivec_vsrb vec_vsrb

static vector signed char __ATTRS_o_ai
vec_vsrb(vector signed char __a, vector unsigned char __b)
{
  return __a >> (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsrb(vector unsigned char __a, vector unsigned char __b)
{
  return __a >> __b;
}

/* vec_vsrh */

#define __builtin_altivec_vsrh vec_vsrh

static vector short __ATTRS_o_ai
vec_vsrh(vector short __a, vector unsigned short __b)
{
  return __a >> (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsrh(vector unsigned short __a, vector unsigned short __b)
{
  return __a >> __b;
}

/* vec_vsrw */

#define __builtin_altivec_vsrw vec_vsrw

static vector int __ATTRS_o_ai
vec_vsrw(vector int __a, vector unsigned int __b)
{
  return __a >> (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsrw(vector unsigned int __a, vector unsigned int __b)
{
  return __a >> __b;
}

/* vec_sra */

static vector signed char __ATTRS_o_ai
vec_sra(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)__builtin_altivec_vsrab((vector char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_sra(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)__builtin_altivec_vsrab((vector char)__a, __b);
}

static vector short __ATTRS_o_ai
vec_sra(vector short __a, vector unsigned short __b)
{
  return __builtin_altivec_vsrah(__a, (vector unsigned short)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_sra(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)__builtin_altivec_vsrah((vector short)__a, __b);
}

static vector int __ATTRS_o_ai
vec_sra(vector int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsraw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_sra(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)__builtin_altivec_vsraw((vector int)__a, __b);
}

/* vec_vsrab */

static vector signed char __ATTRS_o_ai
vec_vsrab(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)__builtin_altivec_vsrab((vector char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsrab(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)__builtin_altivec_vsrab((vector char)__a, __b);
}

/* vec_vsrah */

static vector short __ATTRS_o_ai
vec_vsrah(vector short __a, vector unsigned short __b)
{
  return __builtin_altivec_vsrah(__a, (vector unsigned short)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsrah(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)__builtin_altivec_vsrah((vector short)__a, __b);
}

/* vec_vsraw */

static vector int __ATTRS_o_ai
vec_vsraw(vector int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsraw(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsraw(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)__builtin_altivec_vsraw((vector int)__a, __b);
}

/* vec_srl */

static vector signed char __ATTRS_o_ai
vec_srl(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_srl(vector signed char __a, vector unsigned short __b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_srl(vector signed char __a, vector unsigned int __b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_srl(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_srl(vector unsigned char __a, vector unsigned short __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_srl(vector unsigned char __a, vector unsigned int __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_srl(vector bool char __a, vector unsigned char __b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_srl(vector bool char __a, vector unsigned short __b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_srl(vector bool char __a, vector unsigned int __b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_srl(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_srl(vector short __a, vector unsigned short __b)
{
  return (vector short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_srl(vector short __a, vector unsigned int __b)
{
  return (vector short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_srl(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_srl(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_srl(vector unsigned short __a, vector unsigned int __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_srl(vector bool short __a, vector unsigned char __b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_srl(vector bool short __a, vector unsigned short __b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_srl(vector bool short __a, vector unsigned int __b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_srl(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_srl(vector pixel __a, vector unsigned short __b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_srl(vector pixel __a, vector unsigned int __b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_srl(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vsr(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_srl(vector int __a, vector unsigned short __b)
{
  return (vector int)__builtin_altivec_vsr(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_srl(vector int __a, vector unsigned int __b)
{
  return (vector int)__builtin_altivec_vsr(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_srl(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_srl(vector unsigned int __a, vector unsigned short __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_srl(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_srl(vector bool int __a, vector unsigned char __b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_srl(vector bool int __a, vector unsigned short __b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_srl(vector bool int __a, vector unsigned int __b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

/* vec_vsr */

static vector signed char __ATTRS_o_ai
vec_vsr(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_vsr(vector signed char __a, vector unsigned short __b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_vsr(vector signed char __a, vector unsigned int __b)
{
  return (vector signed char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsr(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsr(vector unsigned char __a, vector unsigned short __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsr(vector unsigned char __a, vector unsigned int __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_vsr(vector bool char __a, vector unsigned char __b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_vsr(vector bool char __a, vector unsigned short __b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool char __ATTRS_o_ai
vec_vsr(vector bool char __a, vector unsigned int __b)
{
  return (vector bool char)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsr(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsr(vector short __a, vector unsigned short __b)
{
  return (vector short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsr(vector short __a, vector unsigned int __b)
{
  return (vector short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsr(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsr(vector unsigned short __a, vector unsigned short __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsr(vector unsigned short __a, vector unsigned int __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_vsr(vector bool short __a, vector unsigned char __b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_vsr(vector bool short __a, vector unsigned short __b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool short __ATTRS_o_ai
vec_vsr(vector bool short __a, vector unsigned int __b)
{
  return (vector bool short)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsr(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsr(vector pixel __a, vector unsigned short __b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsr(vector pixel __a, vector unsigned int __b)
{
  return (vector pixel)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsr(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vsr(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsr(vector int __a, vector unsigned short __b)
{
  return (vector int)__builtin_altivec_vsr(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsr(vector int __a, vector unsigned int __b)
{
  return (vector int)__builtin_altivec_vsr(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsr(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsr(vector unsigned int __a, vector unsigned short __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsr(vector unsigned int __a, vector unsigned int __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_vsr(vector bool int __a, vector unsigned char __b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_vsr(vector bool int __a, vector unsigned short __b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

static vector bool int __ATTRS_o_ai
vec_vsr(vector bool int __a, vector unsigned int __b)
{
  return (vector bool int)__builtin_altivec_vsr((vector int)__a, (vector int)__b);
}

/* vec_sro */

static vector signed char __ATTRS_o_ai
vec_sro(vector signed char __a, vector signed char __b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_sro(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_sro(vector unsigned char __a, vector signed char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_sro(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_sro(vector short __a, vector signed char __b)
{
  return (vector short)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_sro(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_sro(vector unsigned short __a, vector signed char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_sro(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_sro(vector pixel __a, vector signed char __b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_sro(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_sro(vector int __a, vector signed char __b)
{
  return (vector int)__builtin_altivec_vsro(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_sro(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vsro(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_sro(vector unsigned int __a, vector signed char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_sro(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_sro(vector float __a, vector signed char __b)
{
  return (vector float)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_sro(vector float __a, vector unsigned char __b)
{
  return (vector float)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

/* vec_vsro */

static vector signed char __ATTRS_o_ai
vec_vsro(vector signed char __a, vector signed char __b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector signed char __ATTRS_o_ai
vec_vsro(vector signed char __a, vector unsigned char __b)
{
  return (vector signed char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsro(vector unsigned char __a, vector signed char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsro(vector unsigned char __a, vector unsigned char __b)
{
  return (vector unsigned char)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsro(vector short __a, vector signed char __b)
{
  return (vector short)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector short __ATTRS_o_ai
vec_vsro(vector short __a, vector unsigned char __b)
{
  return (vector short)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsro(vector unsigned short __a, vector signed char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsro(vector unsigned short __a, vector unsigned char __b)
{
  return (vector unsigned short)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsro(vector pixel __a, vector signed char __b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector pixel __ATTRS_o_ai
vec_vsro(vector pixel __a, vector unsigned char __b)
{
  return (vector pixel)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsro(vector int __a, vector signed char __b)
{
  return (vector int)__builtin_altivec_vsro(__a, (vector int)__b);
}

static vector int __ATTRS_o_ai
vec_vsro(vector int __a, vector unsigned char __b)
{
  return (vector int)__builtin_altivec_vsro(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsro(vector unsigned int __a, vector signed char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsro(vector unsigned int __a, vector unsigned char __b)
{
  return (vector unsigned int)
           __builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_vsro(vector float __a, vector signed char __b)
{
  return (vector float)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

static vector float __ATTRS_o_ai
vec_vsro(vector float __a, vector unsigned char __b)
{
  return (vector float)__builtin_altivec_vsro((vector int)__a, (vector int)__b);
}

/* vec_st */

static void __ATTRS_o_ai
vec_st(vector signed char __a, int __b, vector signed char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector signed char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool char __a, int __b, vector bool char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector short __a, int __b, vector short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector short __a, int __b, short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool short __a, int __b, short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool short __a, int __b, vector bool short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector pixel __a, int __b, short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector pixel __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector pixel __a, int __b, vector pixel *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector int __a, int __b, vector int *__c)
{
  __builtin_altivec_stvx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector int __a, int __b, int *__c)
{
  __builtin_altivec_stvx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector unsigned int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool int __a, int __b, int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector bool int __a, int __b, vector bool int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector float __a, int __b, vector float *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_st(vector float __a, int __b, float *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

/* vec_stvx */

static void __ATTRS_o_ai
vec_stvx(vector signed char __a, int __b, vector signed char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector signed char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool char __a, int __b, vector bool char *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector short __a, int __b, vector short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector short __a, int __b, short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool short __a, int __b, short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool short __a, int __b, vector bool short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector pixel __a, int __b, short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector pixel __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector pixel __a, int __b, vector pixel *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector int __a, int __b, vector int *__c)
{
  __builtin_altivec_stvx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector int __a, int __b, int *__c)
{
  __builtin_altivec_stvx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector unsigned int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool int __a, int __b, int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector bool int __a, int __b, vector bool int *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector float __a, int __b, vector float *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvx(vector float __a, int __b, float *__c)
{
  __builtin_altivec_stvx((vector int)__a, __b, __c);
}

/* vec_ste */

static void __ATTRS_o_ai
vec_ste(vector signed char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector unsigned char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector bool char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector bool char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector short __a, int __b, short *__c)
{
  __builtin_altivec_stvehx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector unsigned short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector bool short __a, int __b, short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector bool short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector pixel __a, int __b, short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector pixel __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector int __a, int __b, int *__c)
{
  __builtin_altivec_stvewx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector unsigned int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector bool int __a, int __b, int *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector bool int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_ste(vector float __a, int __b, float *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

/* vec_stvebx */

static void __ATTRS_o_ai
vec_stvebx(vector signed char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvebx(vector unsigned char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvebx(vector bool char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvebx(vector bool char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvebx((vector char)__a, __b, __c);
}

/* vec_stvehx */

static void __ATTRS_o_ai
vec_stvehx(vector short __a, int __b, short *__c)
{
  __builtin_altivec_stvehx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvehx(vector unsigned short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvehx(vector bool short __a, int __b, short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvehx(vector bool short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvehx(vector pixel __a, int __b, short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvehx(vector pixel __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvehx((vector short)__a, __b, __c);
}

/* vec_stvewx */

static void __ATTRS_o_ai
vec_stvewx(vector int __a, int __b, int *__c)
{
  __builtin_altivec_stvewx(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvewx(vector unsigned int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvewx(vector bool int __a, int __b, int *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvewx(vector bool int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvewx(vector float __a, int __b, float *__c)
{
  __builtin_altivec_stvewx((vector int)__a, __b, __c);
}

/* vec_stl */

static void __ATTRS_o_ai
vec_stl(vector signed char __a, int __b, vector signed char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector signed char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool char __a, int __b, vector bool char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector short __a, int __b, vector short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector short __a, int __b, short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool short __a, int __b, short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool short __a, int __b, vector bool short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector pixel __a, int __b, short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector pixel __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector pixel __a, int __b, vector pixel *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector int __a, int __b, vector int *__c)
{
  __builtin_altivec_stvxl(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector int __a, int __b, int *__c)
{
  __builtin_altivec_stvxl(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector unsigned int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool int __a, int __b, int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector bool int __a, int __b, vector bool int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector float __a, int __b, vector float *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stl(vector float __a, int __b, float *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

/* vec_stvxl */

static void __ATTRS_o_ai
vec_stvxl(vector signed char __a, int __b, vector signed char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector signed char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool char __a, int __b, signed char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool char __a, int __b, unsigned char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool char __a, int __b, vector bool char *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector short __a, int __b, vector short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector short __a, int __b, short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool short __a, int __b, short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool short __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool short __a, int __b, vector bool short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector pixel __a, int __b, short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector pixel __a, int __b, unsigned short *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector pixel __a, int __b, vector pixel *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector int __a, int __b, vector int *__c)
{
  __builtin_altivec_stvxl(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector int __a, int __b, int *__c)
{
  __builtin_altivec_stvxl(__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector unsigned int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool int __a, int __b, int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool int __a, int __b, unsigned int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector bool int __a, int __b, vector bool int *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector float __a, int __b, vector float *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

static void __ATTRS_o_ai
vec_stvxl(vector float __a, int __b, float *__c)
{
  __builtin_altivec_stvxl((vector int)__a, __b, __c);
}

/* vec_sub */

static vector signed char __ATTRS_o_ai
vec_sub(vector signed char __a, vector signed char __b)
{
  return __a - __b;
}

static vector signed char __ATTRS_o_ai
vec_sub(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a - __b;
}

static vector signed char __ATTRS_o_ai
vec_sub(vector signed char __a, vector bool char __b)
{
  return __a - (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_sub(vector unsigned char __a, vector unsigned char __b)
{
  return __a - __b;
}

static vector unsigned char __ATTRS_o_ai
vec_sub(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a - __b;
}

static vector unsigned char __ATTRS_o_ai
vec_sub(vector unsigned char __a, vector bool char __b)
{
  return __a - (vector unsigned char)__b;
}

static vector short __ATTRS_o_ai
vec_sub(vector short __a, vector short __b)
{
  return __a - __b;
}

static vector short __ATTRS_o_ai
vec_sub(vector bool short __a, vector short __b)
{
  return (vector short)__a - __b;
}

static vector short __ATTRS_o_ai
vec_sub(vector short __a, vector bool short __b)
{
  return __a - (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_sub(vector unsigned short __a, vector unsigned short __b)
{
  return __a - __b;
}

static vector unsigned short __ATTRS_o_ai
vec_sub(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a - __b;
}

static vector unsigned short __ATTRS_o_ai
vec_sub(vector unsigned short __a, vector bool short __b)
{
  return __a - (vector unsigned short)__b;
}

static vector int __ATTRS_o_ai
vec_sub(vector int __a, vector int __b)
{
  return __a - __b;
}

static vector int __ATTRS_o_ai
vec_sub(vector bool int __a, vector int __b)
{
  return (vector int)__a - __b;
}

static vector int __ATTRS_o_ai
vec_sub(vector int __a, vector bool int __b)
{
  return __a - (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_sub(vector unsigned int __a, vector unsigned int __b)
{
  return __a - __b;
}

static vector unsigned int __ATTRS_o_ai
vec_sub(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a - __b;
}

static vector unsigned int __ATTRS_o_ai
vec_sub(vector unsigned int __a, vector bool int __b)
{
  return __a - (vector unsigned int)__b;
}

static vector float __ATTRS_o_ai
vec_sub(vector float __a, vector float __b)
{
  return __a - __b;
}

/* vec_vsububm */

#define __builtin_altivec_vsububm vec_vsububm

static vector signed char __ATTRS_o_ai
vec_vsububm(vector signed char __a, vector signed char __b)
{
  return __a - __b;
}

static vector signed char __ATTRS_o_ai
vec_vsububm(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a - __b;
}

static vector signed char __ATTRS_o_ai
vec_vsububm(vector signed char __a, vector bool char __b)
{
  return __a - (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsububm(vector unsigned char __a, vector unsigned char __b)
{
  return __a - __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsububm(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a - __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vsububm(vector unsigned char __a, vector bool char __b)
{
  return __a - (vector unsigned char)__b;
}

/* vec_vsubuhm */

#define __builtin_altivec_vsubuhm vec_vsubuhm

static vector short __ATTRS_o_ai
vec_vsubuhm(vector short __a, vector short __b)
{
  return __a - __b;
}

static vector short __ATTRS_o_ai
vec_vsubuhm(vector bool short __a, vector short __b)
{
  return (vector short)__a - __b;
}

static vector short __ATTRS_o_ai
vec_vsubuhm(vector short __a, vector bool short __b)
{
  return __a - (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhm(vector unsigned short __a, vector unsigned short __b)
{
  return __a - __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhm(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a - __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhm(vector unsigned short __a, vector bool short __b)
{
  return __a - (vector unsigned short)__b;
}

/* vec_vsubuwm */

#define __builtin_altivec_vsubuwm vec_vsubuwm

static vector int __ATTRS_o_ai
vec_vsubuwm(vector int __a, vector int __b)
{
  return __a - __b;
}

static vector int __ATTRS_o_ai
vec_vsubuwm(vector bool int __a, vector int __b)
{
  return (vector int)__a - __b;
}

static vector int __ATTRS_o_ai
vec_vsubuwm(vector int __a, vector bool int __b)
{
  return __a - (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuwm(vector unsigned int __a, vector unsigned int __b)
{
  return __a - __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuwm(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a - __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuwm(vector unsigned int __a, vector bool int __b)
{
  return __a - (vector unsigned int)__b;
}

/* vec_vsubfp */

#define __builtin_altivec_vsubfp vec_vsubfp

static vector float __attribute__((__always_inline__))
vec_vsubfp(vector float __a, vector float __b)
{
  return __a - __b;
}

/* vec_subc */

static vector unsigned int __attribute__((__always_inline__))
vec_subc(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsubcuw(__a, __b);
}

/* vec_vsubcuw */

static vector unsigned int __attribute__((__always_inline__))
vec_vsubcuw(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsubcuw(__a, __b);
}

/* vec_subs */

static vector signed char __ATTRS_o_ai
vec_subs(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vsubsbs(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_subs(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vsubsbs((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_subs(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vsubsbs(__a, (vector signed char)__b);
}

static vector unsigned char __ATTRS_o_ai
vec_subs(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vsububs(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_subs(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vsububs((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_subs(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vsububs(__a, (vector unsigned char)__b);
}

static vector short __ATTRS_o_ai
vec_subs(vector short __a, vector short __b)
{
  return __builtin_altivec_vsubshs(__a, __b);
}

static vector short __ATTRS_o_ai
vec_subs(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vsubshs((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_subs(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vsubshs(__a, (vector short)__b);
}

static vector unsigned short __ATTRS_o_ai
vec_subs(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vsubuhs(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_subs(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vsubuhs((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_subs(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vsubuhs(__a, (vector unsigned short)__b);
}

static vector int __ATTRS_o_ai
vec_subs(vector int __a, vector int __b)
{
  return __builtin_altivec_vsubsws(__a, __b);
}

static vector int __ATTRS_o_ai
vec_subs(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vsubsws((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_subs(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vsubsws(__a, (vector int)__b);
}

static vector unsigned int __ATTRS_o_ai
vec_subs(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsubuws(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_subs(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsubuws((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_subs(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vsubuws(__a, (vector unsigned int)__b);
}

/* vec_vsubsbs */

static vector signed char __ATTRS_o_ai
vec_vsubsbs(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vsubsbs(__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vsubsbs(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vsubsbs((vector signed char)__a, __b);
}

static vector signed char __ATTRS_o_ai
vec_vsubsbs(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vsubsbs(__a, (vector signed char)__b);
}

/* vec_vsububs */

static vector unsigned char __ATTRS_o_ai
vec_vsububs(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vsububs(__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsububs(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vsububs((vector unsigned char)__a, __b);
}

static vector unsigned char __ATTRS_o_ai
vec_vsububs(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vsububs(__a, (vector unsigned char)__b);
}

/* vec_vsubshs */

static vector short __ATTRS_o_ai
vec_vsubshs(vector short __a, vector short __b)
{
  return __builtin_altivec_vsubshs(__a, __b);
}

static vector short __ATTRS_o_ai
vec_vsubshs(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vsubshs((vector short)__a, __b);
}

static vector short __ATTRS_o_ai
vec_vsubshs(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vsubshs(__a, (vector short)__b);
}

/* vec_vsubuhs */

static vector unsigned short __ATTRS_o_ai
vec_vsubuhs(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vsubuhs(__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhs(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vsubuhs((vector unsigned short)__a, __b);
}

static vector unsigned short __ATTRS_o_ai
vec_vsubuhs(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vsubuhs(__a, (vector unsigned short)__b);
}

/* vec_vsubsws */

static vector int __ATTRS_o_ai
vec_vsubsws(vector int __a, vector int __b)
{
  return __builtin_altivec_vsubsws(__a, __b);
}

static vector int __ATTRS_o_ai
vec_vsubsws(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vsubsws((vector int)__a, __b);
}

static vector int __ATTRS_o_ai
vec_vsubsws(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vsubsws(__a, (vector int)__b);
}

/* vec_vsubuws */

static vector unsigned int __ATTRS_o_ai
vec_vsubuws(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsubuws(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuws(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vsubuws((vector unsigned int)__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_vsubuws(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vsubuws(__a, (vector unsigned int)__b);
}

/* vec_sum4s */

static vector int __ATTRS_o_ai
vec_sum4s(vector signed char __a, vector int __b)
{
  return __builtin_altivec_vsum4sbs(__a, __b);
}

static vector unsigned int __ATTRS_o_ai
vec_sum4s(vector unsigned char __a, vector unsigned int __b)
{
  return __builtin_altivec_vsum4ubs(__a, __b);
}

static vector int __ATTRS_o_ai
vec_sum4s(vector signed short __a, vector int __b)
{
  return __builtin_altivec_vsum4shs(__a, __b);
}

/* vec_vsum4sbs */

static vector int __attribute__((__always_inline__))
vec_vsum4sbs(vector signed char __a, vector int __b)
{
  return __builtin_altivec_vsum4sbs(__a, __b);
}

/* vec_vsum4ubs */

static vector unsigned int __attribute__((__always_inline__))
vec_vsum4ubs(vector unsigned char __a, vector unsigned int __b)
{
  return __builtin_altivec_vsum4ubs(__a, __b);
}

/* vec_vsum4shs */

static vector int __attribute__((__always_inline__))
vec_vsum4shs(vector signed short __a, vector int __b)
{
  return __builtin_altivec_vsum4shs(__a, __b);
}

/* vec_sum2s */

/* The vsum2sws instruction has a big-endian bias, so that the second
   input vector and the result always reference big-endian elements
   1 and 3 (little-endian element 0 and 2).  For ease of porting the
   programmer wants elements 1 and 3 in both cases, so for little
   endian we must perform some permutes.  */

static vector signed int __attribute__((__always_inline__))
vec_sum2s(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  vector int __c = (vector signed int)
    vec_perm(__b, __b, (vector unsigned char)
	     (4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11));
  __c = __builtin_altivec_vsum2sws(__a, __c);
  return (vector signed int)
    vec_perm(__c, __c, (vector unsigned char)
	     (4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11));
#else
  return __builtin_altivec_vsum2sws(__a, __b);
#endif
}

/* vec_vsum2sws */

static vector signed int __attribute__((__always_inline__))
vec_vsum2sws(vector int __a, vector int __b)
{
#ifdef __LITTLE_ENDIAN__
  vector int __c = (vector signed int)
    vec_perm(__b, __b, (vector unsigned char)
	     (4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11));
  __c = __builtin_altivec_vsum2sws(__a, __c);
  return (vector signed int)
    vec_perm(__c, __c, (vector unsigned char)
	     (4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11));
#else
  return __builtin_altivec_vsum2sws(__a, __b);
#endif
}

/* vec_sums */

/* The vsumsws instruction has a big-endian bias, so that the second
   input vector and the result always reference big-endian element 3
   (little-endian element 0).  For ease of porting the programmer
   wants element 3 in both cases, so for little endian we must perform
   some permutes.  */

static vector signed int __attribute__((__always_inline__))
vec_sums(vector signed int __a, vector signed int __b)
{
#ifdef __LITTLE_ENDIAN__
  __b = (vector signed int)vec_splat(__b, 3);
  __b = __builtin_altivec_vsumsws(__a, __b);
  return (vector signed int)(0, 0, 0, __b[0]);
#else
  return __builtin_altivec_vsumsws(__a, __b);
#endif
}

/* vec_vsumsws */

static vector signed int __attribute__((__always_inline__))
vec_vsumsws(vector signed int __a, vector signed int __b)
{
#ifdef __LITTLE_ENDIAN__
  __b = (vector signed int)vec_splat(__b, 3);
  __b = __builtin_altivec_vsumsws(__a, __b);
  return (vector signed int)(0, 0, 0, __b[0]);
#else
  return __builtin_altivec_vsumsws(__a, __b);
#endif
}

/* vec_trunc */

static vector float __attribute__((__always_inline__))
vec_trunc(vector float __a)
{
  return __builtin_altivec_vrfiz(__a);
}

/* vec_vrfiz */

static vector float __attribute__((__always_inline__))
vec_vrfiz(vector float __a)
{
  return __builtin_altivec_vrfiz(__a);
}

/* vec_unpackh */

/* The vector unpack instructions all have a big-endian bias, so for
   little endian we must reverse the meanings of "high" and "low."  */

static vector short __ATTRS_o_ai
vec_unpackh(vector signed char __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupklsb((vector char)__a);
#else
  return __builtin_altivec_vupkhsb((vector char)__a);
#endif
}

static vector bool short __ATTRS_o_ai
vec_unpackh(vector bool char __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool short)__builtin_altivec_vupklsb((vector char)__a);
#else
  return (vector bool short)__builtin_altivec_vupkhsb((vector char)__a);
#endif
}

static vector int __ATTRS_o_ai
vec_unpackh(vector short __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupklsh(__a);
#else
  return __builtin_altivec_vupkhsh(__a);
#endif
}

static vector bool int __ATTRS_o_ai
vec_unpackh(vector bool short __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool int)__builtin_altivec_vupklsh((vector short)__a);
#else
  return (vector bool int)__builtin_altivec_vupkhsh((vector short)__a);
#endif
}

static vector unsigned int __ATTRS_o_ai
vec_unpackh(vector pixel __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned int)__builtin_altivec_vupklpx((vector short)__a);
#else
  return (vector unsigned int)__builtin_altivec_vupkhpx((vector short)__a);
#endif
}

/* vec_vupkhsb */

static vector short __ATTRS_o_ai
vec_vupkhsb(vector signed char __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupklsb((vector char)__a);
#else
  return __builtin_altivec_vupkhsb((vector char)__a);
#endif
}

static vector bool short __ATTRS_o_ai
vec_vupkhsb(vector bool char __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool short)__builtin_altivec_vupklsb((vector char)__a);
#else
  return (vector bool short)__builtin_altivec_vupkhsb((vector char)__a);
#endif
}

/* vec_vupkhsh */

static vector int __ATTRS_o_ai
vec_vupkhsh(vector short __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupklsh(__a);
#else
  return __builtin_altivec_vupkhsh(__a);
#endif
}

static vector bool int __ATTRS_o_ai
vec_vupkhsh(vector bool short __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool int)__builtin_altivec_vupklsh((vector short)__a);
#else
  return (vector bool int)__builtin_altivec_vupkhsh((vector short)__a);
#endif
}

static vector unsigned int __ATTRS_o_ai
vec_vupkhsh(vector pixel __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned int)__builtin_altivec_vupklpx((vector short)__a);
#else
  return (vector unsigned int)__builtin_altivec_vupkhpx((vector short)__a);
#endif
}

/* vec_unpackl */

static vector short __ATTRS_o_ai
vec_unpackl(vector signed char __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupkhsb((vector char)__a);
#else
  return __builtin_altivec_vupklsb((vector char)__a);
#endif
}

static vector bool short __ATTRS_o_ai
vec_unpackl(vector bool char __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool short)__builtin_altivec_vupkhsb((vector char)__a);
#else
  return (vector bool short)__builtin_altivec_vupklsb((vector char)__a);
#endif
}

static vector int __ATTRS_o_ai
vec_unpackl(vector short __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupkhsh(__a);
#else
  return __builtin_altivec_vupklsh(__a);
#endif
}

static vector bool int __ATTRS_o_ai
vec_unpackl(vector bool short __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool int)__builtin_altivec_vupkhsh((vector short)__a);
#else
  return (vector bool int)__builtin_altivec_vupklsh((vector short)__a);
#endif
}

static vector unsigned int __ATTRS_o_ai
vec_unpackl(vector pixel __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned int)__builtin_altivec_vupkhpx((vector short)__a);
#else
  return (vector unsigned int)__builtin_altivec_vupklpx((vector short)__a);
#endif
}

/* vec_vupklsb */

static vector short __ATTRS_o_ai
vec_vupklsb(vector signed char __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupkhsb((vector char)__a);
#else
  return __builtin_altivec_vupklsb((vector char)__a);
#endif
}

static vector bool short __ATTRS_o_ai
vec_vupklsb(vector bool char __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool short)__builtin_altivec_vupkhsb((vector char)__a);
#else
  return (vector bool short)__builtin_altivec_vupklsb((vector char)__a);
#endif
}

/* vec_vupklsh */

static vector int __ATTRS_o_ai
vec_vupklsh(vector short __a)
{
#ifdef __LITTLE_ENDIAN__
  return __builtin_altivec_vupkhsh(__a);
#else
  return __builtin_altivec_vupklsh(__a);
#endif
}

static vector bool int __ATTRS_o_ai
vec_vupklsh(vector bool short __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector bool int)__builtin_altivec_vupkhsh((vector short)__a);
#else
  return (vector bool int)__builtin_altivec_vupklsh((vector short)__a);
#endif
}

static vector unsigned int __ATTRS_o_ai
vec_vupklsh(vector pixel __a)
{
#ifdef __LITTLE_ENDIAN__
  return (vector unsigned int)__builtin_altivec_vupkhpx((vector short)__a);
#else
  return (vector unsigned int)__builtin_altivec_vupklpx((vector short)__a);
#endif
}

/* vec_xor */

#define __builtin_altivec_vxor vec_xor

static vector signed char __ATTRS_o_ai
vec_xor(vector signed char __a, vector signed char __b)
{
  return __a ^ __b;
}

static vector signed char __ATTRS_o_ai
vec_xor(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a ^ __b;
}

static vector signed char __ATTRS_o_ai
vec_xor(vector signed char __a, vector bool char __b)
{
  return __a ^ (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_xor(vector unsigned char __a, vector unsigned char __b)
{
  return __a ^ __b;
}

static vector unsigned char __ATTRS_o_ai
vec_xor(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a ^ __b;
}

static vector unsigned char __ATTRS_o_ai
vec_xor(vector unsigned char __a, vector bool char __b)
{
  return __a ^ (vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_xor(vector bool char __a, vector bool char __b)
{
  return __a ^ __b;
}

static vector short __ATTRS_o_ai
vec_xor(vector short __a, vector short __b)
{
  return __a ^ __b;
}

static vector short __ATTRS_o_ai
vec_xor(vector bool short __a, vector short __b)
{
  return (vector short)__a ^ __b;
}

static vector short __ATTRS_o_ai
vec_xor(vector short __a, vector bool short __b)
{
  return __a ^ (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_xor(vector unsigned short __a, vector unsigned short __b)
{
  return __a ^ __b;
}

static vector unsigned short __ATTRS_o_ai
vec_xor(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a ^ __b;
}

static vector unsigned short __ATTRS_o_ai
vec_xor(vector unsigned short __a, vector bool short __b)
{
  return __a ^ (vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_xor(vector bool short __a, vector bool short __b)
{
  return __a ^ __b;
}

static vector int __ATTRS_o_ai
vec_xor(vector int __a, vector int __b)
{
  return __a ^ __b;
}

static vector int __ATTRS_o_ai
vec_xor(vector bool int __a, vector int __b)
{
  return (vector int)__a ^ __b;
}

static vector int __ATTRS_o_ai
vec_xor(vector int __a, vector bool int __b)
{
  return __a ^ (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_xor(vector unsigned int __a, vector unsigned int __b)
{
  return __a ^ __b;
}

static vector unsigned int __ATTRS_o_ai
vec_xor(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a ^ __b;
}

static vector unsigned int __ATTRS_o_ai
vec_xor(vector unsigned int __a, vector bool int __b)
{
  return __a ^ (vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_xor(vector bool int __a, vector bool int __b)
{
  return __a ^ __b;
}

static vector float __ATTRS_o_ai
vec_xor(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a ^ (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_xor(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a ^ (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_xor(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a ^ (vector unsigned int)__b;
  return (vector float)__res;
}

/* vec_vxor */

static vector signed char __ATTRS_o_ai
vec_vxor(vector signed char __a, vector signed char __b)
{
  return __a ^ __b;
}

static vector signed char __ATTRS_o_ai
vec_vxor(vector bool char __a, vector signed char __b)
{
  return (vector signed char)__a ^ __b;
}

static vector signed char __ATTRS_o_ai
vec_vxor(vector signed char __a, vector bool char __b)
{
  return __a ^ (vector signed char)__b;
}

static vector unsigned char __ATTRS_o_ai
vec_vxor(vector unsigned char __a, vector unsigned char __b)
{
  return __a ^ __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vxor(vector bool char __a, vector unsigned char __b)
{
  return (vector unsigned char)__a ^ __b;
}

static vector unsigned char __ATTRS_o_ai
vec_vxor(vector unsigned char __a, vector bool char __b)
{
  return __a ^ (vector unsigned char)__b;
}

static vector bool char __ATTRS_o_ai
vec_vxor(vector bool char __a, vector bool char __b)
{
  return __a ^ __b;
}

static vector short __ATTRS_o_ai
vec_vxor(vector short __a, vector short __b)
{
  return __a ^ __b;
}

static vector short __ATTRS_o_ai
vec_vxor(vector bool short __a, vector short __b)
{
  return (vector short)__a ^ __b;
}

static vector short __ATTRS_o_ai
vec_vxor(vector short __a, vector bool short __b)
{
  return __a ^ (vector short)__b;
}

static vector unsigned short __ATTRS_o_ai
vec_vxor(vector unsigned short __a, vector unsigned short __b)
{
  return __a ^ __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vxor(vector bool short __a, vector unsigned short __b)
{
  return (vector unsigned short)__a ^ __b;
}

static vector unsigned short __ATTRS_o_ai
vec_vxor(vector unsigned short __a, vector bool short __b)
{
  return __a ^ (vector unsigned short)__b;
}

static vector bool short __ATTRS_o_ai
vec_vxor(vector bool short __a, vector bool short __b)
{
  return __a ^ __b;
}

static vector int __ATTRS_o_ai
vec_vxor(vector int __a, vector int __b)
{
  return __a ^ __b;
}

static vector int __ATTRS_o_ai
vec_vxor(vector bool int __a, vector int __b)
{
  return (vector int)__a ^ __b;
}

static vector int __ATTRS_o_ai
vec_vxor(vector int __a, vector bool int __b)
{
  return __a ^ (vector int)__b;
}

static vector unsigned int __ATTRS_o_ai
vec_vxor(vector unsigned int __a, vector unsigned int __b)
{
  return __a ^ __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vxor(vector bool int __a, vector unsigned int __b)
{
  return (vector unsigned int)__a ^ __b;
}

static vector unsigned int __ATTRS_o_ai
vec_vxor(vector unsigned int __a, vector bool int __b)
{
  return __a ^ (vector unsigned int)__b;
}

static vector bool int __ATTRS_o_ai
vec_vxor(vector bool int __a, vector bool int __b)
{
  return __a ^ __b;
}

static vector float __ATTRS_o_ai
vec_vxor(vector float __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a ^ (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vxor(vector bool int __a, vector float __b)
{
  vector unsigned int __res = (vector unsigned int)__a ^ (vector unsigned int)__b;
  return (vector float)__res;
}

static vector float __ATTRS_o_ai
vec_vxor(vector float __a, vector bool int __b)
{
  vector unsigned int __res = (vector unsigned int)__a ^ (vector unsigned int)__b;
  return (vector float)__res;
}

/* ------------------------ extensions for CBEA ----------------------------- */

/* vec_extract */

static signed char __ATTRS_o_ai
vec_extract(vector signed char __a, int __b)
{
  return __a[__b];
}

static unsigned char __ATTRS_o_ai
vec_extract(vector unsigned char __a, int __b)
{
  return __a[__b];
}

static short __ATTRS_o_ai
vec_extract(vector short __a, int __b)
{
  return __a[__b];
}

static unsigned short __ATTRS_o_ai
vec_extract(vector unsigned short __a, int __b)
{
  return __a[__b];
}

static int __ATTRS_o_ai
vec_extract(vector int __a, int __b)
{
  return __a[__b];
}

static unsigned int __ATTRS_o_ai
vec_extract(vector unsigned int __a, int __b)
{
  return __a[__b];
}

static float __ATTRS_o_ai
vec_extract(vector float __a, int __b)
{
  return __a[__b];
}

/* vec_insert */

static vector signed char __ATTRS_o_ai
vec_insert(signed char __a, vector signed char __b, int __c)
{
  __b[__c] = __a;
  return __b;
}

static vector unsigned char __ATTRS_o_ai
vec_insert(unsigned char __a, vector unsigned char __b, int __c)
{
  __b[__c] = __a;
  return __b;
}

static vector short __ATTRS_o_ai
vec_insert(short __a, vector short __b, int __c)
{
  __b[__c] = __a;
  return __b;
}

static vector unsigned short __ATTRS_o_ai
vec_insert(unsigned short __a, vector unsigned short __b, int __c)
{
  __b[__c] = __a;
  return __b;
}

static vector int __ATTRS_o_ai
vec_insert(int __a, vector int __b, int __c)
{
  __b[__c] = __a;
  return __b;
}

static vector unsigned int __ATTRS_o_ai
vec_insert(unsigned int __a, vector unsigned int __b, int __c)
{
  __b[__c] = __a;
  return __b;
}

static vector float __ATTRS_o_ai
vec_insert(float __a, vector float __b, int __c)
{
  __b[__c] = __a;
  return __b;
}

/* vec_lvlx */

static vector signed char __ATTRS_o_ai
vec_lvlx(int __a, const signed char *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector signed char)(0),
                  vec_lvsl(__a, __b));
}

static vector signed char __ATTRS_o_ai
vec_lvlx(int __a, const vector signed char *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector signed char)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlx(int __a, const unsigned char *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector unsigned char)(0),
                  vec_lvsl(__a, __b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlx(int __a, const vector unsigned char *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector unsigned char)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool char __ATTRS_o_ai
vec_lvlx(int __a, const vector bool char *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector bool char)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector short __ATTRS_o_ai
vec_lvlx(int __a, const short *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector short)(0),
                  vec_lvsl(__a, __b));
}

static vector short __ATTRS_o_ai
vec_lvlx(int __a, const vector short *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector short)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlx(int __a, const unsigned short *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector unsigned short)(0),
                  vec_lvsl(__a, __b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlx(int __a, const vector unsigned short *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector unsigned short)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool short __ATTRS_o_ai
vec_lvlx(int __a, const vector bool short *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector bool short)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector pixel __ATTRS_o_ai
vec_lvlx(int __a, const vector pixel *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector pixel)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector int __ATTRS_o_ai
vec_lvlx(int __a, const int *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector int)(0),
                  vec_lvsl(__a, __b));
}

static vector int __ATTRS_o_ai
vec_lvlx(int __a, const vector int *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector int)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlx(int __a, const unsigned int *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector unsigned int)(0),
                  vec_lvsl(__a, __b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlx(int __a, const vector unsigned int *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector unsigned int)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool int __ATTRS_o_ai
vec_lvlx(int __a, const vector bool int *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector bool int)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector float __ATTRS_o_ai
vec_lvlx(int __a, const float *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector float)(0),
                  vec_lvsl(__a, __b));
}

static vector float __ATTRS_o_ai
vec_lvlx(int __a, const vector float *__b)
{
  return vec_perm(vec_ld(__a, __b),
                  (vector float)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

/* vec_lvlxl */

static vector signed char __ATTRS_o_ai
vec_lvlxl(int __a, const signed char *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector signed char)(0),
                  vec_lvsl(__a, __b));
}

static vector signed char __ATTRS_o_ai
vec_lvlxl(int __a, const vector signed char *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector signed char)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlxl(int __a, const unsigned char *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector unsigned char)(0),
                  vec_lvsl(__a, __b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvlxl(int __a, const vector unsigned char *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector unsigned char)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool char __ATTRS_o_ai
vec_lvlxl(int __a, const vector bool char *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector bool char)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector short __ATTRS_o_ai
vec_lvlxl(int __a, const short *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector short)(0),
                  vec_lvsl(__a, __b));
}

static vector short __ATTRS_o_ai
vec_lvlxl(int __a, const vector short *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector short)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlxl(int __a, const unsigned short *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector unsigned short)(0),
                  vec_lvsl(__a, __b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvlxl(int __a, const vector unsigned short *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector unsigned short)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool short __ATTRS_o_ai
vec_lvlxl(int __a, const vector bool short *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector bool short)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector pixel __ATTRS_o_ai
vec_lvlxl(int __a, const vector pixel *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector pixel)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector int __ATTRS_o_ai
vec_lvlxl(int __a, const int *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector int)(0),
                  vec_lvsl(__a, __b));
}

static vector int __ATTRS_o_ai
vec_lvlxl(int __a, const vector int *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector int)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlxl(int __a, const unsigned int *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector unsigned int)(0),
                  vec_lvsl(__a, __b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvlxl(int __a, const vector unsigned int *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector unsigned int)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool int __ATTRS_o_ai
vec_lvlxl(int __a, const vector bool int *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector bool int)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector float __ATTRS_o_ai
vec_lvlxl(int __a, const float *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector float)(0),
                  vec_lvsl(__a, __b));
}

static vector float __ATTRS_o_ai
vec_lvlxl(int __a, vector float *__b)
{
  return vec_perm(vec_ldl(__a, __b),
                  (vector float)(0),
                  vec_lvsl(__a, (unsigned char *)__b));
}

/* vec_lvrx */

static vector signed char __ATTRS_o_ai
vec_lvrx(int __a, const signed char *__b)
{
  return vec_perm((vector signed char)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector signed char __ATTRS_o_ai
vec_lvrx(int __a, const vector signed char *__b)
{
  return vec_perm((vector signed char)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrx(int __a, const unsigned char *__b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrx(int __a, const vector unsigned char *__b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool char __ATTRS_o_ai
vec_lvrx(int __a, const vector bool char *__b)
{
  return vec_perm((vector bool char)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector short __ATTRS_o_ai
vec_lvrx(int __a, const short *__b)
{
  return vec_perm((vector short)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector short __ATTRS_o_ai
vec_lvrx(int __a, const vector short *__b)
{
  return vec_perm((vector short)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrx(int __a, const unsigned short *__b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrx(int __a, const vector unsigned short *__b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool short __ATTRS_o_ai
vec_lvrx(int __a, const vector bool short *__b)
{
  return vec_perm((vector bool short)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector pixel __ATTRS_o_ai
vec_lvrx(int __a, const vector pixel *__b)
{
  return vec_perm((vector pixel)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector int __ATTRS_o_ai
vec_lvrx(int __a, const int *__b)
{
  return vec_perm((vector int)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector int __ATTRS_o_ai
vec_lvrx(int __a, const vector int *__b)
{
  return vec_perm((vector int)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrx(int __a, const unsigned int *__b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrx(int __a, const vector unsigned int *__b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool int __ATTRS_o_ai
vec_lvrx(int __a, const vector bool int *__b)
{
  return vec_perm((vector bool int)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector float __ATTRS_o_ai
vec_lvrx(int __a, const float *__b)
{
  return vec_perm((vector float)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector float __ATTRS_o_ai
vec_lvrx(int __a, const vector float *__b)
{
  return vec_perm((vector float)(0),
                  vec_ld(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

/* vec_lvrxl */

static vector signed char __ATTRS_o_ai
vec_lvrxl(int __a, const signed char *__b)
{
  return vec_perm((vector signed char)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector signed char __ATTRS_o_ai
vec_lvrxl(int __a, const vector signed char *__b)
{
  return vec_perm((vector signed char)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrxl(int __a, const unsigned char *__b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector unsigned char __ATTRS_o_ai
vec_lvrxl(int __a, const vector unsigned char *__b)
{
  return vec_perm((vector unsigned char)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool char __ATTRS_o_ai
vec_lvrxl(int __a, const vector bool char *__b)
{
  return vec_perm((vector bool char)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector short __ATTRS_o_ai
vec_lvrxl(int __a, const short *__b)
{
  return vec_perm((vector short)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector short __ATTRS_o_ai
vec_lvrxl(int __a, const vector short *__b)
{
  return vec_perm((vector short)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrxl(int __a, const unsigned short *__b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector unsigned short __ATTRS_o_ai
vec_lvrxl(int __a, const vector unsigned short *__b)
{
  return vec_perm((vector unsigned short)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool short __ATTRS_o_ai
vec_lvrxl(int __a, const vector bool short *__b)
{
  return vec_perm((vector bool short)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector pixel __ATTRS_o_ai
vec_lvrxl(int __a, const vector pixel *__b)
{
  return vec_perm((vector pixel)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector int __ATTRS_o_ai
vec_lvrxl(int __a, const int *__b)
{
  return vec_perm((vector int)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector int __ATTRS_o_ai
vec_lvrxl(int __a, const vector int *__b)
{
  return vec_perm((vector int)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrxl(int __a, const unsigned int *__b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector unsigned int __ATTRS_o_ai
vec_lvrxl(int __a, const vector unsigned int *__b)
{
  return vec_perm((vector unsigned int)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector bool int __ATTRS_o_ai
vec_lvrxl(int __a, const vector bool int *__b)
{
  return vec_perm((vector bool int)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

static vector float __ATTRS_o_ai
vec_lvrxl(int __a, const float *__b)
{
  return vec_perm((vector float)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, __b));
}

static vector float __ATTRS_o_ai
vec_lvrxl(int __a, const vector float *__b)
{
  return vec_perm((vector float)(0),
                  vec_ldl(__a, __b),
                  vec_lvsl(__a, (unsigned char *)__b));
}

/* vec_stvlx */

static void __ATTRS_o_ai
vec_stvlx(vector signed char __a, int __b, signed char *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector signed char __a, int __b, vector signed char *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned char __a, int __b, unsigned char *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector bool char __a, int __b, vector bool char *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector short __a, int __b, short *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector short __a, int __b, vector short *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned short __a, int __b, unsigned short *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector bool short __a, int __b, vector bool short *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector pixel __a, int __b, vector pixel *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector int __a, int __b, int *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector int __a, int __b, vector int *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned int __a, int __b, unsigned int *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector bool int __a, int __b, vector bool int *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvlx(vector float __a, int __b, vector float *__c)
{
  return vec_st(vec_perm(vec_lvrx(__b, __c),
                         __a,
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

/* vec_stvlxl */

static void __ATTRS_o_ai
vec_stvlxl(vector signed char __a, int __b, signed char *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector signed char __a, int __b, vector signed char *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned char __a, int __b, unsigned char *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector bool char __a, int __b, vector bool char *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector short __a, int __b, short *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector short __a, int __b, vector short *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned short __a, int __b, unsigned short *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector bool short __a, int __b, vector bool short *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector pixel __a, int __b, vector pixel *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector int __a, int __b, int *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector int __a, int __b, vector int *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned int __a, int __b, unsigned int *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector bool int __a, int __b, vector bool int *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvlxl(vector float __a, int __b, vector float *__c)
{
  return vec_stl(vec_perm(vec_lvrx(__b, __c),
                          __a,
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

/* vec_stvrx */

static void __ATTRS_o_ai
vec_stvrx(vector signed char __a, int __b, signed char *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector signed char __a, int __b, vector signed char *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned char __a, int __b, unsigned char *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector bool char __a, int __b, vector bool char *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector short __a, int __b, short *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector short __a, int __b, vector short *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned short __a, int __b, unsigned short *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector bool short __a, int __b, vector bool short *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector pixel __a, int __b, vector pixel *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector int __a, int __b, int *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector int __a, int __b, vector int *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned int __a, int __b, unsigned int *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, __c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector bool int __a, int __b, vector bool int *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

static void __ATTRS_o_ai
vec_stvrx(vector float __a, int __b, vector float *__c)
{
  return vec_st(vec_perm(__a,
                         vec_lvlx(__b, __c),
                         vec_lvsr(__b, (unsigned char *)__c)),
                __b, __c);
}

/* vec_stvrxl */

static void __ATTRS_o_ai
vec_stvrxl(vector signed char __a, int __b, signed char *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector signed char __a, int __b, vector signed char *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned char __a, int __b, unsigned char *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned char __a, int __b, vector unsigned char *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector bool char __a, int __b, vector bool char *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector short __a, int __b, short *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector short __a, int __b, vector short *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned short __a, int __b, unsigned short *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned short __a, int __b, vector unsigned short *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector bool short __a, int __b, vector bool short *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector pixel __a, int __b, vector pixel *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector int __a, int __b, int *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector int __a, int __b, vector int *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned int __a, int __b, unsigned int *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, __c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector unsigned int __a, int __b, vector unsigned int *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector bool int __a, int __b, vector bool int *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

static void __ATTRS_o_ai
vec_stvrxl(vector float __a, int __b, vector float *__c)
{
  return vec_stl(vec_perm(__a,
                          vec_lvlx(__b, __c),
                          vec_lvsr(__b, (unsigned char *)__c)),
                 __b, __c);
}

/* vec_promote */

static vector signed char __ATTRS_o_ai
vec_promote(signed char __a, int __b)
{
  vector signed char __res = (vector signed char)(0);
  __res[__b] = __a;
  return __res;
}

static vector unsigned char __ATTRS_o_ai
vec_promote(unsigned char __a, int __b)
{
  vector unsigned char __res = (vector unsigned char)(0);
  __res[__b] = __a;
  return __res;
}

static vector short __ATTRS_o_ai
vec_promote(short __a, int __b)
{
  vector short __res = (vector short)(0);
  __res[__b] = __a;
  return __res;
}

static vector unsigned short __ATTRS_o_ai
vec_promote(unsigned short __a, int __b)
{
  vector unsigned short __res = (vector unsigned short)(0);
  __res[__b] = __a;
  return __res;
}

static vector int __ATTRS_o_ai
vec_promote(int __a, int __b)
{
  vector int __res = (vector int)(0);
  __res[__b] = __a;
  return __res;
}

static vector unsigned int __ATTRS_o_ai
vec_promote(unsigned int __a, int __b)
{
  vector unsigned int __res = (vector unsigned int)(0);
  __res[__b] = __a;
  return __res;
}

static vector float __ATTRS_o_ai
vec_promote(float __a, int __b)
{
  vector float __res = (vector float)(0);
  __res[__b] = __a;
  return __res;
}

/* vec_splats */

static vector signed char __ATTRS_o_ai
vec_splats(signed char __a)
{
  return (vector signed char)(__a);
}

static vector unsigned char __ATTRS_o_ai
vec_splats(unsigned char __a)
{
  return (vector unsigned char)(__a);
}

static vector short __ATTRS_o_ai
vec_splats(short __a)
{
  return (vector short)(__a);
}

static vector unsigned short __ATTRS_o_ai
vec_splats(unsigned short __a)
{
  return (vector unsigned short)(__a);
}

static vector int __ATTRS_o_ai
vec_splats(int __a)
{
  return (vector int)(__a);
}

static vector unsigned int __ATTRS_o_ai
vec_splats(unsigned int __a)
{
  return (vector unsigned int)(__a);
}

static vector float __ATTRS_o_ai
vec_splats(float __a)
{
  return (vector float)(__a);
}

/* ----------------------------- predicates --------------------------------- */

/* vec_all_eq */

static int __ATTRS_o_ai
vec_all_eq(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_eq(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned short __a, vector unsigned short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned short __a, vector bool short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool short __a, vector short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool short __a, vector unsigned short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool short __a, vector bool short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector pixel __a, vector pixel __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_eq(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_eq(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT, __a, __b);
}

/* vec_all_ge */

static int __ATTRS_o_ai
vec_all_ge(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, __b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, (vector signed char)__b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, __b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, (vector unsigned char)__b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, __b, (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, __b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, (vector short)__b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, __b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, (vector unsigned short)__b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, __b, (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, __b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, (vector int)__b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, __b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, (vector unsigned int)__b, __a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, __b, (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_all_ge(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT, __a, __b);
}

/* vec_all_gt */

static int __ATTRS_o_ai
vec_all_gt(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, __a, (vector signed char)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, __a, (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, (vector unsigned char)__a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, __a, (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, (vector unsigned short)__a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, __a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, __a, (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, (vector unsigned int)__a, __b);
}

static int __ATTRS_o_ai
vec_all_gt(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_all_gt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT, __a, __b);
}

/* vec_all_in */

static int __attribute__((__always_inline__))
vec_all_in(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpbfp_p(__CR6_EQ, __a, __b);
}

/* vec_all_le */

static int __ATTRS_o_ai
vec_all_le(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, __a, (vector signed char)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, __a, (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, (vector unsigned char)__a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, __a, (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, (vector unsigned short)__a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, __a, (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, (vector unsigned int)__a, __b);
}

static int __ATTRS_o_ai
vec_all_le(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_all_le(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT, __b, __a);
}

/* vec_all_lt */

static int __ATTRS_o_ai
vec_all_lt(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, __b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, (vector signed char)__b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, __b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, (vector unsigned char)__b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, __b, (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, __b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, (vector short)__b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, __b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, (vector unsigned short)__b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, __b, (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, __b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, (vector int)__b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, __b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, (vector unsigned int)__b, __a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, __b, (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_all_lt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT, __b, __a);
}

/* vec_all_nan */

static int __attribute__((__always_inline__))
vec_all_nan(vector float __a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ, __a, __a);
}

/* vec_all_ne */

static int __ATTRS_o_ai
vec_all_ne(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_ne(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned short __a, vector unsigned short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned short __a, vector bool short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool short __a, vector short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool short __a, vector unsigned short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool short __a, vector bool short __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector pixel __a, vector pixel __b)
{
  return
    __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)__a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, __a, __b);
}

static int __ATTRS_o_ai
vec_all_ne(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_all_ne(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ, __a, __b);
}

/* vec_all_nge */

static int __attribute__((__always_inline__))
vec_all_nge(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ, __a, __b);
}

/* vec_all_ngt */

static int __attribute__((__always_inline__))
vec_all_ngt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ, __a, __b);
}

/* vec_all_nle */

static int __attribute__((__always_inline__))
vec_all_nle(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ, __b, __a);
}

/* vec_all_nlt */

static int __attribute__((__always_inline__))
vec_all_nlt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ, __b, __a);
}

/* vec_all_numeric */

static int __attribute__((__always_inline__))
vec_all_numeric(vector float __a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT, __a, __a);
}

/* vec_any_eq */

static int __ATTRS_o_ai
vec_any_eq(vector signed char __a, vector signed char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector signed char __a, vector bool char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned char __a, vector unsigned char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned char __a, vector bool char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool char __a, vector signed char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool char __a, vector unsigned char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool char __a, vector bool char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_eq(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, 
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, 
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector pixel __a, vector pixel __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, 
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_eq(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned int __a, vector unsigned int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector unsigned int __a, vector bool int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool int __a, vector int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool int __a, vector unsigned int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector bool int __a, vector bool int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_eq(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ_REV, __a, __b);
}

/* vec_any_ge */

static int __ATTRS_o_ai
vec_any_ge(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, (vector signed char)__b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, (vector unsigned char)__b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, __b, (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, (vector short)__b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned short __a, vector bool short __b)
{
  return
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, (vector unsigned short)__b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool short __a, vector unsigned short __b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, __b, (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, (vector int)__b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, (vector unsigned int)__b, __a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, __b, (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_any_ge(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ_REV, __a, __b);
}

/* vec_any_gt */

static int __ATTRS_o_ai
vec_any_gt(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, __a, (vector signed char)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned char __a, vector bool char __b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, __a, (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool char __a, vector unsigned char __b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, (vector unsigned char)__a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned short __a, vector bool short __b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, __a, (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool short __a, vector unsigned short __b)
{
  return
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, (vector unsigned short)__a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, __a, (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, (vector unsigned int)__a, __b);
}

static int __ATTRS_o_ai
vec_any_gt(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_any_gt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ_REV, __a, __b);
}

/* vec_any_le */

static int __ATTRS_o_ai
vec_any_le(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, __a, (vector signed char)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned char __a, vector bool char __b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, __a, (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool char __a, vector unsigned char __b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, (vector unsigned char)__a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV,
                                      (vector unsigned char)__a,
                                      (vector unsigned char)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned short __a, vector bool short __b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, __a, (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool short __a, vector unsigned short __b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, (vector unsigned short)__a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV,
                                      (vector unsigned short)__a,
                                      (vector unsigned short)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, __a, (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, (vector unsigned int)__a, __b);
}

static int __ATTRS_o_ai
vec_any_le(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV,
                                      (vector unsigned int)__a,
                                      (vector unsigned int)__b);
}

static int __ATTRS_o_ai
vec_any_le(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ_REV, __b, __a);
}

/* vec_any_lt */

static int __ATTRS_o_ai
vec_any_lt(vector signed char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector signed char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, (vector signed char)__b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned char __a, vector unsigned char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned char __a, vector bool char __b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, (vector unsigned char)__b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool char __a, vector signed char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool char __a, vector unsigned char __b)
{
  return 
    __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, __b, (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool char __a, vector bool char __b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV,
                                      (vector unsigned char)__b,
                                      (vector unsigned char)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, (vector short)__b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned short __a, vector bool short __b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, (vector unsigned short)__b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool short __a, vector unsigned short __b)
{
  return 
    __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, __b, (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV,
                                      (vector unsigned short)__b,
                                      (vector unsigned short)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, (vector int)__b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, __b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector unsigned int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, (vector unsigned int)__b, __a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool int __a, vector int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool int __a, vector unsigned int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, __b, (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector bool int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV,
                                      (vector unsigned int)__b,
                                      (vector unsigned int)__a);
}

static int __ATTRS_o_ai
vec_any_lt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ_REV, __b, __a);
}

/* vec_any_nan */

static int __attribute__((__always_inline__))
vec_any_nan(vector float __a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT_REV, __a, __a);
}

/* vec_any_ne */

static int __ATTRS_o_ai
vec_any_ne(vector signed char __a, vector signed char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector signed char __a, vector bool char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned char __a, vector unsigned char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned char __a, vector bool char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool char __a, vector signed char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool char __a, vector unsigned char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool char __a, vector bool char __b)
{
  return
    __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)__a, (vector char)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector short __a, vector short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_ne(vector short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, __a, (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, 
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool short __a, vector short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool short __a, vector unsigned short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool short __a, vector bool short __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector pixel __a, vector pixel __b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV,
                                      (vector short)__a,
                                      (vector short)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector int __a, vector int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT_REV, __a, __b);
}

static int __ATTRS_o_ai
vec_any_ne(vector int __a, vector bool int __b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT_REV, __a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned int __a, vector unsigned int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector unsigned int __a, vector bool int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool int __a, vector int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool int __a, vector unsigned int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector bool int __a, vector bool int __b)
{
  return
    __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)__a, (vector int)__b);
}

static int __ATTRS_o_ai
vec_any_ne(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT_REV, __a, __b);
}

/* vec_any_nge */

static int __attribute__((__always_inline__))
vec_any_nge(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT_REV, __a, __b);
}

/* vec_any_ngt */

static int __attribute__((__always_inline__))
vec_any_ngt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, __a, __b);
}

/* vec_any_nle */

static int __attribute__((__always_inline__))
vec_any_nle(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT_REV, __b, __a);
}

/* vec_any_nlt */

static int __attribute__((__always_inline__))
vec_any_nlt(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, __b, __a);
}

/* vec_any_numeric */

static int __attribute__((__always_inline__))
vec_any_numeric(vector float __a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ_REV, __a, __a);
}

/* vec_any_out */

static int __attribute__((__always_inline__))
vec_any_out(vector float __a, vector float __b)
{
  return __builtin_altivec_vcmpbfp_p(__CR6_EQ_REV, __a, __b);
}

#undef __ATTRS_o_ai

#endif /* __ALTIVEC_H */
