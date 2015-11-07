// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The sin, cos, exp, and log functions of this file come from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

#ifndef EIGEN_MATH_FUNCTIONS_ALTIVEC_H
#define EIGEN_MATH_FUNCTIONS_ALTIVEC_H

#include <iostream>

#define DUMP(v) do { std::cout << #v " = " << (v) << std::endl; } while(0)

namespace Eigen {

namespace internal {

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f plog<Packet4f>(const Packet4f& _x)
{
  Packet4f x = _x;
  _EIGEN_DECLARE_CONST_Packet4f(1 , 1.0f);
  _EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);
  _EIGEN_DECLARE_CONST_Packet4i(23, 23);

  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(inv_mant_mask, ~0x7f800000);

  /* the smallest non denormalized float number */
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(min_norm_pos,  0x00800000);
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(minus_inf,     0xff800000); // -1.f/0.f
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(minus_nan,     0xffffffff);

  /* natural logarithm computed for 4 simultaneous float
    return NaN for x <= 0
  */
  _EIGEN_DECLARE_CONST_Packet4f(cephes_SQRTHF, 0.707106781186547524f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p0, 7.0376836292E-2f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p1, - 1.1514610310E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p2, 1.1676998740E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p3, - 1.2420140846E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p4, + 1.4249322787E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p5, - 1.6668057665E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p6, + 2.0000714765E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p7, - 2.4999993993E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p8, + 3.3333331174E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_q1, -2.12194440e-4f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_q2, 0.693359375f);


  Packet4i emm0;

  /* isvalid_mask is 0 if x < 0 or x is NaN. */
  Packet4ui isvalid_mask = reinterpret_cast<Packet4ui>(vec_cmpge(x, p4f_ZERO));
  Packet4ui iszero_mask = reinterpret_cast<Packet4ui>(vec_cmpeq(x, p4f_ZERO));

  x = pmax(x, p4f_min_norm_pos);  /* cut off denormalized stuff */
  emm0 = vec_sr(reinterpret_cast<Packet4i>(x),
                reinterpret_cast<Packet4ui>(p4i_23));

  /* keep only the fractional part */
  x = pand(x, p4f_inv_mant_mask);
  x = por(x, p4f_half);

  emm0 = psub(emm0, p4i_0x7f);
  Packet4f e = padd(vec_ctf(emm0, 0), p4f_1);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  Packet4f mask = reinterpret_cast<Packet4f>(vec_cmplt(x, p4f_cephes_SQRTHF));
  Packet4f tmp = pand(x, mask);
  x = psub(x, p4f_1);
  e = psub(e, pand(p4f_1, mask));
  x = padd(x, tmp);

  Packet4f x2 = pmul(x,x);
  Packet4f x3 = pmul(x2,x);

  Packet4f y, y1, y2;
  y  = pmadd(p4f_cephes_log_p0, x, p4f_cephes_log_p1);
  y1 = pmadd(p4f_cephes_log_p3, x, p4f_cephes_log_p4);
  y2 = pmadd(p4f_cephes_log_p6, x, p4f_cephes_log_p7);
  y  = pmadd(y , x, p4f_cephes_log_p2);
  y1 = pmadd(y1, x, p4f_cephes_log_p5);
  y2 = pmadd(y2, x, p4f_cephes_log_p8);
  y = pmadd(y, x3, y1);
  y = pmadd(y, x3, y2);
  y = pmul(y, x3);

  y1 = pmul(e, p4f_cephes_log_q1);
  tmp = pmul(x2, p4f_half);
  y = padd(y, y1);
  x = psub(x, tmp);
  y2 = pmul(e, p4f_cephes_log_q2);
  x = padd(x, y);
  x = padd(x, y2);
  // negative arg will be NAN, 0 will be -INF
  x = vec_sel(x, p4f_minus_inf, iszero_mask);
  x = vec_sel(p4f_minus_nan, x, isvalid_mask);
  return x;
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pexp<Packet4f>(const Packet4f& _x)
{
  Packet4f x = _x;
  _EIGEN_DECLARE_CONST_Packet4f(1 , 1.0f);
  _EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);
  _EIGEN_DECLARE_CONST_Packet4i(23, 23);


  _EIGEN_DECLARE_CONST_Packet4f(exp_hi,  88.3762626647950f);
  _EIGEN_DECLARE_CONST_Packet4f(exp_lo, -88.3762626647949f);

  _EIGEN_DECLARE_CONST_Packet4f(cephes_LOG2EF, 1.44269504088896341f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_C1, 0.693359375f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_C2, -2.12194440e-4f);

  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p0, 1.9875691500E-4f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p1, 1.3981999507E-3f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p2, 8.3334519073E-3f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p3, 4.1665795894E-2f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p4, 1.6666665459E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p5, 5.0000001201E-1f);

  Packet4f tmp, fx;
  Packet4i emm0;

  // clamp x
  x = vec_max(vec_min(x, p4f_exp_hi), p4f_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = pmadd(x, p4f_cephes_LOG2EF, p4f_half);

  fx = vec_floor(fx);

  tmp = pmul(fx, p4f_cephes_exp_C1);
  Packet4f z = pmul(fx, p4f_cephes_exp_C2);
  x = psub(x, tmp);
  x = psub(x, z);

  z = pmul(x,x);

  Packet4f y = p4f_cephes_exp_p0;
  y = pmadd(y, x, p4f_cephes_exp_p1);
  y = pmadd(y, x, p4f_cephes_exp_p2);
  y = pmadd(y, x, p4f_cephes_exp_p3);
  y = pmadd(y, x, p4f_cephes_exp_p4);
  y = pmadd(y, x, p4f_cephes_exp_p5);
  y = pmadd(y, z, x);
  y = padd(y, p4f_1);

  // build 2^n
  emm0 = vec_cts(fx, 0);
  emm0 = vec_add(emm0, p4i_0x7f);
  emm0 = vec_sl(emm0, reinterpret_cast<Packet4ui>(p4i_23));

  // Altivec's max & min operators just drop silent NaNs. Check NaNs in
  // inputs and return them unmodified.
  Packet4ui isnumber_mask = reinterpret_cast<Packet4ui>(vec_cmpeq(_x, _x));
  return vec_sel(_x, pmax(pmul(y, reinterpret_cast<Packet4f>(emm0)), _x),
                 isnumber_mask);
}

#ifdef __VSX__

#undef GCC_VERSION
#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)

// VSX support varies between different compilers and even different
// versions of the same compiler.  For gcc version >= 4.9.3, we can use
// vec_cts to efficiently convert Packet2d to Packet2l.  Otherwise, use
// a slow version that works with older compilers.
static inline Packet2l ConvertToPacket2l(const Packet2d& x) {
#if GCC_VERSION >= 40903 || defined(__clang__)
  return vec_cts(x, 0);
#else
  double tmp[2];
  memcpy(tmp, &x, sizeof(tmp));
  Packet2l l = { static_cast<long long>(tmp[0]),
                 static_cast<long long>(tmp[1]) };
  return l;
#endif
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d pexp<Packet2d>(const Packet2d& _x)
{
  Packet2d x = _x;

  _EIGEN_DECLARE_CONST_Packet2d(1 , 1.0);
  _EIGEN_DECLARE_CONST_Packet2d(2 , 2.0);
  _EIGEN_DECLARE_CONST_Packet2d(half, 0.5);

  _EIGEN_DECLARE_CONST_Packet2d(exp_hi,  709.437);
  _EIGEN_DECLARE_CONST_Packet2d(exp_lo, -709.436139303);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_LOG2EF, 1.4426950408889634073599);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p0, 1.26177193074810590878e-4);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p1, 3.02994407707441961300e-2);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p2, 9.99999999999999999910e-1);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q0, 3.00198505138664455042e-6);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q1, 2.52448340349684104192e-3);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q2, 2.27265548208155028766e-1);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q3, 2.00000000000000000009e0);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C1, 0.693145751953125);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C2, 1.42860682030941723212e-6);

  Packet2d tmp, fx;
  Packet2l emm0;

  // clamp x
  x = pmax(pmin(x, p2d_exp_hi), p2d_exp_lo);
  /* express exp(x) as exp(g + n*log(2)) */
  fx = pmadd(p2d_cephes_LOG2EF, x, p2d_half);

  fx = vec_floor(fx);

  tmp = pmul(fx, p2d_cephes_exp_C1);
  Packet2d z = pmul(fx, p2d_cephes_exp_C2);
  x = psub(x, tmp);
  x = psub(x, z);

  Packet2d x2 = pmul(x,x);

  Packet2d px = p2d_cephes_exp_p0;
  px = pmadd(px, x2, p2d_cephes_exp_p1);
  px = pmadd(px, x2, p2d_cephes_exp_p2);
  px = pmul (px, x);

  Packet2d qx = p2d_cephes_exp_q0;
  qx = pmadd(qx, x2, p2d_cephes_exp_q1);
  qx = pmadd(qx, x2, p2d_cephes_exp_q2);
  qx = pmadd(qx, x2, p2d_cephes_exp_q3);

  x = pdiv(px,psub(qx,px));
  x = pmadd(p2d_2,x,p2d_1);

  // build 2^n
  emm0 = ConvertToPacket2l(fx);

#ifdef __POWER8_VECTOR__
  static const Packet2l p2l_1023 = { 1023, 1023 };
  static const Packet2ul p2ul_52 = { 52, 52 };

  emm0 = vec_add(emm0, p2l_1023);
  emm0 = vec_sl(emm0, p2ul_52);
#else
  // Code is a bit complex for POWER7.  There is actually a
  // vec_xxsldi intrinsic but it is not supported by some gcc versions.
  // So we shift (52-32) bits and do a word swap with zeros.
  _EIGEN_DECLARE_CONST_Packet4i(1023, 1023);
  _EIGEN_DECLARE_CONST_Packet4i(20, 20);    // 52 - 32

  Packet4i emm04i = reinterpret_cast<Packet4i>(emm0);
  emm04i = vec_add(emm04i, p4i_1023);
  emm04i = vec_sl(emm04i, reinterpret_cast<Packet4ui>(p4i_20));
  static const Packet16uc perm = {
    0x14, 0x15, 0x16, 0x17, 0x00, 0x01, 0x02, 0x03,
    0x1c, 0x1d, 0x1e, 0x1f, 0x08, 0x09, 0x0a, 0x0b };
#ifdef  _BIG_ENDIAN
  emm0 = reinterpret_cast<Packet2l>(vec_perm(p4i_ZERO, emm04i, perm));
#else
  emm0 = reinterpret_cast<Packet2l>(vec_perm(emm04i, p4i_ZERO, perm));
#endif

#endif

  // Altivec's max & min operators just drop silent NaNs. Check NaNs in
  // inputs and return them unmodified.
  Packet2ul isnumber_mask = reinterpret_cast<Packet2ul>(vec_cmpeq(_x, _x));
  return vec_sel(_x, pmax(pmul(x, reinterpret_cast<Packet2d>(emm0)), _x),
                 isnumber_mask);
}
#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_ALTIVEC_H
