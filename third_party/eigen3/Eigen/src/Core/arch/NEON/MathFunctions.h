// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The sin, cos, exp, and log functions of this file come from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

#ifndef EIGEN_MATH_FUNCTIONS_NEON_H
#define EIGEN_MATH_FUNCTIONS_NEON_H

namespace Eigen {

namespace internal {

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pexp<Packet4f>(const Packet4f& _x)
{
  Packet4f x = _x;
  Packet4f tmp, fx;

  _EIGEN_DECLARE_CONST_Packet4f(1 , 1.0f);
  _EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);
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

  x = vminq_f32(x, p4f_exp_hi);
  x = vmaxq_f32(x, p4f_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(p4f_half, x, p4f_cephes_LOG2EF);

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  Packet4ui mask = vcgtq_f32(tmp, fx);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(p4f_1));

  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, p4f_cephes_exp_C1);
  Packet4f z = vmulq_f32(fx, p4f_cephes_exp_C2);
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  Packet4f y = vmulq_f32(p4f_cephes_exp_p0, x);
  z = vmulq_f32(x, x);
  y = vaddq_f32(y, p4f_cephes_exp_p1);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, p4f_cephes_exp_p2);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, p4f_cephes_exp_p3);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, p4f_cephes_exp_p4);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, p4f_cephes_exp_p5);

  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, p4f_1);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, p4i_0x7f);
  mm = vshlq_n_s32(mm, 23);
  Packet4f pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_MATH_FUNCTIONS_NEON_H
