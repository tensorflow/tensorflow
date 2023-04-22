/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_SPECIAL_MATH_SPECIAL_MATH_OP_MISC_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_SPECIAL_MATH_SPECIAL_MATH_OP_MISC_IMPL_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <functional>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"

namespace Eigen {
namespace internal {

// Implementation of Dawson's integral based on Cephes.

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_dawsn_interval_1(const Scalar& x) {
  // Rational approximation on [0, 3.25)
  const Scalar AN[] = {
      Scalar(1.13681498971755972054E-11), Scalar(8.49262267667473811108E-10),
      Scalar(1.94434204175553054283E-8),  Scalar(9.53151741254484363489E-7),
      Scalar(3.07828309874913200438E-6),  Scalar(3.52513368520288738649E-4),
      Scalar(-8.50149846724410912031E-4), Scalar(4.22618223005546594270E-2),
      Scalar(-9.17480371773452345351E-2), Scalar(9.99999999999999994612E-1),
  };
  const Scalar AD[] = {
      Scalar(2.40372073066762605484E-11), Scalar(1.48864681368493396752E-9),
      Scalar(5.21265281010541664570E-8),  Scalar(1.27258478273186970203E-6),
      Scalar(2.32490249820789513991E-5),  Scalar(3.25524741826057911661E-4),
      Scalar(3.48805814657162590916E-3),  Scalar(2.79448531198828973716E-2),
      Scalar(1.58874241960120565368E-1),  Scalar(5.74918629489320327824E-1),
      Scalar(1.00000000000000000539E0),
  };
  const Scalar x2 = x * x;
  Scalar y = (x * internal::ppolevl<Scalar, 9>::run(x2, AN)) /
             internal::ppolevl<Scalar, 10>::run(x2, AD);
  return y;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_dawsn_interval_2(const Scalar& x) {
  // Rational approximation on [3.25, 6.25)
  const Scalar BN[] = {
      Scalar(5.08955156417900903354E-1),  Scalar(-2.44754418142697847934E-1),
      Scalar(9.41512335303534411857E-2),  Scalar(-2.18711255142039025206E-2),
      Scalar(3.66207612329569181322E-3),  Scalar(-4.23209114460388756528E-4),
      Scalar(3.59641304793896631888E-5),  Scalar(-2.14640351719968974225E-6),
      Scalar(9.10010780076391431042E-8),  Scalar(-2.40274520828250956942E-9),
      Scalar(3.59233385440928410398E-11),
  };
  const Scalar BD[] = {
      Scalar(1.0),
      Scalar(-6.31839869873368190192E-1),
      Scalar(2.36706788228248691528E-1),
      Scalar(-5.31806367003223277662E-2),
      Scalar(8.48041718586295374409E-3),
      Scalar(-9.47996768486665330168E-4),
      Scalar(7.81025592944552338085E-5),
      Scalar(-4.55875153252442634831E-6),
      Scalar(1.89100358111421846170E-7),
      Scalar(-4.91324691331920606875E-9),
      Scalar(7.18466403235734541950E-11),
  };
  const Scalar one = Scalar(1);
  const Scalar half = Scalar(0.5);

  const Scalar inverse_x = one / x;
  const Scalar inverse_x2 = inverse_x * inverse_x;
  Scalar z = (internal::ppolevl<Scalar, 10>::run(inverse_x2, BN) /
              (x * internal::ppolevl<Scalar, 10>::run(inverse_x2, BD)));
  Scalar y = inverse_x2 * z + inverse_x;
  return half * y;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_dawsn_interval_3(const Scalar& x) {
  // Rational approximation on [6.25, 1.0e9)
  const Scalar CN[] = {
      Scalar(-5.90592860534773254987E-1), Scalar(6.29235242724368800674E-1),
      Scalar(-1.72858975380388136411E-1), Scalar(1.64837047825189632310E-2),
      Scalar(-4.86827613020462700845E-4),
  };
  const Scalar CD[] = {
      Scalar(1.0),
      Scalar(-2.69820057197544900361E0),
      Scalar(1.73270799045947845857E0),
      Scalar(-3.93708582281939493482E-1),
      Scalar(3.44278924041233391079E-2),
      Scalar(-9.73655226040941223894E-4),
  };
  const Scalar one = Scalar(1);
  const Scalar half = Scalar(0.5);

  const Scalar inverse_x = one / x;
  Scalar inverse_x2 = inverse_x * inverse_x;
  Scalar z = (internal::ppolevl<Scalar, 4>::run(inverse_x2, CN) /
              (x * internal::ppolevl<Scalar, 5>::run(inverse_x2, CD)));
  Scalar y = inverse_x2 * z + inverse_x;
  return half * y;
  return pmul(half, y);
}

template <typename Scalar>
struct dawsn_op {
  EIGEN_EMPTY_STRUCT_CTOR(dawsn_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    const Scalar half = Scalar(0.5);
    const Scalar a = Scalar(3.25);
    const Scalar b = Scalar(6.25);
    const Scalar c = Scalar(1.0e9);

    Scalar abs_x = pabs(x);

    Scalar dawsn;
    if (abs_x < a) {
      dawsn = generic_dawsn_interval_1<Scalar>(abs_x);
    } else if (abs_x < b) {
      dawsn = generic_dawsn_interval_2<Scalar>(abs_x);
    } else if (abs_x < c) {
      dawsn = generic_dawsn_interval_3<Scalar>(abs_x);
    } else {
      dawsn = half / x;
    }

    if (x < Scalar(0)) {
      dawsn = -dawsn;
    }
    return dawsn;
  }
};

// Implementation of exponential integral, based on Cephes.

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_expint_interval_1(const Scalar& x) {
  /* 0 < x <= 2
   Ei(x) - EUL - ln(x) = x A(x)/B(x)
   Theoretical peak relative error 9.73e-18  */
  const Scalar A[] = {
      Scalar(-5.350447357812542947283E0), Scalar(2.185049168816613393830E2),
      Scalar(-4.176572384826693777058E3), Scalar(5.541176756393557601232E4),
      Scalar(-3.313381331178144034309E5), Scalar(1.592627163384945414220E6),
  };
  const Scalar B[] = {
      Scalar(1.0),
      Scalar(-5.250547959112862969197E1),
      Scalar(1.259616186786790571525E3),
      Scalar(-1.756549581973534652631E4),
      Scalar(1.493062117002725991967E5),
      Scalar(-7.294949239640527645655E5),
      Scalar(1.592627163384945429726E6),
  };

  // Euler gamma.
  const Scalar EUL = Scalar(0.5772156649015329);

  const Scalar f = (internal::ppolevl<Scalar, 5>::run(x, A) /
                    internal::ppolevl<Scalar, 6>::run(x, B));
  return x * f + EUL + numext::log(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_expint_interval_2(const Scalar& x) {
  /* 2 <= x <= 4
   x exp(-x) Ei(x) - 1  =  1/x A6(1/x) / B6(1/x)
   Theoretical absolute error = 4.89e-17  */
  const Scalar A6[] = {
      Scalar(1.981808503259689673238E-2),  Scalar(-1.271645625984917501326E0),
      Scalar(-2.088160335681228318920E0),  Scalar(2.755544509187936721172E0),
      Scalar(-4.409507048701600257171E-1), Scalar(4.665623805935891391017E-2),
      Scalar(-1.545042679673485262580E-3), Scalar(7.059980605299617478514E-5),
  };
  const Scalar B6[] = {
      Scalar(1.0),
      Scalar(1.476498670914921440652E0),
      Scalar(5.629177174822436244827E-1),
      Scalar(1.699017897879307263248E-1),
      Scalar(2.291647179034212017463E-2),
      Scalar(4.450150439728752875043E-3),
      Scalar(1.727439612206521482874E-4),
      Scalar(3.953167195549672482304E-5),
  };

  const Scalar one = Scalar(1.0);
  Scalar w = one / x;
  Scalar f = (internal::ppolevl<Scalar, 7>::run(w, A6) /
              internal::ppolevl<Scalar, 7>::run(w, B6));
  f = w * f + one;
  return numext::exp(x) * w * f;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_expint_interval_3(const Scalar& x) {
  /* 4 <= x <= 8
     x exp(-x) Ei(x) - 1  =  1/x A5(1/x) / B5(1/x)
     Theoretical absolute error = 2.20e-17  */
  const Scalar A5[] = {
      Scalar(-1.373215375871208729803E0), Scalar(-7.084559133740838761406E-1),
      Scalar(1.580806855547941010501E0),  Scalar(-2.601500427425622944234E-1),
      Scalar(2.994674694113713763365E-2), Scalar(-1.038086040188744005513E-3),
      Scalar(4.371064420753005429514E-5), Scalar(2.141783679522602903795E-6),
  };
  const Scalar B5[] = {
      Scalar(1.0),
      Scalar(8.585231423622028380768E-1),
      Scalar(4.483285822873995129957E-1),
      Scalar(7.687932158124475434091E-2),
      Scalar(2.449868241021887685904E-2),
      Scalar(8.832165941927796567926E-4),
      Scalar(4.590952299511353531215E-4),
      Scalar(-4.729848351866523044863E-6),
      Scalar(2.665195537390710170105E-6),
  };

  const Scalar one = Scalar(1.0);
  Scalar w = one / x;
  Scalar f = (internal::ppolevl<Scalar, 7>::run(w, A5) /
              internal::ppolevl<Scalar, 8>::run(w, B5));
  f = w * f + one;
  return numext::exp(x) * w * f;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_expint_interval_4(const Scalar& x) {
  /* 8 <= x <= 16
   x exp(-x) Ei(x) - 1 = 1/x R(1/x)
   Theoretical peak absolute error = 1.07e-17  */
  const Scalar A2[] = {
      Scalar(-2.106934601691916512584E0),  Scalar(1.732733869664688041885E0),
      Scalar(-2.423619178935841904839E-1), Scalar(2.322724180937565842585E-2),
      Scalar(2.372880440493179832059E-4),  Scalar(-8.343219561192552752335E-5),
      Scalar(1.363408795605250394881E-5),  Scalar(-3.655412321999253963714E-7),
      Scalar(1.464941733975961318456E-8),  Scalar(6.176407863710360207074E-10),
  };
  const Scalar B2[] = {
      Scalar(1.0),
      Scalar(-2.298062239901678075778E-1),
      Scalar(1.105077041474037862347E-1),
      Scalar(-1.566542966630792353556E-2),
      Scalar(2.761106850817352773874E-3),
      Scalar(-2.089148012284048449115E-4),
      Scalar(1.708528938807675304186E-5),
      Scalar(-4.459311796356686423199E-7),
      Scalar(1.394634930353847498145E-8),
      Scalar(6.150865933977338354138E-10),
  };

  const Scalar one = Scalar(1.0);
  Scalar w = one / x;
  Scalar f = (internal::ppolevl<Scalar, 9>::run(w, A2) /
              internal::ppolevl<Scalar, 9>::run(w, B2));
  f = w * f + one;
  return numext::exp(x) * w * f;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_expint_interval_5(const Scalar& x) {
  /* 16 <= x <= 32
   x exp(-x) Ei(x) - 1  =  1/x A4(1/x) / B4(1/x)
   Theoretical absolute error = 1.22e-17  */
  const Scalar A4[] = {
      Scalar(-2.458119367674020323359E-1), Scalar(-1.483382253322077687183E-1),
      Scalar(7.248291795735551591813E-2),  Scalar(-1.348315687380940523823E-2),
      Scalar(1.342775069788636972294E-3),  Scalar(-7.942465637159712264564E-5),
      Scalar(2.644179518984235952241E-6),  Scalar(-4.239473659313765177195E-8),
  };
  const Scalar B4[] = {
      Scalar(1.0),
      Scalar(-1.044225908443871106315E-1),
      Scalar(-2.676453128101402655055E-1),
      Scalar(9.695000254621984627876E-2),
      Scalar(-1.601745692712991078208E-2),
      Scalar(1.496414899205908021882E-3),
      Scalar(-8.462452563778485013756E-5),
      Scalar(2.728938403476726394024E-6),
      Scalar(-4.239462431819542051337E-8),
  };

  const Scalar one = Scalar(1.0);
  Scalar w = one / x;
  Scalar f = (internal::ppolevl<Scalar, 7>::run(w, A4) /
              internal::ppolevl<Scalar, 8>::run(w, B4));
  f = w * f + one;
  return numext::exp(x) * w * f;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_expint_interval_6(const Scalar& x) {
  /* 32 <= x <= 64
   x exp(-x) Ei(x) - 1  =  1/x A7(1/x) / B7(1/x)
   Theoretical absolute error = 7.71e-18  */
  const Scalar A7[] = {
      Scalar(1.212561118105456670844E-1), Scalar(-5.823133179043894485122E-1),
      Scalar(2.348887314557016779211E-1), Scalar(-3.040034318113248237280E-2),
      Scalar(1.510082146865190661777E-3), Scalar(-2.523137095499571377122E-5),
  };
  const Scalar B7[] = {
      Scalar(1.0),
      Scalar(-1.002252150365854016662E0),
      Scalar(2.928709694872224144953E-1),
      Scalar(-3.337004338674007801307E-2),
      Scalar(1.560544881127388842819E-3),
      Scalar(-2.523137093603234562648E-5),
  };

  const Scalar one = Scalar(1.0);
  Scalar w = one / x;
  Scalar f = (internal::ppolevl<Scalar, 5>::run(w, A7) /
              internal::ppolevl<Scalar, 5>::run(w, B7));
  f = w * f + one;
  return numext::exp(x) * w * f;
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_expint_interval_7(const Scalar& x) {
  /* x > 64
   x exp(-x) Ei(x) - 1  =  1/x A3(1/x)/B3(1/x)
   Theoretical absolute error = 6.15e-17  */
  const Scalar A3[] = {
      Scalar(-7.657847078286127362028E-1), Scalar(6.886192415566705051750E-1),
      Scalar(-2.132598113545206124553E-1), Scalar(3.346107552384193813594E-2),
      Scalar(-3.076541477344756050249E-3), Scalar(1.747119316454907477380E-4),
      Scalar(-6.103711682274170530369E-6), Scalar(1.218032765428652199087E-7),
      Scalar(-1.086076102793290233007E-9),
  };
  const Scalar B3[] = {
      Scalar(1.0),
      Scalar(-1.888802868662308731041E0),
      Scalar(1.066691687211408896850E0),
      Scalar(-2.751915982306380647738E-1),
      Scalar(3.930852688233823569726E-2),
      Scalar(-3.414684558602365085394E-3),
      Scalar(1.866844370703555398195E-4),
      Scalar(-6.345146083130515357861E-6),
      Scalar(1.239754287483206878024E-7),
      Scalar(-1.086076102793126632978E-9),
  };

  const Scalar one = Scalar(1.0);
  Scalar w = one / x;
  Scalar f = (internal::ppolevl<Scalar, 8>::run(w, A3) /
              internal::ppolevl<Scalar, 9>::run(w, B3));
  f = w * f + one;
  return numext::exp(x) * w * f;
}

template <typename Scalar>
struct expint_op {
  EIGEN_EMPTY_STRUCT_CTOR(expint_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    const Scalar zero = Scalar(0.0);
    const Scalar two = Scalar(2.0);
    const Scalar four = Scalar(4.0);
    const Scalar eight = Scalar(8.0);
    const Scalar sixteen = Scalar(16.0);
    const Scalar thirty_two = Scalar(32.0);
    const Scalar sixty_four = Scalar(64.0);
    const Scalar nan = Scalar(NumTraits<Scalar>::quiet_NaN());

    if (x < zero) {
      return nan;
    }

    if (x < two) {
      return generic_expint_interval_1<Scalar>(x);
    } else if (x < four) {
      return generic_expint_interval_2<Scalar>(x);
    } else if (x < eight) {
      return generic_expint_interval_3<Scalar>(x);
    } else if (x < sixteen) {
      return generic_expint_interval_4<Scalar>(x);
    } else if (x < thirty_two) {
      return generic_expint_interval_5<Scalar>(x);
    } else if (x < sixty_four) {
      return generic_expint_interval_6<Scalar>(x);
    }
    return generic_expint_interval_7<Scalar>(x);
  }
};

// Implementation of Fresnel cosine and sine integrals, based on Cephes.

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_fresnel_cos_interval_1(const Scalar& x) {
  const Scalar CN[] = {
      Scalar(-4.98843114573573548651E-8), Scalar(9.50428062829859605134E-6),
      Scalar(-6.45191435683965050962E-4), Scalar(1.88843319396703850064E-2),
      Scalar(-2.05525900955013891793E-1), Scalar(9.99999999999999998822E-1),
  };
  const Scalar CD[] = {
      Scalar(3.99982968972495980367E-12), Scalar(9.15439215774657478799E-10),
      Scalar(1.25001862479598821474E-7),  Scalar(1.22262789024179030997E-5),
      Scalar(8.68029542941784300606E-4),  Scalar(4.12142090722199792936E-2),
      Scalar(1.00000000000000000118E0),
  };

  const Scalar x2 = x * x;
  Scalar x4 = x2 * x2;
  return (x * internal::ppolevl<Scalar, 5>::run(x4, CN) /
          internal::ppolevl<Scalar, 6>::run(x4, CD));
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_fresnel_sin_interval_1(const Scalar& x) {
  const Scalar SN[] = {
      Scalar(-2.99181919401019853726E3),  Scalar(7.08840045257738576863E5),
      Scalar(-6.29741486205862506537E7),  Scalar(2.54890880573376359104E9),
      Scalar(-4.42979518059697779103E10), Scalar(3.18016297876567817986E11),
  };
  const Scalar SD[] = {
      Scalar(1.0),
      Scalar(2.81376268889994315696E2),
      Scalar(4.55847810806532581675E4),
      Scalar(5.17343888770096400730E6),
      Scalar(4.19320245898111231129E8),
      Scalar(2.24411795645340920940E10),
      Scalar(6.07366389490084639049E11),
  };

  const Scalar x2 = x * x;
  Scalar x4 = x2 * x2;
  Scalar z = x * x2;
  return (z * internal::ppolevl<Scalar, 5>::run(x4, SN) /
          internal::ppolevl<Scalar, 6>::run(x4, SD));
}

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
generic_fresnel_asymp(const Scalar& x, bool use_sin) {
  const Scalar FN[] = {
      Scalar(4.21543555043677546506E-1),  Scalar(1.43407919780758885261E-1),
      Scalar(1.15220955073585758835E-2),  Scalar(3.45017939782574027900E-4),
      Scalar(4.63613749287867322088E-6),  Scalar(3.05568983790257605827E-8),
      Scalar(1.02304514164907233465E-10), Scalar(1.72010743268161828879E-13),
      Scalar(1.34283276233062758925E-16), Scalar(3.76329711269987889006E-20),
  };
  const Scalar FD[] = {
      Scalar(1.0),
      Scalar(7.51586398353378947175E-1),
      Scalar(1.16888925859191382142E-1),
      Scalar(6.44051526508858611005E-3),
      Scalar(1.55934409164153020873E-4),
      Scalar(1.84627567348930545870E-6),
      Scalar(1.12699224763999035261E-8),
      Scalar(3.60140029589371370404E-11),
      Scalar(5.88754533621578410010E-14),
      Scalar(4.52001434074129701496E-17),
      Scalar(1.25443237090011264384E-20),
  };
  const Scalar GN[] = {
      Scalar(5.04442073643383265887E-1),  Scalar(1.97102833525523411709E-1),
      Scalar(1.87648584092575249293E-2),  Scalar(6.84079380915393090172E-4),
      Scalar(1.15138826111884280931E-5),  Scalar(9.82852443688422223854E-8),
      Scalar(4.45344415861750144738E-10), Scalar(1.08268041139020870318E-12),
      Scalar(1.37555460633261799868E-15), Scalar(8.36354435630677421531E-19),
      Scalar(1.86958710162783235106E-22),
  };
  const Scalar GD[] = {
      Scalar(1.0),
      Scalar(1.47495759925128324529E0),
      Scalar(3.37748989120019970451E-1),
      Scalar(2.53603741420338795122E-2),
      Scalar(8.14679107184306179049E-4),
      Scalar(1.27545075667729118702E-5),
      Scalar(1.04314589657571990585E-7),
      Scalar(4.60680728146520428211E-10),
      Scalar(1.10273215066240270757E-12),
      Scalar(1.38796531259578871258E-15),
      Scalar(8.39158816283118707363E-19),
      Scalar(1.86958710162783236342E-22),
  };

  const Scalar HALF_PI = Scalar(1.5707963267948966);
  const Scalar PI = Scalar(EIGEN_PI);
  const Scalar one = Scalar(1);
  const Scalar half = Scalar(0.5);

  const Scalar x2 = x * x;
  const Scalar t = one / pmul(PI, x2);
  Scalar u = t * t;

  Scalar f = one - u * (internal::ppolevl<Scalar, 9>::run(u, FN) /
                        internal::ppolevl<Scalar, 10>::run(u, FD));
  Scalar g = (t * internal::ppolevl<Scalar, 10>::run(u, GN) /
              internal::ppolevl<Scalar, 11>::run(u, GD));

  const Scalar z = HALF_PI * x2;
  const Scalar c = numext::cos(z);
  const Scalar s = numext::sin(z);
  const Scalar y = one / (PI * x);
  if (use_sin) {
    Scalar intermediate = f * c;
    intermediate += g * s;
    return half - intermediate * y;
  }
  Scalar intermediate = f * s;
  intermediate -= g * c;
  return half + intermediate * y;
}

template <typename Scalar>
struct fresnel_cos_op {
  EIGEN_EMPTY_STRUCT_CTOR(fresnel_cos_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    const Scalar zero = Scalar(0.);
    const Scalar half = Scalar(0.5);
    const Scalar a = Scalar(2.5625);
    const Scalar b = Scalar(36974.0);

    const Scalar abs_x = numext::abs(x);

    if (abs_x > b) {
      if (x < zero) {
        return -half;
      }
      return half;
    }

    const Scalar x2 = x * x;

    Scalar fresnel_cos;
    if (x2 < a) {
      fresnel_cos = generic_fresnel_cos_interval_1<Scalar>(abs_x);
    } else {
      fresnel_cos = generic_fresnel_asymp<Scalar>(abs_x, false);
    }
    if (x < zero) {
      return -fresnel_cos;
    }
    return fresnel_cos;
  }
};

template <typename Scalar>
struct fresnel_sin_op {
  EIGEN_EMPTY_STRUCT_CTOR(fresnel_sin_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    const Scalar zero = Scalar(0.);
    const Scalar half = Scalar(0.5);
    const Scalar a = Scalar(2.5625);
    const Scalar b = Scalar(36974.0);
    const Scalar abs_x = numext::abs(x);

    if (abs_x > b) {
      if (x < zero) {
        return -half;
      }
      return half;
    }

    const Scalar x2 = x * x;

    Scalar fresnel_sin;
    if (x2 < a) {
      fresnel_sin = generic_fresnel_sin_interval_1<Scalar>(abs_x);
    } else {
      fresnel_sin = generic_fresnel_asymp<Scalar>(abs_x, true);
    }

    if (x < zero) {
      return -fresnel_sin;
    }
    return fresnel_sin;
  }
};

// Implementation of Spence's Integral based on Cephes.
template <typename Scalar>
struct spence_op {
  EIGEN_EMPTY_STRUCT_CTOR(spence_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    const Scalar A[] = {
        Scalar(4.65128586073990045278E-5), Scalar(7.31589045238094711071E-3),
        Scalar(1.33847639578309018650E-1), Scalar(8.79691311754530315341E-1),
        Scalar(2.71149851196553469920E0),  Scalar(4.25697156008121755724E0),
        Scalar(3.29771340985225106936E0),  Scalar(1.00000000000000000126E0),
    };
    const Scalar B[] = {
        Scalar(6.90990488912553276999E-4), Scalar(2.54043763932544379113E-2),
        Scalar(2.82974860602568089943E-1), Scalar(1.41172597751831069617E0),
        Scalar(3.63800533345137075418E0),  Scalar(5.03278880143316990390E0),
        Scalar(3.54771340985225096217E0),  Scalar(9.99999999999999998740E-1),
    };
    const Scalar zero = Scalar(0.0);
    const Scalar one = Scalar(1.0);
    const Scalar three_halves = Scalar(1.5);
    const Scalar two = Scalar(2.0);
    const Scalar half = Scalar(0.5);
    const Scalar nan = Scalar(NumTraits<Scalar>::quiet_NaN());
    // pi**2 / 6.
    const Scalar PI2O6 = Scalar(EIGEN_PI * EIGEN_PI / 6.0);

    if (x < zero) {
      return nan;
    } else if (x == zero) {
      return PI2O6;
    } else if (x == one) {
      return zero;
    }

    Scalar y;
    if (x < two) {
      y = x;
    } else {
      y = one / x;
    }

    Scalar w;
    if (three_halves < y) {
      w = one / y - one;
    } else {
      if (y < half) {
        w = -y;
      } else {
        w = y - one;
      }
    }
    Scalar spence = -w * (internal::ppolevl<Scalar, 7>::run(w, A) /
                          internal::ppolevl<Scalar, 7>::run(w, B));
    Scalar z = numext::log(y);
    if (y < half) {
      spence = -z * numext::log1p(-y) + PI2O6 - spence;
    }
    if (three_halves < x) {
      spence = -half * z * z - spence;
    }
    return spence;
  }
};

}  // end namespace internal
}  // end namespace Eigen

namespace tensorflow {
namespace functor {

template <typename T>
struct dawsn : base<T, Eigen::internal::dawsn_op<T>> {};

template <typename T>
struct expint : base<T, Eigen::internal::expint_op<T>> {};

template <typename T>
struct fresnel_cos : base<T, Eigen::internal::fresnel_cos_op<T>> {};

template <typename T>
struct fresnel_sin : base<T, Eigen::internal::fresnel_sin_op<T>> {};

template <typename T>
struct spence : base<T, Eigen::internal::spence_op<T>> {};

// Bessel Functions

template <typename T>
struct bessel_i0 : base<T, Eigen::internal::scalar_bessel_i0_op<T>> {};

template <typename T>
struct bessel_i0e : base<T, Eigen::internal::scalar_bessel_i0e_op<T>> {};

template <typename T>
struct bessel_i1 : base<T, Eigen::internal::scalar_bessel_i1_op<T>> {};

template <typename T>
struct bessel_i1e : base<T, Eigen::internal::scalar_bessel_i1e_op<T>> {};

template <typename T>
struct bessel_k0 : base<T, Eigen::internal::scalar_bessel_k0_op<T>> {};

template <typename T>
struct bessel_k0e : base<T, Eigen::internal::scalar_bessel_k0e_op<T>> {};

template <typename T>
struct bessel_k1 : base<T, Eigen::internal::scalar_bessel_k1_op<T>> {};

template <typename T>
struct bessel_k1e : base<T, Eigen::internal::scalar_bessel_k1e_op<T>> {};

template <typename T>
struct bessel_j0 : base<T, Eigen::internal::scalar_bessel_j0_op<T>> {};

template <typename T>
struct bessel_j1 : base<T, Eigen::internal::scalar_bessel_j1_op<T>> {};

template <typename T>
struct bessel_y0 : base<T, Eigen::internal::scalar_bessel_y0_op<T>> {};

template <typename T>
struct bessel_y1 : base<T, Eigen::internal::scalar_bessel_y1_op<T>> {};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPECIAL_MATH_SPECIAL_MATH_OP_MISC_IMPL_H_
