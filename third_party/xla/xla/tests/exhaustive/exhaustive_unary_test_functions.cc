/* Copyright 2024 The OpenXLA Authors.

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

#include <algorithm>
#include <array>
#include <cfenv>  // NOLINT
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "xla/client/lib/constants.h"
#include "xla/client/lib/math.h"
#include "xla/client/xla_builder.h"
#include "xla/tests/exhaustive/error_spec.h"
#include "xla/tests/exhaustive/exhaustive_op_test_base.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/exhaustive/exhaustive_unary_test_definitions.h"
#include "xla/types.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

using Eigen::half;

template <typename T, size_t N>
T EvaluatePolynomial(T x, const std::array<T, N>& coeffs) {
  // Evaluate the polynomial as accurately as we can using double precision and
  // FMA.
  double result = 0;
  double x_d = static_cast<double>(x);
  for (T c : coeffs) {
    result = std::fma(result, x_d, static_cast<double>(c));
  }
  return static_cast<T>(result);
}

// There's no std::erfinv, so we have to implement it ourselves.  This follows
// Wichura 1998, https://www.jstor.org/stable/2347330 which, notably, is a
// different implementation from that in math.cc.
template <typename NativeRefT>
NativeRefT HostErfInv(NativeRefT x) {
  std::array<double, 8> kPolyA = {
      8.8709406962545514830200e2, 1.1819493347062294404278e4,
      2.3782041382114385731252e4, 1.6235862515167575384252e4,
      4.8548868893843886794648e3, 6.9706266534389598238465e2,
      4.7072688112383978012285e1, 1.1975323115670912564578e0,
  };
  std::array<double, 8> kPolyB = {
      5.2264952788528545610e3, 2.8729085735721942674e4, 3.9307895800092710610e4,
      2.1213794301586595867e4, 5.3941960214247511077e3, 6.8718700749205790830e2,
      4.2313330701600911252e1, 1.0000000000000000000e0,
  };
  std::array<double, 8> kPolyC = {
      7.74545014278341407640e-4, 2.27238449892691845833e-2,
      2.41780725177450611770e-1, 1.27045825245236838258e0,
      3.64784832476320460504e0,  5.76949722146069140550e0,
      4.63033784615654529590e0,  1.42343711074968357734e0,
  };
  std::array<double, 8> kPolyD = {
      1.4859850019840355905497876e-9, 7.7441459065157709165577218e-4,
      2.1494160384252876777097297e-2, 2.0945065210512749128288442e-1,
      9.7547832001787427186894837e-1, 2.3707661626024532365971225e0,
      2.9036514445419946173133295e0,  1.4142135623730950488016887e0,
  };
  std::array<double, 8> kPolyE = {
      2.01033439929228813265e-7, 2.71155556874348757815e-5,
      1.24266094738807843860e-3, 2.65321895265761230930e-2,
      2.96560571828504891230e-1, 1.78482653991729133580e0,
      5.46378491116411436990e0,  6.65790464350110377720e0,
  };
  std::array<double, 8> kPolyF = {
      2.891024605872965461538222e-15, 2.010321207683943062279931e-7,
      2.611088405080593625138020e-5,  1.112800997078859844711555e-3,
      2.103693768272068968719679e-2,  1.936480946950659106176712e-1,
      8.482908416595164588112026e-1,  1.414213562373095048801689e0,
  };

  if (std::abs(x) > 1 || std::isnan(x)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (std::abs(x) == 1) {
    return std::copysign(std::numeric_limits<double>::infinity(), x);
  }

  double unsigned_result = [&] {
    double y = std::abs(x);
    if (y <= 0.85) {
      double r = 0.180625 - 0.25 * y * y;
      return (y * EvaluatePolynomial(r, kPolyA)) /
             EvaluatePolynomial(r, kPolyB);
    } else {
      double r = std::sqrt(std::log(2.0) - std::log1p(-y));
      if (r <= 5.0) {
        r -= 1.6;
        return EvaluatePolynomial(r, kPolyC) / EvaluatePolynomial(r, kPolyD);
      } else {
        r -= 5;
        return EvaluatePolynomial(r, kPolyE) / EvaluatePolynomial(r, kPolyF);
      }
    }
  }();
  return static_cast<NativeRefT>(std::copysign(unsigned_result, x));
}

// Digamma implementation using a polynomial from Cephes.  Notably this is a
// different implementation from the one in math.cc.
template <typename NativeRefT>
NativeRefT HostDigamma(NativeRefT x) {
  // Euler-Mascheroni constant
  double kGamma = 0.57721566490153286061;
  double kPi = M_PI;

  std::array<double, 4> kPoly = {
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  double reflection = 0;
  if (x <= 0) {
    double floor = std::floor(x);
    if (x == floor) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Compute reflection term, pi * cot(pi * x).
    reflection = x - floor;
    if (reflection == 0.5) {
      reflection = 0;
    } else {
      if (reflection > 0.5) {
        reflection = x - (floor + 1.0f);
      }
      reflection = kPi / std::tan(kPi * reflection);
    }
    x = 1 - x;
  }

  double result = 0;
  if (x <= 10 && x == std::floor(x)) {
    // Special case for integers <= 10.
    for (int i = 1; i < x; ++i) {
      result += 1.0 / i;
    }
    result -= kGamma;
  } else {
    double w = 0;
    for (; x < 10; ++x) {
      w += 1.0 / x;
    }
    if (x < 1e8) {
      double z = 1.0 / (x * x);
      result = z * EvaluatePolynomial(z, kPoly);
    }
    result = std::log(x) - 0.5 / x - result - w;
  }

  // Compute the final, reflected value.
  return static_cast<NativeRefT>(result - reflection);
}

UNARY_TEST(Log, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsPreV6Tpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(2e-4).rel_err(eps).build();
    };
  }
  Run(Log, std::log, error_spec_gen);
})
UNARY_TEST(Log1p, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(2e-4).rel_err(eps).build();
    };
  }
  Run(Log1p, std::log1p, error_spec_gen);
})
UNARY_TEST(Exp, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsPreV6Tpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT min = std::numeric_limits<NativeT>::min();
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(min).rel_err(75 * eps).build();
    };
  } else if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT min = std::numeric_limits<NativeT>::min();
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(min).rel_err(33 * eps).build();
    };
  }
  Run(Exp, std::exp, error_spec_gen);
})

UNARY_TEST(Expm1, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsPreV6Tpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(1e-5).rel_err(100 * eps).build();
    };
  } else if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT min = std::numeric_limits<NativeT>::min();
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(2 * min).rel_err(33 * eps).build();
    };
  }

  Run(Expm1, std::expm1, error_spec_gen);
})

UNARY_TEST(Logistic, {
  // FIXME(rmlarsen): Break into region around zero and everything else.
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      NativeT atol = std::min(static_cast<NativeT>(0.004),
                              static_cast<NativeT>(200 * eps));
      return ErrorSpec::Builder().abs_err(atol).rel_err(0).build();
    };
  }
  EvaluateOp fn = +[](NativeRefT x) { return 1.0f / (1.0f + std::exp(-x)); };
  auto range_checker = +[](NativeInputs in, NativeT out) {
    if (Eigen::numext::isnan(in[0])) {
      return Eigen::numext::isnan(out);
    }
    return Eigen::numext::abs(out) <= 1.0f;
  };
  Run(Logistic, fn, error_spec_gen, range_checker);
})

// It feels a little overkill to exhaustively test sqrt and pow(x, 0.5), but
// this *did* find a bug, namely that some backends were assuming sqrt(x) ==
// pow(x, 0.5), but this is not true for x == -inf.
UNARY_TEST(PowOneHalf, {
  EvaluateOp fn = +[](NativeRefT x) { return std::pow(x, 0.5f); };
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); }, fn);
})

UNARY_TEST(Rsqrt, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder()
        .abs_err(0)
        .rel_err(2 * eps)
        .strict_signed_zeros()
        .build();
  };
  Run(Rsqrt, +[](NativeRefT x) { return 1 / std::sqrt(x); }, error_spec_gen);
})

UNARY_TEST(Sqrt, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder()
        .abs_err(0)
        .rel_err(2 * eps)
        .strict_signed_zeros()
        .build();
  };
  Run(Sqrt, std::sqrt, error_spec_gen);
})

UNARY_TEST(Cbrt, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder()
        .abs_err(0)
        .rel_err(16 * eps)
        .strict_signed_zeros()
        .build();
  };
  if (IsCpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      // While GPUs and TPUs flush subnormal inputs to zero, the CPU returns a
      // relatively inaccurate approximation for such inputs. Therefore we
      // allow a small absolute error (e.g. ~9e-16 for F32). This corresponds
      // to a 0.5% relative error for the smallest normalized floating point
      // values, increasing gradually to 100% for the smallest subnormal
      // value.
      NativeT denorm_min = std::numeric_limits<NativeT>::denorm_min();
      double abs_err = std::cbrt(denorm_min);

      if constexpr (std::is_same_v<NativeT, double>) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(abs_err)
            .rel_err(70 * eps)
            .strict_signed_zeros()
            .build();
      } else {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(abs_err)
            .rel_err(10 * eps)
            .strict_signed_zeros()
            .build();
      }
    };
  }
  Run(Cbrt, std::cbrt, error_spec_gen);
})

// Tests for inverse hyperbolic functions.
UNARY_TEST(Acosh, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder().abs_err(1e-7).rel_err(50 * eps).build();
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(2e-4).rel_err(eps).build();
    };
  }
  Run(Acosh, std::acosh, error_spec_gen);
})
UNARY_TEST(Asinh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(2e-4).rel_err(eps).build();
    };
  }
  Run(Asinh, std::asinh, error_spec_gen);
})
UNARY_TEST(Atanh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(1e-4).rel_err(eps).build();
    };
  }
  Run(Atanh, std::atanh, error_spec_gen);
})

// Tests for inverse trigonometric functions.
UNARY_TEST(Acos, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host") {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(1e-6).rel_err(10 * eps).build();
    };
  }
  Run(Acos, std::acos, error_spec_gen);
})
UNARY_TEST(Asin, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder().abs_err(2.0f * min).rel_err(10 * eps).build();
  };
  Run(Asin, std::asin, error_spec_gen);
})
UNARY_TEST(Atan, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder().abs_err(2.0f * min).rel_err(20 * eps).build();
  };
  Run(Atan, std::atan, error_spec_gen);
})

UNARY_TEST(Cosh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      // Cosh is always greater than or equal to 1, so an absolute
      // tolerance does not make sense.
      return ErrorSpec::Builder().abs_err(0).rel_err(100 * eps).build();
    };
  }
  auto range_checker =
      +[](NativeInputs in, NativeT actual) { return !(actual < 1); };
  Run(Cosh, std::cosh, error_spec_gen, range_checker);
})

UNARY_TEST(Sinh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(1e-5).rel_err(100 * eps).build();
    };
  }
  Run(Sinh, std::sinh, error_spec_gen);
})

UNARY_TEST(Tanh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsPreV6Tpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      // The range of tanh is [-1:1], so no point in giving a relative
      // tolerance when we have an absolute one.
      return ErrorSpec::Builder().abs_err(5e-4).rel_err(0).build();
    };
  }
  Run(Tanh, std::tanh, error_spec_gen,
      [](NativeInputs in, NativeT out) -> bool {
        if (Eigen::numext::isnan(in[0])) {
          return Eigen::numext::isnan(out);
        }
        return Eigen::numext::abs(out) <= 1.0f;
      });
})

UNARY_TEST(Cos, {
  auto range_checker =
      +[](NativeInputs in, NativeT out) { return !(out < -1 || out > 1); };
  Run(
      Cos, std::cos,
      +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(2 * eps).build();
      },
      range_checker);
})

UNARY_TEST(Sin, {
  auto range_checker =
      +[](NativeInputs in, NativeT out) { return !(out < -1 || out > 1); };
  Run(
      Sin, std::sin,
      +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(2 * eps).build();
      },
      range_checker);
})

UNARY_TEST(Tan, {
  Run(
      Tan, std::tan, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 4 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(4 * eps).build();
      });
})

UNARY_TEST(Erf, { Run(Erf, std::erf); })
UNARY_TEST(Erfc, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder().abs_err(2 * min).rel_err(50 * eps).build();
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT min = std::numeric_limits<NativeT>::min();
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(2 * min).rel_err(100 * eps).build();
    };
  }
  Run(Erfc, std::erfc, error_spec_gen);
})
UNARY_TEST(ErfInv, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder().abs_err(2 * min).rel_err(50 * eps).build();
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(5e-5).rel_err(2 * eps).build();
    };
  }
  Run(ErfInv, HostErfInv, error_spec_gen);
})

UNARY_TEST(Digamma, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder().abs_err(2e-5).rel_err(10 * eps).build();
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(2e-4).rel_err(10 * eps).build();
    };
  }
  Run(Digamma, HostDigamma, error_spec_gen);
})

UNARY_TEST(Lgamma, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec::Builder().abs_err(1e-5).rel_err(150 * eps).build();
  };
  if (IsGpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      if constexpr (std::is_same_v<NativeT, double>) {
        // Very large error on the smallest subnormal input.
        if (static_cast<double>(std::abs(x)) == 4.9406564584124654e-324) {
          return ErrorSpec::Builder().abs_err(0.05).build();
        } else {
          return ErrorSpec::Builder().distance_err(2).build();
        }
      } else {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(1e-5).rel_err(5000 * eps).build();
      }
    };
  } else if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder().abs_err(5e-4).rel_err(5000 * eps).build();
    };
  }
  Run(Lgamma, std::lgamma, error_spec_gen);
})

UNARY_TEST(Round, { Run(Round, std::round); })

UNARY_TEST(RoundNearestEven, {
  auto error_spec_gen = +[](NativeT) {
    return ErrorSpec::Builder().abs_err(0.0).rel_err(0.0).build();
  };
  int curr_direction = fegetround();
  fesetround(FE_TONEAREST);
  Run(RoundNearestEven, std::nearbyint, error_spec_gen);
  fesetround(curr_direction);
})

UNARY_TEST(Reciprocal, {
  // Can be thought of as an absolute error of `<=
  // |std::numeric_limits<Native>::min()|`.
  auto abs_err = +[](NativeT val) -> double {
    NativeT output = static_cast<NativeT>(1.0) / val;
    if (IsSubnormal(output)) {
      return std::numeric_limits<NativeT>::min();
    }
    return 0.0;
  };
  auto abs_err_bf16 = +[](NativeT val) -> double {
    NativeT output = static_cast<NativeT>(1.0) / val;
    if (IsSubnormalOrMinNormal(output)) {
      return std::numeric_limits<NativeT>::min();
    }
    return 0.0;
  };

  ErrorSpecGen error_spec_gen = [](NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };
  if (IsCpu(platform_)) {
    error_spec_gen = [&](NativeT val) {
      return ErrorSpec::Builder()
          .abs_err(abs_err(val))
          .strict_signed_zeros()
          .build();
    };
  }
  if (IsGpu(platform_)) {
    error_spec_gen = [&](NativeT val) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec::Builder()
          .abs_err(abs_err(val))
          .rel_err(eps)
          .strict_signed_zeros()
          .build();
    };
  }
  if (IsTpu(platform_)) {
    error_spec_gen = [&](NativeT val) {
      if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
        return ErrorSpec::Builder()
            .abs_err(abs_err_bf16(val))
            .strict_signed_zeros()
            .build();
      } else if constexpr (std::is_same_v<NativeT, xla::half>) {
        return ErrorSpec::Builder().strict_signed_zeros().build();
      } else if constexpr (std::is_same_v<NativeT, float>) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(abs_err(val))
            .rel_err(eps)
            .strict_signed_zeros()
            .build();
      } else {
        return ErrorSpec{};
      }
    };
  }
  if (IsPreV6Tpu(platform_)) {
    error_spec_gen = [&](NativeT val) {
      if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
        return ErrorSpec::Builder()
            .abs_err(abs_err_bf16(val))
            .strict_signed_zeros()
            .build();
      } else if constexpr (std::is_same_v<NativeT, xla::half>) {
        return ErrorSpec::Builder().strict_signed_zeros().build();
      } else if constexpr (std::is_same_v<NativeT, float>) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(abs_err(val))
            .rel_err(34 * eps)
            .strict_signed_zeros()
            .build();
      } else {
        return ErrorSpec{};
      }
    };
  }
  if (IsPreV5Tpu(platform_)) {
    error_spec_gen = [&](NativeT val) {
      if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
        return ErrorSpec::Builder()
            .abs_err(abs_err_bf16(val))
            .strict_signed_zeros()
            .build();
      } else if constexpr (std::is_same_v<NativeT, xla::half>) {
        return ErrorSpec::Builder().strict_signed_zeros().build();
      } else if constexpr (std::is_same_v<NativeT, float>) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(abs_err(val))
            .rel_err(136 * eps)
            .strict_signed_zeros()
            .build();
      } else {
        return ErrorSpec{};
      }
    };
  }
  Run(Reciprocal, +[](NativeRefT x) { return 1 / x; }, error_spec_gen);
})

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
