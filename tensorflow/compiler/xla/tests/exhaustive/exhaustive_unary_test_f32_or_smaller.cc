/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <fenv.h>  // NOLINT

#include <cmath>
#include <limits>

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "tensorflow/compiler/xla/util.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {

using Eigen::half;

template <typename T, size_t N>
T EvaluatePolynomial(T x, const std::array<T, N>& coeffs) {
  T result = 0;
  for (T c : coeffs) {
    result = result * x + c;
  }
  return result;
}

// There's no std::erfinv, so we have to implement it ourselves.  This follows
// Wichura 1998, https://www.jstor.org/stable/2347330 which, notably, is a
// different implementation from that in math.cc.
float HostErfInv(float x) {
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
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (std::abs(x) == 1) {
    return std::copysign(std::numeric_limits<float>::infinity(), x);
  }

  float unsigned_result = [&] {
    float y = std::abs(x);
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
  return std::copysign(unsigned_result, x);
}

// Digamma implementation using a polynomial from Cephes.  Notably this is a
// different implementation from the one in math.cc.
float HostDigamma(float x) {
  // Euler-Mascheroni constant
  float kGamma = 0.57721566490153286061;
  float kPi = M_PI;

  std::array<float, 4> kPoly = {
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  float reflection = 0;
  if (x <= 0) {
    float floor = std::floor(x);
    if (x == floor) {
      return std::numeric_limits<float>::quiet_NaN();
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

  float result = 0;
  if (x <= 10 && x == std::floor(x)) {
    // Special case for integers <= 10.
    for (int i = 1; i < x; ++i) {
      result += 1.0f / i;
    }
    result -= kGamma;
  } else {
    float w = 0;
    for (; x < 10; ++x) {
      w += 1.0f / x;
    }
    if (x < 1e8) {
      float z = 1.0f / (x * x);
      result = z * EvaluatePolynomial(z, kPoly);
    }
    result = std::log(x) - 0.5f / x - result - w;
  }

  // Compute the final, reflected value.
  return result - reflection;
}

// Exhaustive test for unary operations for <= 32bit floating point types.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - (begin, end) range under test, as zero-extended int64_ts bitcast to the
//     primitive type under test.
template <PrimitiveType T>
class Exhaustive32BitOrLessUnaryTest
    : public ExhaustiveUnaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {
 public:
  // Sets error parameters appropriately for testing tan.
  void SetParamsForTan();

 protected:
  using typename ExhaustiveUnaryTest<T>::NativeT;

 private:
  int64_t GetInputSize() override {
    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    VLOG(2) << "Checking range [" << begin << ", " << end << ")";
    return end - begin;
  }

  // Generates all the input values for the test. The range of the bit
  // representation of the input values is described by the test parameter as
  // a pair of int64_t representing the starting bit pattern and the ending
  // pattern. Each bit representation is first truncated to the integral type of
  // the same bit as the type being tested, if needed, and then bitcasted to the
  // type being tested.
  void FillInput(std::array<Literal, 1>* input_literal) override {
    using IntegralT =
        typename ExhaustiveOpTestBase<T, 1>::ComponentIntegralNativeT;
    int64_t input_size = (*input_literal)[0].element_count();
    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    VLOG(2) << "Checking range [" << begin << ", " << end << ")";
    CHECK_EQ(input_size, end - begin);

    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      IntegralT input_val = i + begin;
      input_arr[i] =
          this->ConvertAndReplaceKnownIncorrectValueWith(input_val, 0);
    }
  }
};

using ExhaustiveF32UnaryTest = Exhaustive32BitOrLessUnaryTest<F32>;
using ExhaustiveF16UnaryTest = Exhaustive32BitOrLessUnaryTest<F16>;
using ExhaustiveBF16UnaryTest = Exhaustive32BitOrLessUnaryTest<BF16>;

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
#define NEED_UNARY_F16 true
#else
#define NEED_UNARY_F16 false
#endif
#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
#define NEED_UNARY_BF16 true
#else
#define NEED_UNARY_BF16 false
#endif

#define UNARY_TEST_F32(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF32UnaryTest, test_name) \
  __VA_ARGS__

#if NEED_UNARY_F16
#define UNARY_TEST_F16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF16UnaryTest, test_name) \
  __VA_ARGS__
#else
#define UNARY_TEST_F16(test_name, ...)
#endif

#if NEED_UNARY_BF16
#define UNARY_TEST_BF16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveBF16UnaryTest, test_name) \
  __VA_ARGS__
#else
#define UNARY_TEST_BF16(test_name, ...)
#endif

#define UNARY_TEST_FLOAT_32_BITS_OR_LESS(test_name, ...) \
  UNARY_TEST_F32(test_name, __VA_ARGS__)                 \
  UNARY_TEST_F16(test_name, __VA_ARGS__)                 \
  UNARY_TEST_BF16(test_name, __VA_ARGS__)

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Log, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = +[](NativeT x) { return ErrorSpec{0.001, 0.001}; };
  }
  Run(Log, std::log, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Log1p, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = +[](NativeT x) { return ErrorSpec{0.001, 0.001}; };
  }
  Run(Log1p, std::log1p, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Exp, {
  // When x < -105, the true value of exp(x) is smaller than the smallest F32,
  // so exp(x) should return exactly 0. We want our implementation of exp to
  // return exactly 0 as well, as not doing so implies either that our
  // implementation of exp is not following the asymptotic behavior that exp(x)
  // approaches 0 as x approaches -inf, or that our implementation is not
  // approaching 0 fast enough.
  ErrorSpecGen error_spec_gen = +[](NativeT x) {
    if (x < static_cast<NativeT>(-105)) {
      return ErrorSpec{0, 0};
    }
    return GetDefaultSpecGenerator()(x);
  };

  // Our CPU implementation of exp returns one incorrect value: says
  // exp(88.7228394) = max-float, but the correct answer is inf.  We deem this
  // acceptable and check for it explicitly so that we can be aware if anything
  // changes.
  if (platform_ == "Host") {
    auto host_exp_with_overflow = +[](float f) {
      if (f == 88.7228394f) {
        return 3.40282347e+38f;
      }
      return std::exp(f);
    };
    Run(Exp, host_exp_with_overflow, error_spec_gen);
  } else {
    Run(Exp, std::exp, error_spec_gen);
  }
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Expm1, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (ty_ == F32) {
    if (platform_ == "Host") {
      error_spec_gen = +[](NativeT x) {
        // We expect no worse than an error of 8 ULPs.
        return ErrorSpec{
            0.0, std::scalbn(8.0f, -std::numeric_limits<float>::digits)};
      };
    } else {
      error_spec_gen = +[](NativeT x) { return ErrorSpec{0, 0.00015}; };
    }
  }

  // Our CPU implementation of expm1 returns one incorrect value: says
  // exp(88.7228394) = max-float, but the correct answer is inf.  We deem this
  // acceptable and check for it explicitly so that we can be aware if anything
  // changes.
  if (platform_ == "Host") {
    auto host_expm1_with_overflow = +[](float f) {
      if (f == 88.7228394f) {
        return 3.40282347e+38f;
      }
      return std::expm1(f);
    };
    Run(Expm1, host_expm1_with_overflow, error_spec_gen);
  } else {
    Run(Expm1, std::expm1, error_spec_gen);
  }
})

// It feels a little overkill to exhaustively test sqrt and pow(x, 0.5), but
// this *did* find a bug, namely that some backends were assuming sqrt(x) ==
// pow(x, 0.5), but this is not true for x == -inf.
UNARY_TEST_FLOAT_32_BITS_OR_LESS(PowOneHalf, {
  EvaluateOp fn = +[](float x) { return std::pow(x, 0.5f); };
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); }, fn);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Rsqrt, {
  Run(
      Rsqrt, +[](float x) { return 1 / std::sqrt(x); });
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Sqrt, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "Host" || platform_ == "CUDA") {
    error_spec_gen = +[](NativeT x) {
      auto spec = GetDefaultSpecGenerator()(x);
      spec.strict_signed_zeros = true;
      return spec;
    };
  }

  Run(Sqrt, std::sqrt, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Cbrt, {
  if (platform_ == "Host" || platform_ == "CUDA") {
    ErrorSpecGen error_spec_gen = +[](NativeT x) {
      return ErrorSpec{0.01, 0.01};
    };
    Run(Cbrt, std::cbrt, error_spec_gen);
  } else {
    Run(Cbrt, std::cbrt);
  }
})

// TODO(jlebar): Test trig functions over complex inputs.
XLA_TEST_P(ExhaustiveF32UnaryTest, Acosh) {
  // Error inherited from Log, which our implementation of Acosh uses.
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA") {
    error_spec_gen = +[](float x) { return ErrorSpec{0.001, 0.001}; };
  }

  Run(Acosh, std::acosh, error_spec_gen);
}

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
XLA_TEST_P(ExhaustiveF16UnaryTest, Acosh) { Run(Acosh, std::acosh); }
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
XLA_TEST_P(ExhaustiveBF16UnaryTest, Acosh) { Run(Acosh, std::acosh); }
#endif

// Tests for Asinh
XLA_TEST_P(ExhaustiveF32UnaryTest, Asinh) {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA") {
    error_spec_gen = +[](float x) { return ErrorSpec{0.001, 0.001}; };
  }

  Run(Asinh, std::asinh, error_spec_gen);
}

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
XLA_TEST_P(ExhaustiveF16UnaryTest, Asinh) { Run(Asinh, std::asinh); }
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
XLA_TEST_P(ExhaustiveBF16UnaryTest, Asinh) { Run(Asinh, std::asinh); }
#endif

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Atanh, { Run(Atanh, std::atanh); })
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Acos, { Run(Acos, std::acos); })
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Asin, { Run(Asin, std::asin); })

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Cosh, {
  // Our cosh implementation incorrectly overflows to inf for +/-89.4159851.
  // The correct answer of 3.40281961e+38 (0x7f7fffec) is very close to
  // max-float, so we deem this acceptable.
  //
  // This does not occur on CPU because we have an offsetting error in our
  // implementation of exp.
  float (*host_cosh)(float);
  if (platform_ == "Host") {
    host_cosh = &std::cosh;
  } else {
    host_cosh = +[](float x) {
      if (std::abs(x) == 89.4159851f) {
        return std::numeric_limits<float>::infinity();
      }
      return std::cosh(x);
    };
  }
  Run(Cosh, host_cosh);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Sinh, {
  // Our sinh implementation incorrectly overflows to +/-inf for +/-89.4159851.
  // The correct answer of 3.40281961e+38 (0x7f7fffec) is very close to
  // max-float, so we deem this acceptable.
  //
  // This does not occur on CPU because we have an offsetting error in our
  // implementation of exp.
  float (*host_sinh)(float);
  if (platform_ == "Host") {
    host_sinh = &std::sinh;
  } else {
    host_sinh = +[](float x) {
      if (std::abs(x) == 89.4159851f) {
        return std::copysign(std::numeric_limits<float>::infinity(), x);
      }
      return std::sinh(x);
    };
  }
  Run(Sinh, host_sinh);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Tanh, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA") {
    error_spec_gen = +[](NativeT x) {
      return x <= static_cast<NativeT>(-20.0) || x >= static_cast<NativeT>(20.0)
                 ? ErrorSpec{0, 0}
                 : GetDefaultSpecGenerator()(x);
    };
  }
  Run(Tanh, std::tanh, error_spec_gen);
})

UNARY_TEST_F32(Cos, {
  Run(
      Cos, std::cos, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        return ErrorSpec(0, 2 * std::numeric_limits<float>::epsilon());
      });
})

UNARY_TEST_F16(Cos, { Run(Cos, std::cos); })

UNARY_TEST_BF16(Cos, { Run(Cos, std::cos); })

UNARY_TEST_F32(Sin, {
  Run(
      Sin, std::sin, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        return ErrorSpec(0, 2 * std::numeric_limits<float>::epsilon());
      });
})

UNARY_TEST_F16(Sin, { Run(Sin, std::sin); })

UNARY_TEST_BF16(Sin, { Run(Sin, std::sin); })

UNARY_TEST_F32(Tan, {
  Run(
      Tan, std::tan, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 4 ULP.
        return ErrorSpec(0, 4 * std::numeric_limits<float>::epsilon());
      });
})

UNARY_TEST_F16(Tan, {
  Run(
      Tan, std::tan, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        return ErrorSpec(0, 2 * std::numeric_limits<Eigen::half>::epsilon());
      });
})

UNARY_TEST_BF16(Tan, {
  Run(
      Tan, std::tan, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 1 ULP.
        return ErrorSpec(0, std::numeric_limits<Eigen::bfloat16>::epsilon());
      });
})

// TODO(jlebar): Enable these.
// UNARY_TEST_FLOAT_32_BITS_OR_LESS(Atan) { Run(Atan, std::atan); }
// UNARY_TEST_FLOAT_32_BITS_OR_LESS(Atan2) { Run(Atan2, std::atan2); }

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Erf, {
  Run(
      Erf, std::erf, +[](NativeT x) {
        NativeT tol =
            std::max(std::numeric_limits<NativeT>::epsilon(),
                     NativeT(4 * std::numeric_limits<float>::epsilon()));
        NativeT abs_tol = std::min(tol, NativeT(1 - std::abs(std::erf(x))));
        return ErrorSpec(abs_tol, tol);
      });
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Erfc, { Run(Erfc, std::erfc); })

UNARY_TEST_F32(ErfInv, { Run(ErfInv, HostErfInv); })

UNARY_TEST_F16(ErfInv, {
  Run(ErfInv, HostErfInv, [](Eigen::half) { return ErrorSpec{0.002, 0.002}; });
})

UNARY_TEST_BF16(ErfInv, { Run(ErfInv, HostErfInv); })

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Digamma, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    error_spec_gen = +[](NativeT) { return ErrorSpec{0.01, 0.01}; };
  }

  if (platform_ == "CUDA") {
    // On GPU we get a wrong answer for the denormal inputs +/-2.93873588e-39
    // (0x00200000 and 0x80200000).  These should return -/+inf (at least
    // according to our reference implementation!) but XLA:GPU returns
    // -/+3.40282326e+38 (0xff7ffffe and 0x7f7ffffe).
    //
    // I deem this an acceptable result, as XLA:GPU flushes denormals, and as
    // the results we get here are very close to MAX_FLOAT.  We just hardcode
    // these results, as this is better than ignoring these inputs altogether.
    auto host_digamma_with_gpu_ftz_errors = +[](float x) {
      if (BitCast<uint32_t>(x) == 0x00200000 ||
          BitCast<uint32_t>(x) == 0x80200000) {
        return std::copysign(std::numeric_limits<float>::max(), -x);
      }
      return HostDigamma(x);
    };
    Run(Digamma, host_digamma_with_gpu_ftz_errors, error_spec_gen);
  } else {
    Run(Digamma, HostDigamma, error_spec_gen);
  }
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Lgamma, {
  // Our implementation gets within 0.0001 rel error except for ~20 denormal
  // inputs on GPU.  Anyway 0.001 rel error should be good enough for lgamma.
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA" && (ty_ == F32 || ty_ == F16)) {
    error_spec_gen = +[](NativeT x) {
      auto spec = GetDefaultSpecGenerator()(x);
      spec.rel_err = 0.001;
      return spec;
    };
  }

  float (*host_lgamma)(float) = std::lgamma;
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    error_spec_gen = +[](NativeT) { return ErrorSpec{0.01, 0.01}; };

    // Overflows to inf for input 4.08500343e+36 (0x7c44af8e).
    if (ty_ == F32) {
      host_lgamma = +[](float v) {
        if (BitCast<uint32_t>(v) == 0x7c44af8e) {
          return std::numeric_limits<float>::infinity();
        }
        return std::lgamma(v);
      };
    }
  }
  Run(Lgamma, host_lgamma, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Round, { Run(Round, std::round); })

UNARY_TEST_FLOAT_32_BITS_OR_LESS(RoundNearestEven, {
  ErrorSpecGen error_spec_gen = +[](NativeT) { return ErrorSpec{0.0, 0.0}; };
  int curr_direction = fegetround();
  fesetround(FE_TONEAREST);
  Run(RoundNearestEven, std::nearbyint, error_spec_gen);
  fesetround(curr_direction);
})

INSTANTIATE_TEST_SUITE_P(F32, ExhaustiveF32UnaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(F16, ExhaustiveF16UnaryTest,
                         ::testing::Values(std::make_pair(0, 1 << 16)));
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
INSTANTIATE_TEST_SUITE_P(BF16, ExhaustiveBF16UnaryTest,
                         ::testing::Values(std::make_pair(0, 1 << 16)));
#endif

}  // namespace exhaustive_op_test
}  // namespace xla
