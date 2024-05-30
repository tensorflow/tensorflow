/* Copyright 2019 The OpenXLA Authors.

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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>

#include "xla/client/xla_builder.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/util.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {

extern int GetEupVersion();

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
  return static_cast<float>(std::copysign(unsigned_result, x));
}

// Digamma implementation using a polynomial from Cephes.  Notably this is a
// different implementation from the one in math.cc.
float HostDigamma(float x) {
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
  return static_cast<float>(result - reflection);
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
  static constexpr size_t kRandomInputSize = 2048;

 public:
  Exhaustive32BitOrLessUnaryTest()
      : input_lower_bounder_(0),
        input_upper_bounder_(0),
        special_input_bounder_(false) {}

 public:
  // Sets error parameters appropriately for testing tan.
  void SetParamsForTan();

  void SetBounder(const float lower_bounder, const float upper_bounder) {
    input_lower_bounder_ = lower_bounder;
    input_upper_bounder_ = upper_bounder;
    special_input_bounder_ = true;
  }

  bool IsGpu(const std::string& platform) const { return platform == "CUDA"; }
  bool IsCpu(const std::string& platform) const { return platform == "Host"; }
  bool IsTpu(const std::string& platform) const {
    return !IsGpu(platform) && !IsCpu(platform);
  }
  int EupVersion() { return xla::exhaustive_op_test::GetEupVersion(); }

 protected:
  using typename ExhaustiveUnaryTest<T>::NativeT;

 private:
  int64_t GetInputSize() override {
    int64_t begin, end;
    if (special_input_bounder_) {
      return kRandomInputSize;
    } else {
      std::tie(begin, end) = GetParam();
    }
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
    int64_t begin, end;
    if (special_input_bounder_) {
      begin = input_lower_bounder_;
      end = input_upper_bounder_;
      FillRandomInput(input_literal, begin, end);
    } else {
      std::tie(begin, end) = GetParam();
      FillNormalInput(input_literal, begin, end);
    }
  }
  void FillNormalInput(std::array<Literal, 1>* input_literal,
                       const int64_t begin, const int64_t end) {
    using IntegralT =
        typename ExhaustiveOpTestBase<T, 1>::ComponentIntegralNativeT;
    int64_t input_size = (*input_literal)[0].element_count();
    VLOG(2) << "Checking range [" << begin << ", " << end << ")";
    CHECK_EQ(input_size, end - begin);

    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      IntegralT input_val = i + begin;
      input_arr[i] =
          this->ConvertAndReplaceKnownIncorrectValueWith(input_val, 0);
    }
  }

  void FillRandomInput(std::array<Literal, 1>* input_literal,
                       const int64_t begin, const int64_t end) {
    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();

    uint32_t size = kRandomInputSize;
    NativeT inputs[kRandomInputSize];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(static_cast<double>(begin),
                                          static_cast<double>(end));
    for (uint32_t i = 0; i < size; ++i) {
      inputs[i] = NativeT(dist(gen));
      input_arr[i] = inputs[i];
    }
  }
  float input_lower_bounder_;
  float input_upper_bounder_;
  bool special_input_bounder_;
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
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 2e-4, .rel_err = eps};
    };
  }
  Run(Log, std::log, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Log1p, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 2e-4, .rel_err = eps};
    };
  }
  Run(Log1p, std::log1p, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Exp, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT min = std::numeric_limits<NativeT>::min();
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = min, .rel_err = 75 * eps};
    };
  }
  Run(Exp, std::exp, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Expm1, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      // FIXME(rmlarsen): Break into region around zero and everything else.
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 5e-4, .rel_err = 200 * eps};
    };
  }
  Run(Expm1, std::expm1, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Logistic, {
  // FIXME(rmlarsen): Break into region around zero and everything else.
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      float eps = std::numeric_limits<NativeT>::epsilon();
      float atol = std::min(0.004f, 200 * eps);
      return ErrorSpec{.abs_err = atol, .rel_err = 0};
    };
  }
  EvaluateOp fn = +[](float x) { return 1.0f / (1.0f + std::exp(-x)); };
  Run(Logistic, fn, error_spec_gen);
})

// It feels a little overkill to exhaustively test sqrt and pow(x, 0.5), but
// this *did* find a bug, namely that some backends were assuming sqrt(x) ==
// pow(x, 0.5), but this is not true for x == -inf.
UNARY_TEST_FLOAT_32_BITS_OR_LESS(PowOneHalf, {
  EvaluateOp fn = +[](float x) { return std::pow(x, 0.5f); };
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); }, fn);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Rsqrt, {
  auto error_spec_gen = +[](NativeT x) {
    float eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{
        .abs_err = 0, .rel_err = 2 * eps, .strict_signed_zeros = true};
  };
  Run(Rsqrt, +[](float x) { return 1 / std::sqrt(x); }, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Sqrt, {
  auto error_spec_gen = +[](NativeT x) {
    float eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{
        .abs_err = 0, .rel_err = 2 * eps, .strict_signed_zeros = true};
  };
  Run(Sqrt, std::sqrt, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Cbrt, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{
        .abs_err = 0, .rel_err = 5 * eps, .strict_signed_zeros = true};
  };
  if (IsCpu(platform_)) {
    // While GPUs and TPUs flush subnormal inputs to zero, the CPU returns a
    // relatively inaccurate approximation for such inputs. Therefore we allow a
    // small absolute error (e.g. ~9e-16 for F32). This corresponds to a 0.5%
    // relative error for the smallest normalized floating point values,
    // increasing gradually to 100% for the smallest subnormal value.
    error_spec_gen = +[](NativeT x) {
      NativeT denorm_min = std::numeric_limits<NativeT>::denorm_min();
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = std::cbrt(denorm_min),
                       .rel_err = 10 * eps,
                       .strict_signed_zeros = true};
    };
  }
  Run(Cbrt, std::cbrt, error_spec_gen);
})

// Tests for inverse hyperbolic functions.
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Acosh, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{.abs_err = 1e-7, .rel_err = 50 * eps};
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{2e-4, eps};
    };
  }
  Run(Acosh, std::acosh, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Asinh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 2e-4, .rel_err = eps};
    };
  }
  Run(Asinh, std::asinh, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Atanh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 1e-4, .rel_err = eps};
    };
  }
  Run(Atanh, std::atanh, error_spec_gen);
})

// Tests for inverse trogonometric functions.
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Acos, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ != "Host") {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 1e-6, .rel_err = 10 * eps};
    };
  }
  Run(Acos, std::acos, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Asin, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{.abs_err = 2.0f * min, .rel_err = 10 * eps};
  };
  Run(Asin, std::asin, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Atan, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{.abs_err = 2.0f * min, .rel_err = 20 * eps};
  };
  Run(Atan, std::atan, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Cosh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      // Cosh is always greater than or equal to 1, so an absolute
      // tolerance does not make sense.
      return ErrorSpec{.abs_err = 0, .rel_err = 100 * eps};
    };
  }
  Run(Cosh, std::cosh, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Sinh, {
  auto error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 1e-5, .rel_err = 100 * eps};
    };
  }
  Run(Sinh, std::sinh, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(TanhBounderTestUpperBound, {
  SetBounder(8, 9);
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (!IsTpu(platform_)) {
    error_spec_gen =
        +[](NativeT x) { return ErrorSpec{.abs_err = 0, .rel_err = 0}; };
  }
  Run(
      Tanh, +[](float) { return 1.0f; }, error_spec_gen,
      [](NativeT actual) { return actual >= -1 && actual <= 1; });
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(TanhBounderTestLowerBound, {
  SetBounder(-9, -8);
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) { return ErrorSpec{0, 0}; };
  }
  Run(
      Tanh, +[](float) { return -1.0f; }, error_spec_gen,
      [](NativeT actual) { return actual >= -1 && actual <= 1; });
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(TanhNormalTest, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      // The range of tanh is [-1:1], so no point in giving a relative
      // tolerance when we have an absolute one.
      return ErrorSpec{.abs_err = 5e-4, .rel_err = 0};
    };
  }
  Run(Tanh, std::tanh, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Cos, {
  Run(
      Cos, std::cos, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec{.abs_err = 0, .rel_err = 2 * eps};
      });
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Sin, {
  Run(
      Sin, std::sin, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec{.abs_err = 0, .rel_err = 2 * eps};
      });
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Tan, {
  Run(
      Tan, std::tan, +[](NativeT) {
        // This error spec corresponds to a maximum relative error of 4 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec{.abs_err = 0, .rel_err = 4 * eps};
      });
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Erf, { Run(Erf, std::erf); })
UNARY_TEST_FLOAT_32_BITS_OR_LESS(Erfc, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{.abs_err = 2 * min, .rel_err = 50 * eps};
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT min = std::numeric_limits<NativeT>::min();
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 2 * min, .rel_err = 100 * eps};
    };
  }
  Run(Erfc, std::erfc, error_spec_gen);
})
UNARY_TEST_FLOAT_32_BITS_OR_LESS(ErfInv, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT min = std::numeric_limits<NativeT>::min();
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{.abs_err = 2 * min, .rel_err = 50 * eps};
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 5e-5, .rel_err = 2 * eps};
    };
  }
  Run(ErfInv, HostErfInv, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Digamma, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{2e-5, 10 * eps};
  };
  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 2e-4, .rel_err = 10 * eps};
    };
  }
  Run(Digamma, HostDigamma, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Lgamma, {
  auto error_spec_gen = +[](NativeT x) {
    NativeT eps = std::numeric_limits<NativeT>::epsilon();
    return ErrorSpec{.abs_err = 1e-5, .rel_err = 150 * eps};
  };
  if (IsGpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 1e-5, .rel_err = 5000 * eps};
    };
  } else if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeT x) {
      NativeT eps = std::numeric_limits<NativeT>::epsilon();
      return ErrorSpec{.abs_err = 5e-4, .rel_err = 5000 * eps};
    };
  }
  Run(Lgamma, std::lgamma, error_spec_gen);
})

UNARY_TEST_FLOAT_32_BITS_OR_LESS(Round, { Run(Round, std::round); })

UNARY_TEST_FLOAT_32_BITS_OR_LESS(RoundNearestEven, {
  ErrorSpecGen error_spec_gen =
      +[](NativeT) { return ErrorSpec{.abs_err = 0.0, .rel_err = 0.0}; };
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
