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

#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {

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

class ExhaustiveRealUnaryTestBase : public ExhaustiveOpTestBase {
 public:
  explicit ExhaustiveRealUnaryTestBase(PrimitiveType ty)
      : ExhaustiveOpTestBase(ty) {}

  // A helper for implementing the Run method for unary op test. It constructs
  // the HLO module, compiles and runs the module and checks the result.
  //
  // T: is the input and output data type.
  // RefT: is the type used for the host function to get the reference result.
  //  RefT is different from T when T is of less than 32 bits, that is half and
  //  bfloat16.
  //
  // We use a function pointer for evaluate_op for performance because it is
  // called each time an output element is compared inside a loop in routine
  // ExpectNear.
  template <typename T, typename RefT>
  void RunImpl(std::function<XlaOp(XlaOp)> enqueue_op,
               RefT (*evaluate_op)(RefT), const Literal& input_literal,
               std::function<ErrorSpec(float)> error_spec_gen) {
    XlaBuilder builder(TestName());
    XlaOp input = Parameter(&builder, 0, input_literal.shape(), "input");
    enqueue_op(input);
    TF_ASSERT_OK_AND_ASSIGN(XlaComputation comp, builder.Build());
    TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                            RunComputation(comp, {&input_literal}));
    ExpectNear<T, RefT>(input_literal, result_literal, evaluate_op,
                        error_spec_gen);
  }

  // We essentially reimplement LiteralTestUtil::Near here because
  //  a) this streamlined implementation is much faster, and
  //  b) we can print out better error messages (namely, we can print out
  //     which floating-point value input failed, while LiteralTestUtil::Near
  //     can only print out the input index that failed).
  //  c) we need special handling of certain inputs.  For example, we say that
  //     a denormal input has multiple correct outputs (namely, f(x) and f(0))
  //     and just needs to be close to one of them.
  template <typename T, typename RefT>
  void ExpectNear(const Literal& input_literal, const Literal& result_literal,
                  RefT (*evaluate_op)(RefT),
                  std::function<ErrorSpec(float)> error_spec_gen) {
    absl::Span<const T> input_arr = input_literal.data<T>();
    absl::Span<const T> result_arr = result_literal.data<T>();
    ASSERT_EQ(result_arr.size(), input_arr.size());
    int64 mismatches = 0;
    // Hoisting these out of the loop is a nice speedup on shards that have many
    // denormals.
    const T expected_at_pos_zero = static_cast<T>(evaluate_op(0));
    const T expected_at_neg_zero = static_cast<T>(evaluate_op(-0.0));
    const T expected_at_pos_min_normal_float =
        static_cast<T>(evaluate_op(std::numeric_limits<RefT>::min()));
    const T expected_at_neg_min_normal_float =
        static_cast<T>(evaluate_op(-std::numeric_limits<RefT>::min()));

    for (int64 i = 0; i < input_arr.size(); ++i) {
      T input = input_arr[i];
      RefT input_ref_ty = static_cast<RefT>(input);
      T actual = result_arr[i];
      T expected = static_cast<T>(evaluate_op(input_ref_ty));

      ErrorSpec error_spec = error_spec_gen(input_ref_ty);

      // We only implement fpclassify for float and double, so we call
      // IsClose<float> for half and bfloat16.
      if (IsClose(static_cast<RefT>(expected), static_cast<RefT>(actual),
                  error_spec)) {
        continue;
      }

      // Easy case: If `input` is not denormal and !IsClose(expected, actual,
      // error_spec), print an error.
      if (std::fpclassify(input_ref_ty) != FP_SUBNORMAL) {
        PrintMismatch(&mismatches, [&] {
          return absl::StrFormat("Mismatch on %s. Expected %s, but got %s.",
                                 StringifyNum(input), StringifyNum(expected),
                                 StringifyNum(actual));
        });
        continue;
      }

      // Otherwise, `input` is denormal.  For denormal inputs, we accept answers
      // that are close to any of:
      //
      //   - evaluate_op(input)
      //   - evaluate_op(+/-0), where the sign of 0 equal to the sign of
      //     `input`,
      //   - evaluate_op(+/-min_normal_float), where the sign of
      //     min_normal_float matches `input`.
      //   - if relaxed_denormal_signs_, evaluate_op(-/+0), where the sign of
      //     0 is the opposite of `input`.
      //
      // (In particular, the XLA:CPU implementation of log flushes positive
      // denormals to min-normal-float.  This seems kind of reasonable if our
      // goal is to avoid infinities because they cause nans?)
      T sign_preserving_ftz_expected = std::signbit(input_ref_ty)
                                           ? expected_at_neg_zero
                                           : expected_at_pos_zero;
      T flush_to_normal_expected = std::signbit(input_ref_ty)
                                       ? expected_at_neg_min_normal_float
                                       : expected_at_pos_min_normal_float;
      T sign_nonpreserving_ftz_expected = std::signbit(input_ref_ty)
                                              ? expected_at_pos_zero
                                              : expected_at_neg_zero;
      if (IsClose(static_cast<RefT>(sign_preserving_ftz_expected),
                  static_cast<RefT>(actual), error_spec) ||
          IsClose(static_cast<RefT>(flush_to_normal_expected),
                  static_cast<RefT>(actual), error_spec) ||
          (relaxed_denormal_signs_ &&
           IsClose(static_cast<RefT>(sign_nonpreserving_ftz_expected),
                   static_cast<RefT>(actual), error_spec))) {
        continue;
      }

      if (relaxed_denormal_signs_) {
        PrintMismatch(&mismatches, [&] {
          return absl::StrFormat(
              "Mismatch on denormal value %s.  Expected one of:\n"
              "  %10s (evaluated at full-precision value)\n"
              "  %10s (evaluated at sign-preserving min-normal-float)\n"
              "  %10s (evaluated after flushing to sign-preserving zero)\n"
              "  %10s (evaluated after flushing to non-sign-preserving "
              "zero)\n"
              "but got %s.",
              StringifyNum(input),  //
              StringifyNum(expected), StringifyNum(flush_to_normal_expected),
              StringifyNum(sign_preserving_ftz_expected),
              StringifyNum(sign_nonpreserving_ftz_expected),
              StringifyNum(actual));
        });
      } else {
        PrintMismatch(&mismatches, [&] {
          return absl::StrFormat(
              "Mismatch on denormal value %s.  Expected one of:\n"
              "  %10s (evaluated at full-precision value)\n"
              "  %10s (evaluated at sign-preserving min-normal-float)\n"
              "  %10s (evaluated after flushing to sign-preserving zero)\n"
              "but got %s.",
              StringifyNum(input),  //
              StringifyNum(expected), StringifyNum(flush_to_normal_expected),
              StringifyNum(sign_preserving_ftz_expected),  //
              StringifyNum(actual));
        });
      }
    }
    EXPECT_EQ(mismatches, 0);
  }
};

// Exhaustive test for unary operations for <= 32bit floating point types.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - (begin, end) range under test, as zero-extended int64s bitcast to the
//     primtive type under test.
class Exhaustive32BitOrLessUnaryTest
    : public ExhaustiveRealUnaryTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, std::pair<int64, int64>>> {
 public:
  typedef float (*F32EvaluateOp)(float);

  Exhaustive32BitOrLessUnaryTest()
      : ExhaustiveRealUnaryTestBase(std::get<0>(GetParam())) {}

  void Run(std::function<XlaOp(XlaOp)> enqueue_op, F32EvaluateOp evaluate_op) {
    return Run(enqueue_op, evaluate_op, GetDefaultSpecGenerator(ty_));
  }

  void Run(std::function<XlaOp(XlaOp)> enqueue_op, F32EvaluateOp evaluate_op,
           std::function<ErrorSpec(float)> error_spec_gen) {
    SetFastMathDisabled(true);

    // Run all HLO passes.  In particular, constant folding is disabled by
    // default for tests, but we need to run it in order to tickle some bugs.
    mutable_debug_options()->clear_xla_disable_hlo_passes();
    Literal input_literal = CreateInputLiteral();
    switch (ty_) {
      case F32:
        FillInput<float>(&input_literal);
        return RunImpl<float, float>(enqueue_op, evaluate_op, input_literal,
                                     error_spec_gen);
      case F16:
        FillInput<half>(&input_literal);
        return RunImpl<half, float>(enqueue_op, evaluate_op, input_literal,
                                    error_spec_gen);
      case BF16:
        FillInput<bfloat16>(&input_literal);
        return RunImpl<bfloat16, float>(enqueue_op, evaluate_op, input_literal,
                                        error_spec_gen);
      default:
        LOG(FATAL) << "Unhandled type.";
    }
  }

  // Sets error parameters appropriately for testing sin/cos/tan.
  void SetParamsForSinCosTan();

 private:
  int64 GetInputSize() override {
    int64 begin, end;
    std::tie(begin, end) = std::get<1>(GetParam());
    VLOG(2) << "Checking range [" << begin << ", " << end << ")";
    return end - begin;
  }

  // Generates all the input values for the test. The the range of the bit
  // representation of the input values is described by the test parameter as
  // a pair of int64 representing the starting bit pattern and the ending
  // pattern. Each bit representation is first truncated to the integral type of
  // the same bit as the type being tested, if needed, and then bitcasted to the
  // type being tested.
  template <typename T>
  void FillInput(Literal* input_literal) {
    using IntegralT =
    typename int_type::IntegralTypeWithByteWidth<sizeof(T)>::type;
    int64 input_size = input_literal->element_count();
    int64 begin, end;
    std::tie(begin, end) = std::get<1>(GetParam());
    VLOG(2) << "Checking range [" << begin << ", " << end << ")";
    CHECK_EQ(input_size, end - begin);

    absl::Span<T> input_arr = input_literal->data<T>();
    for (int64 i = 0; i < input_size; i++) {
      IntegralT input_val = i + begin;
      input_arr[i] = ConvertAndReplaceKnownIncorrectValueWith<T>(input_val, 0);
    }
  }
};

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Log) {
  auto error_spec_gen = GetDefaultSpecGenerator(ty_);
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = [](float x) { return ErrorSpec{0.001, 0.001}; };
  }

  Run(Log, std::log, error_spec_gen);
}

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Log1p) {
  auto error_spec_gen = GetDefaultSpecGenerator(ty_);
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = [](float x) { return ErrorSpec{0.001, 0.001}; };
  }

  Run(Log1p, std::log1p, error_spec_gen);
}

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Exp) {
  // When x < -105, the true value of exp(x) is smaller than the smallest F32,
  // so exp(x) should return exactly 0. We want our implementation of exp to
  // return exactly 0 as well, as not doing so implies either that our
  // implementation of exp is not following the asymptotic behavior that exp(x)
  // approaches 0 as x approaches -inf, or that our implementation is not
  // approaching 0 fast enough.
  auto default_spec_gen = GetDefaultSpecGenerator(ty_);
  auto error_spec_gen = [default_spec_gen](float x) {
    if (x < -105) {
      return ErrorSpec{0, 0};
    }
    return default_spec_gen(x);
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
}

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Expm1) {
  auto default_spec_gen = GetDefaultSpecGenerator(ty_);
  auto error_spec_gen = [default_spec_gen](float x) {
    if (x < -105) {
      return ErrorSpec{0, 0};
    }
    return default_spec_gen(x);
  };

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
}

// It feels a little overkill to exhaustively test sqrt and pow(x, 0.5), but
// this *did* find a bug, namely that some backends were assuming sqrt(x) ==
// pow(x, 0.5), but this is not true for x == -inf.
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, PowOneHalf) {
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); },
      +[](float x) { return std::pow(x, 0.5f); });
}

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Rsqrt) {
  Run(
      Rsqrt, +[](float x) { return 1 / std::sqrt(x); });
}

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Sqrt) {
  auto default_spec_gen = GetDefaultSpecGenerator(ty_);
  std::function<ErrorSpec(float)> error_spec_gen;
  if (platform_ == "Host" || platform_ == "CUDA") {
    error_spec_gen = [default_spec_gen](float x) {
      ErrorSpec spec = default_spec_gen(x);
      spec.strict_signed_zeros = true;
      return spec;
    };
  } else {
    error_spec_gen = default_spec_gen;
  }

  Run(Sqrt, std::sqrt, error_spec_gen);
}

// TODO(jlebar): Test trig functions over complex inputs.

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Acosh) {
  // Error inherited from Log, which our implementation of Acosh uses.
  std::function<ErrorSpec(float)> error_spec_gen;
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = [](float x) { return ErrorSpec{0.001, 0.001}; };
  } else {
    error_spec_gen = GetDefaultSpecGenerator(ty_);
  }

  Run(Acosh, std::acosh, error_spec_gen);
}
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Asinh) {
  // Error inherited from Log, which our implementation of Asinh uses.
  std::function<ErrorSpec(float)> error_spec_gen;
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    error_spec_gen = [](float x) { return ErrorSpec{0.001, 0.001}; };
  } else {
    error_spec_gen = GetDefaultSpecGenerator(ty_);
  }
  Run(Asinh, std::asinh, error_spec_gen);
}
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Atanh) { Run(Atanh, std::atanh); }
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Acos) { Run(Acos, std::acos); }
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Asin) { Run(Asin, std::asin); }

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Cosh) {
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
}
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Sinh) {
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
}
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Tanh) { Run(Tanh, std::tanh); }

void Exhaustive32BitOrLessUnaryTest::SetParamsForSinCosTan() {
  if (platform_ == "Host" || platform_ == "CUDA") {
    return;
  }

  // Non CPU/GPU targets may have used the Cody-Waite range reduction technique
  // and will not provide meaningful results for sin/cos/tan if magnitudes
  // exceed 2**p.
  if (ty_ == F32) {
    known_incorrect_fn_ = [](int64 v) {
      float f = BitCast<float>(static_cast<uint32>(v));
      return std::abs(f) > (1 << 13);
    };
  } else if (ty_ == BF16) {
    known_incorrect_fn_ = [](int64 v) {
      float f = static_cast<float>(BitCast<bfloat16>(static_cast<uint16>(v)));
      return std::abs(f) > (1 << 13);
    };
  }
}

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Cos) {
  SetParamsForSinCosTan();
  std::function<ErrorSpec(float)> error_spec_gen;
  if (ty_ == F32) {
    error_spec_gen = [](float) { return ErrorSpec{0.001, 0.001}; };
  } else {
    error_spec_gen = GetDefaultSpecGenerator(ty_);
  }
  Run(Cos, std::cos, error_spec_gen);
}
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Sin) {
  SetParamsForSinCosTan();
  std::function<ErrorSpec(float)> error_spec_gen;
  if (ty_ == F32) {
    error_spec_gen = [](float) { return ErrorSpec{0.001, 0.001}; };
  } else {
    error_spec_gen = GetDefaultSpecGenerator(ty_);
  }
  Run(Sin, std::sin, error_spec_gen);
}
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Tan) {
  SetParamsForSinCosTan();
  std::function<ErrorSpec(float)> error_spec_gen;
  if (ty_ == F32) {
    error_spec_gen = [](float) { return ErrorSpec{0.001, 0.001}; };
  } else {
    error_spec_gen = GetDefaultSpecGenerator(ty_);
  }
  Run(Tan, std::tan, error_spec_gen);
}

// TODO(jlebar): Enable these.
// XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Atan) { Run(Atan, std::atan); }
// XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Atan2) { Run(Atan2, std::atan2); }

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Erf) { Run(Erf, std::erf); }
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Erfc) { Run(Erfc, std::erfc); }
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, ErfInv) { Run(ErfInv, HostErfInv); }
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Digamma) {
  std::function<ErrorSpec(float)> error_spec_gen;
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    error_spec_gen = [](float) { return ErrorSpec{0.01, 0.01}; };
  } else {
    error_spec_gen = GetDefaultSpecGenerator(ty_);
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
      if (BitCast<uint32>(x) == 0x00200000 ||
          BitCast<uint32>(x) == 0x80200000) {
        return std::copysign(std::numeric_limits<float>::max(), -x);
      }
      return HostDigamma(x);
    };
    Run(Digamma, host_digamma_with_gpu_ftz_errors, error_spec_gen);
  } else {
    Run(Digamma, HostDigamma, error_spec_gen);
  }
}
XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Lgamma) {
  // Our implementation gets within 0.0001 rel error except for ~20 denormal
  // inputs on GPU.  Anyway 0.001 rel error should be good enough for lgamma.
  auto default_spec_gen = GetDefaultSpecGenerator(ty_);
  std::function<ErrorSpec(float)> error_spec_gen;
  if (platform_ == "CUDA" && (ty_ == F32 || ty_ == F16)) {
    error_spec_gen = [default_spec_gen](float x) {
      ErrorSpec spec = default_spec_gen(x);
      spec.rel_err = 0.001;
      return spec;
    };
  } else {
    error_spec_gen = default_spec_gen;
  }

  float (*host_lgamma)(float) = std::lgamma;
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    error_spec_gen = [](float) { return ErrorSpec{0.01, 0.01}; };

    // Overflows to inf for input 4.08500343e+36 (0x7c44af8e).
    if (ty_ == F32) {
      host_lgamma = +[](float v) {
        if (BitCast<uint32>(v) == 0x7c44af8e) {
          return std::numeric_limits<float>::infinity();
        }
        return std::lgamma(v);
      };
    }
  }
  Run(Lgamma, host_lgamma, error_spec_gen);
}

XLA_TEST_P(Exhaustive32BitOrLessUnaryTest, Round) { Run(Round, std::round); }

INSTANTIATE_TEST_SUITE_P(
    F32, Exhaustive32BitOrLessUnaryTest,
    ::testing::Combine(::testing::Values(F32),
                       ::testing::ValuesIn(
                           ExhaustiveOpTestBase::CreateExhaustiveF32Ranges())));

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(
    F16, Exhaustive32BitOrLessUnaryTest,
    ::testing::Combine(::testing::Values(F16),
                       ::testing::Values(std::make_pair(0, 1 << 16))));
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
INSTANTIATE_TEST_SUITE_P(
    BF16, Exhaustive32BitOrLessUnaryTest,
    ::testing::Combine(::testing::Values(BF16),
                       ::testing::Values(std::make_pair(0, 1 << 16))));
#endif

}  // namespace xla
