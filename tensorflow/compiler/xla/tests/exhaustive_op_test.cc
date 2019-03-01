/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

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

// For f32, f16, and bf16, we need 9, 5, and 4 decimal places of precision to be
// guaranteed that we're printing the full number.
//
// (The general formula is, given a floating-point number with S significand
// bits, the number of decimal digits needed to print it to full precision is
//
//   ceil(1 + S * log_10(2)) ~= ceil(1 + S * 0.30103).
//
// See https://people.eecs.berkeley.edu/~wkahan/Math128/BinDecBin.pdf.)
string StringifyNum(float x) {
  return absl::StrFormat("%0.9g (0x%08x)", x, absl::bit_cast<uint32>(x));
}

string StringifyNum(half x) {
  return absl::StrFormat("%0.5g (0x%04x)", static_cast<float>(x),
                         absl::bit_cast<uint16>(x));
}

string StringifyNum(bfloat16 x) {
  return absl::StrFormat("%0.4g (0x%04x)", static_cast<float>(x),
                         absl::bit_cast<uint16>(x));
}

// Test parameter is a tuple containing
//   - primitive type under test,
//   - (begin, end) range under test, as zero-extended int64s bitcast to the
//     primtive type under test.
class ExhaustiveOpTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, std::pair<int64, int64>>> {
 public:
  ExhaustiveOpTest()
      : ty_(std::get<0>(GetParam())), platform_(client_->platform()->Name()) {}

  void Run(std::function<XlaOp(XlaOp)> enqueue_op,
           float (*evaluate_op)(float)) {
    SetFastMathDisabled(true);

    // Run all HLO passes.  In particular, constant folding is disabled by
    // default for tests, but we need to run it in order to tickle some bugs.
    mutable_debug_options()->clear_xla_disable_hlo_passes();

    PrimitiveType ty;
    std::tie(ty, std::ignore) = GetParam();

    switch (ty) {
      case F32:
        SetDefaultErrSpec(0.0001, 0.0001);
        RunImpl<float, uint32>(enqueue_op, evaluate_op);
        break;
      case F16:
        SetDefaultErrSpec(0.001, 0.001);
        RunImpl<half, uint16>(enqueue_op, evaluate_op);
        break;
      case BF16:
        SetDefaultErrSpec(0.001, 0.01);
        RunImpl<bfloat16, uint16>(enqueue_op, evaluate_op);
        break;
      default:
        LOG(FATAL) << "Unhandled type.";
    }
  }

  void SetDefaultErrSpec(float abs_err, float rel_err) {
    if (!abs_err_.has_value()) {
      abs_err_ = abs_err;
    }
    if (!rel_err_.has_value()) {
      rel_err_ = rel_err;
    }
  }

  template <typename T, typename IntegralT>
  void RunImpl(std::function<XlaOp(XlaOp)> enqueue_op,
               float (*evaluate_op)(float)) {
    static_assert(
        sizeof(T) == sizeof(IntegralT),
        "IntegralT must be an unsigned integer type of the same width as T.");

    PrimitiveType ty;
    std::pair<int64, int64> test_range;
    std::tie(ty, test_range) = GetParam();
    int64 begin, end;
    std::tie(begin, end) = test_range;

    if (begin >= known_incorrect_begin_ && end <= known_incorrect_end_) {
      LOG(INFO) << absl::StreamFormat(
          "Skipping this shard, as the range under test, [%d, %d), falls "
          "entirely within the known-incorrect range [%d, %d).",
          begin, end, known_incorrect_begin_, known_incorrect_end_);
      return;
    }

    LOG(INFO) << "Checking range [" << begin << ", " << end << ")";

    int64 input_size = end - begin;
    Literal input_literal = LiteralUtil::CreateFromDimensions(ty, {input_size});
    absl::Span<T> input_arr = input_literal.data<T>();
    for (int64 i = 0; i < input_size; i++) {
      IntegralT input_val = i + begin;
      // If the operation is known to be buggy on a specific input clamp that
      // input to 0 under the assumption that the op is at least correct on 0.
      if (input_val >= known_incorrect_begin_ &&
          input_val < known_incorrect_end_) {
        input_arr[i] = T{0};
      } else {
        input_arr[i] = absl::bit_cast<T>(input_val);
      }
    }

    TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                            BuildAndRunComputation(enqueue_op, input_literal));
    ExpectNear<T>(input_literal, result_literal, evaluate_op);
  }

  StatusOr<Literal> BuildAndRunComputation(
      const std::function<XlaOp(XlaOp)>& enqueue_op,
      const Literal& input_literal) {
    XlaBuilder builder(TestName());
    auto input = Parameter(&builder, 0, input_literal.shape(), "input");
    enqueue_op(input);
    TF_ASSIGN_OR_RETURN(XlaComputation comp, builder.Build());

    // Build and run the computation using the LocalClient API, rather than the
    // plain Client API, which is used by ClientLibraryTestBase.  This is
    // because the plain Client API results does more memcpys to/from Literals,
    // and that's slow given that we're touching a lot of data here.
    //
    // Copy debug options from ClientLibraryTestBase.  In particular, we're
    // interested in disabling constant folding.
    ExecutableBuildOptions build_opts;
    *build_opts.mutable_debug_options() = *mutable_debug_options();
    TF_ASSIGN_OR_RETURN(
        auto executable,
        client_->Compile(comp, {&input_literal.shape()}, build_opts));

    TF_ASSIGN_OR_RETURN(
        ScopedShapedBuffer input_data,
        client_->LiteralToShapedBuffer(input_literal, /*device_ordinal=*/0));

    ExecutableRunOptions run_opts;
    run_opts.set_allocator(client_->backend().memory_allocator());
    run_opts.set_intra_op_thread_pool(
        client_->backend().eigen_intra_op_thread_pool_device());
    TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                        executable->Run({&input_data}, run_opts));

    TF_ASSIGN_OR_RETURN(Literal result_literal,
                        client_->ShapedBufferToLiteral(result));
    return std::move(result_literal);
  }

  template <typename T>
  bool IsClose(T expected, T actual) {
    float expected_f32 = static_cast<float>(expected);
    float actual_f32 = static_cast<float>(actual);
    float abs_err = std::abs(expected_f32 - actual_f32);
    float rel_err = abs_err / std::abs(expected_f32);
    if (strict_signed_zeros_ && actual == T{0} && expected == T{0}) {
      // Check sign of zero.
      return std::signbit(actual_f32) == std::signbit(expected_f32);
    }
    return abs_err < *abs_err_ || rel_err < *rel_err_ ||
           (std::isnan(expected_f32) && std::isnan(actual_f32)) ||
           (std::isinf(expected_f32) && std::isinf(actual_f32) &&
            (expected_f32 > 0) == (actual_f32 > 0));
  }

  template <typename T>
  void ExpectNear(const Literal& input_literal, const Literal& result_literal,
                  float (*evaluate_op)(float)) {
    // We essentially reimplement LiteralTestUtil::Near here because
    //  a) this streamlined implementation is much faster, and
    //  b) we can print out better error messages (namely, we can print out
    //     which floating-point value input failed, while LiteralTestUtil::Near
    //     can only print out the input index that failed).
    //  c) we need special handling of certain inputs.  For example, we say that
    //     a denormal input has multiple correct outputs (namely, f(x) and f(0))
    //     and just needs to be close to one of them.
    absl::Span<const T> input_arr = input_literal.data<T>();
    absl::Span<const T> result_arr = result_literal.data<T>();
    ASSERT_EQ(result_arr.size(), input_arr.size());
    int64 mismatches = 0;
    // Hoisting these out of the loop is a nice speedup on shards that have many
    // denormals.
    const T expected_at_pos_zero = static_cast<T>(evaluate_op(0));
    const T expected_at_neg_zero = static_cast<T>(evaluate_op(-0.0));
    for (int64 i = 0; i < input_arr.size(); ++i) {
      T input = input_arr[i];
      float input_f32 = static_cast<float>(input);
      T actual = result_arr[i];
      T expected = static_cast<T>(evaluate_op(input_f32));

      if (IsClose(expected, actual)) {
        continue;
      }

      // Easy case: If `input` is not denormal and !IsClose(expected, actual),
      // print an error.
      //
      // (This doesn't correctly detect f16 and bfloat16 denormals!  This seems
      // to be OK for now, but at some point we may need to implement fpclassify
      // for half and bfloat.)
      if (std::fpclassify(input_f32) != FP_SUBNORMAL) {
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
      //   - if relaxed_denormal_signs_, evaluate_op(-/+0), where the sign of
      //     0 is the opposite of `input`.
      T sign_preserving_ftz_expected =
          std::signbit(input_f32) ? expected_at_neg_zero : expected_at_pos_zero;
      T sign_nonpreserving_ftz_expected =
          std::signbit(input_f32) ? expected_at_pos_zero : expected_at_neg_zero;
      if (IsClose(sign_preserving_ftz_expected, actual) ||
          (relaxed_denormal_signs_ &&
           IsClose(sign_nonpreserving_ftz_expected, actual))) {
        continue;
      }

      if (relaxed_denormal_signs_) {
        PrintMismatch(&mismatches, [&] {
          return absl::StrFormat(
              "Mismatch on denormal value %s.  Expected one of:\n"
              "  %10s (evaluated at full-precision value)\n"
              "  %10s (evaluated after flushing to sign-preserving zero)\n"
              "  %10s (evaluated after flushing to non-sign-preserving "
              "zero)\n"
              "but got %s.",
              StringifyNum(input), StringifyNum(expected),
              StringifyNum(sign_preserving_ftz_expected),
              StringifyNum(sign_nonpreserving_ftz_expected),
              StringifyNum(actual));
        });
      } else {
        PrintMismatch(&mismatches, [&] {
          return absl::StrFormat(
              "Mismatch on denormal value %s.  Expected one of:\n"
              "  %10s (evaluated at full-precision value)\n"
              "  %10s (evaluated after flushing to sign-preserving zero)\n"
              "but got %s.",
              StringifyNum(input), StringifyNum(expected),
              StringifyNum(sign_preserving_ftz_expected), StringifyNum(actual));
        });
      }
    }
    EXPECT_EQ(mismatches, 0);
  }

  template <typename ErrorGenerator>
  void PrintMismatch(int64* mismatches, const ErrorGenerator& err_generator) {
    // We send a few mismatches to gunit so they show up nicely in test logs.
    // Then we send more to LOG(ERROR).  The remainder we squelch unless we're
    // at vlog level 2.
    constexpr int64 kMaxMismatchesLoggedToGunit = 10;
    constexpr int64 kMaxMismatchesLoggedToErr = 1000;

    (*mismatches)++;
    if (*mismatches < kMaxMismatchesLoggedToGunit) {
      FAIL() << err_generator();
    } else if (*mismatches < kMaxMismatchesLoggedToErr || VLOG_IS_ON(2)) {
      LOG(ERROR) << err_generator();
    } else if (*mismatches == kMaxMismatchesLoggedToErr) {
      LOG(ERROR) << "Not printing any more mismatches; pass "
                    "--vmodule=exhaustive_f32__op_test=2 to see "
                    "all of them.";
    }
  }

  // The following members are set during construction so testcases can read
  // these values and use them e.g. to influence the values given to the mutable
  // members below.

  // The primitive type under test.
  const PrimitiveType ty_;

  // The platform under test.
  const string platform_;

  // Tests can set the following variables for control over execution.  This is
  // safe because each XLA_TEST_P instantiates a new instance of this class.

  // Testing will ignore the given range (encoded as bitwise representations of
  // the type under test zero-extended to int64).
  int64 known_incorrect_begin_ = 0;
  int64 known_incorrect_end_ = 0;

  // If unset, reasonable defaults will be used depending on the type under
  // test.
  absl::optional<float> abs_err_;
  absl::optional<float> rel_err_;

  // If true, will consider -0 not near to +0 and vice versa.  Note that
  // +epsilon may still be considered close to -0, depending on the error spec;
  // this only covers the case when both `expected` and `actual` are equal to 0.
  bool strict_signed_zeros_ = false;

  // If true, allows denormals to be flushed to non-sign-preserving 0.
  //
  // For example, normally we'd expect sqrt(-denormal) to be either nan (sqrt of
  // a negative number) or -inf (flush the denormal to sign-perserving zero,
  // then sqrt(-0)).  But with this as true, we'll also accept 0 (sqrt(0)).
  //
  // XLA:GPU preserves denormal signs, but other backends don't.
  bool relaxed_denormal_signs_ = platform_ != "CUDA";
};

XLA_TEST_P(ExhaustiveOpTest, Log) {
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    abs_err_ = 0.001;
    rel_err_ = 0.001;
  }

  Run(Log, std::log);
}

XLA_TEST_P(ExhaustiveOpTest, Log1p) {
  if (platform_ != "Host" && platform_ != "CUDA" && ty_ == F32) {
    abs_err_ = 0.001;
    rel_err_ = 0.001;
  }

  Run(Log1p, std::log1p);
}

XLA_TEST_P(ExhaustiveOpTest, Exp) {
  if (platform_ == "Host" && ty_ == F32) {
    // TODO(b/73142289): The vectorized Exp implementation gives results outside
    // our error spec in this range.
    known_incorrect_begin_ = 1107296256 + 11583654;
    known_incorrect_end_ = 1107296256 + 11629080;
  } else if (platform_ == "Host" && ty_ == BF16) {
    // TODO(jlebar): Is this a rounding error?  Why doesn't it occur on XLA:GPU?
    //
    // Mismatch on 88.5 (0x42b1).
    //   Expected 2.72491739e+38 (0x7f4d), but got inf (0x7f80).
    known_incorrect_begin_ = 0x42b1;
    known_incorrect_end_ = 0x42b2;
  }

  Run(Exp, std::exp);
}

XLA_TEST_P(ExhaustiveOpTest, Expm1) {
  // Expm1 has the same erroneous behavior on CPU as Exp.
  if (platform_ == "Host" && ty_ == F32) {
    // TODO(b/73142289): The vectorized Exp implementation gives results outside
    // our error spec in this range.
    known_incorrect_begin_ = 1107296256 + 11583654;
    known_incorrect_end_ = 1107296256 + 11629080;
  } else if (platform_ == "Host" && ty_ == BF16) {
    // TODO(jlebar): Is this a rounding error?  Why doesn't it occur on XLA:GPU?
    //
    // Mismatch on 88.5 (0x42b1).
    //   Expected 2.72491739e+38 (0x7f4d), but got inf (0x7f80).
    known_incorrect_begin_ = 0x42b1;
    known_incorrect_end_ = 0x42b2;
  }

  Run(Expm1, std::expm1);
}

// It feels a little overkill to exhaustively test sqrt and pow(x, 0.5), but
// this *did* find a bug, namely that some backends were assuming sqrt(x) ==
// pow(x, 0.5), but this is not true for x == -inf.
XLA_TEST_P(ExhaustiveOpTest, PowOneHalf) {
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); },
      +[](float x) { return std::pow(x, 0.5f); });
}

XLA_TEST_P(ExhaustiveOpTest, Rsqrt) {
  Run(
      Rsqrt, +[](float x) { return 1 / std::sqrt(x); });
}

XLA_TEST_P(ExhaustiveOpTest, Sqrt) {
  if (platform_ == "Host" || platform_ == "CUDA") {
    strict_signed_zeros_ = true;
  }

  Run(Sqrt, std::sqrt);
}

// TODO(jlebar): Add remaining trig functions.  Don't forget Atan2!
// TODO(jlebar): Test trig functions over complex inputs.
XLA_TEST_P(ExhaustiveOpTest, Tanh) { Run(Tanh, std::tanh); }

XLA_TEST_P(ExhaustiveOpTest, Erf) { Run(Erf, std::erf); }
XLA_TEST_P(ExhaustiveOpTest, Erfc) { Run(Erfc, std::erfc); }
XLA_TEST_P(ExhaustiveOpTest, ErfInv) { Run(ErfInv, HostErfInv); }
XLA_TEST_P(ExhaustiveOpTest, Digamma) {
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    rel_err_ = 0.01;
    abs_err_ = 0.01;
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
      if (absl::bit_cast<uint32>(x) == 0x00200000 ||
          absl::bit_cast<uint32>(x) == 0x80200000) {
        return std::copysign(std::numeric_limits<float>::max(), -x);
      }
      return HostDigamma(x);
    };
    Run(Digamma, host_digamma_with_gpu_ftz_errors);
  } else {
    Run(Digamma, HostDigamma);
  }
}
XLA_TEST_P(ExhaustiveOpTest, Lgamma) {
  // Our implementation gets within 0.0001 rel error except for ~20 denormal
  // inputs on GPU.  Anyway 0.001 rel error should be good enough for lgamma.
  if (platform_ == "CUDA" && (ty_ == F32 || ty_ == F16)) {
    rel_err_ = 0.001;
  }
  if (platform_ != "Host" && platform_ != "CUDA") {
    // TODO(b/123956399): This is a fairly high error, significantly higher than
    // we see on CPU/GPU.
    rel_err_ = 0.01;
    abs_err_ = 0.01;

    // Overflows for to inf for input 4.08500343e+36 (0x7c44af8e).
    if (ty_ == F32) {
      known_incorrect_begin_ = 0x7c44af8e;
      known_incorrect_end_ = 0x7c44af8e + 1;
    }
  }
  Run(Lgamma, std::lgamma);
}

XLA_TEST_P(ExhaustiveOpTest, Round) { Run(Round, std::round); }

std::vector<std::pair<int64, int64>> CreateExhaustiveF32Ranges() {
  // We break up the 2^32-element space into small'ish chunks to keep peak
  // memory usage low.
  std::vector<std::pair<int64, int64>> result;
  const int64 step = 1 << 25;
  for (int64 i = 0; i < (1l << 32); i += step) {
    result.push_back({i, i + step});
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(
    F32, ExhaustiveOpTest,
    ::testing::Combine(::testing::Values(F32),
                       ::testing::ValuesIn(CreateExhaustiveF32Ranges())));

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(
    F16, ExhaustiveOpTest,
    ::testing::Combine(::testing::Values(F16),
                       ::testing::Values(std::make_pair(0, 1 << 16))));
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
INSTANTIATE_TEST_SUITE_P(
    BF16, ExhaustiveOpTest,
    ::testing::Combine(::testing::Values(BF16),
                       ::testing::Values(std::make_pair(0, 1 << 16))));
#endif

}  // namespace
}  // namespace xla
