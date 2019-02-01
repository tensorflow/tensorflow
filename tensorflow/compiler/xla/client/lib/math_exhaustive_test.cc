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

#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

using Eigen::half;

struct Testcase {
  Testcase(string name, XlaOp (*op)(XlaOp), float (*host_op)(float))
      : name(name), op(op), host_op(host_op) {}

  Testcase& set_tolerance(float abs_err, float rel_err) {
    error.abs = abs_err;
    error.rel = rel_err;
    return *this;
  }

  Testcase& set_relaxed_nans() {
    error.relaxed_nans = true;
    return *this;
  }

  Testcase& set_fewer_infs_ok() {
    error.fewer_infs_ok = true;
    return *this;
  }

  Testcase& set_skip_pos_inf() {
    skip_pos_inf = true;
    return *this;
  }

  Testcase& set_skip_neg_inf() {
    skip_neg_inf = true;
    return *this;
  }

  Testcase& set_skip_infs() {
    skip_pos_inf = true;
    skip_neg_inf = true;
    return *this;
  }

  Testcase& set_skip_neg_zero() {
    skip_neg_zero = true;
    return *this;
  }

  string name;
  XlaOp (*op)(XlaOp);
  float (*host_op)(float);

  ErrorSpec error{0.01, 0.01};

  // If true, don't test +/-infinity or negative 0.
  bool skip_pos_inf = false;
  bool skip_neg_inf = false;
  bool skip_neg_zero = false;
};

void PrintTo(const Testcase& tc, std::ostream* os) { *os << tc.name; }

class MathExhaustiveTest : public ClientLibraryTestBase,
                           public ::testing::WithParamInterface<Testcase> {
 public:
  MathExhaustiveTest() {
    // Disable fast-math, otherwise we get the wrong results for e.g.
    // sqrt(-inf).
    SetFastMathDisabled(true);
  }
};

// Checks a function's behavior on all fp16 values.
//
// TODO(jlebar): asin and lgamma tests fail on interpreter.
XLA_TEST_P(MathExhaustiveTest, DISABLED_ON_INTERPRETER(F16)) {
  const Testcase& tc = GetParam();
  XlaBuilder b(TestName());

  std::vector<half> input;
  for (uint32 i = 0; i < 1 << 16; ++i) {
    half h;
    h.x = i;

    // If we're not using infinity as an input, use 0 as a placeholder rather
    // than simply skipping this element.  We do this because when the test
    // framework reports an incorrect answer, it tells us which index failed.
    // So long as our inputs are a simple list of all possible float16s, we can
    // convert an index to a half with e.g. the following Python:
    //
    //   np.frombuffer(array('H', [12345]), dtype=np.float16)[0]
    //
    // but as soon as our list of inputs has any gaps, this doesn't work.
    if (std::isinf(static_cast<float>(h)) &&
        ((tc.skip_pos_inf && h > half{0}) ||
         (tc.skip_neg_inf && h < half{0}))) {
      h = half{0};
    }

    if (h == half{0} && tc.skip_neg_zero &&
        std::signbit(static_cast<float>(h))) {
      h = half{0};
    }

    input.push_back(h);
  }

  std::vector<half> expected_result;
  for (const auto& h : input) {
    expected_result.push_back(
        static_cast<half>(tc.host_op(static_cast<float>(h))));
  }

  XlaOp param = AddParam(LiteralUtil::CreateR1<half>(input), &b);
  tc.op(param);
  ComputeAndCompareR1<half>(&b, expected_result, {}, tc.error);
}

// TODO(b/123355973): The following tests from math.cc are missing.
//
// - Many failures.
//
//   Testcase{"acosh", Acosh, std::acosh}.set_relaxed_nans(),
//   Testcase{"asinh", Asinh, std::asinh},
//   Testcase{"sinh", Sinh, std::sinh},
//   Testcase{"cosh", Cosh, std::cosh}.set_fewer_infs_ok(),
//   Testcase{"round_to_even", RoundToEven,
//            [](float x) { return std::nearbyint(x / 2) * 2; }},
//
// - No equivalent std function to compare with.
//
//   Testcase{"erfinv", ErfInv, std::erfinv},
//   Testcase{"digamma", Digamma, std::digamma},
//
// - Needs a special test (function takes two args, and simply computing in f32
//   and downcasting to f16 doesn't give the correct answer).
//
//   Testcase{"nextafter", NextAfter, std::nextafter},
//
// TODO(b/123355973): Test math functions not from math.cc (e.g. log).
// TODO(b/123355973): Test bf16 and f32.
// TODO(b/123355973): Get rid of skip_infs / skip_neg_zero below if possible.
// TODO(b/123355973): Reduce lgamma error if possible; it is very high.
INSTANTIATE_TEST_CASE_P(
    MathExhaustiveTest_Instantiation, MathExhaustiveTest,
    ::testing::ValuesIn(std::vector<Testcase>{
        Testcase{"sqrt", Sqrt, std::sqrt}.set_skip_neg_inf(),
        Testcase{"rsqrt", Rsqrt, [](float x) { return 1 / std::sqrt(x); }}
            .set_tolerance(0.05, 0.05)
            .set_skip_infs()
            .set_skip_neg_zero(),
        Testcase{"square", Square, [](float x) { return x * x; }},
        Testcase{"reciprocal", Reciprocal, [](float x) { return 1 / x; }},
        Testcase{"erf", Erf, std::erf}.set_tolerance(0.001, 0.0001),
        Testcase{"erfc", Erfc, std::erfc}.set_tolerance(0.001, 0.0001),
        Testcase{"lgamma", Lgamma, std::lgamma}
            .set_tolerance(0.1, 0.15)
            .set_fewer_infs_ok(),
        Testcase{"asin", Asin, std::asin}.set_skip_infs(),
        Testcase{"acos", Acos, std::acos}.set_skip_infs(),
        Testcase{"atan", Atan, std::atan},
        Testcase{"tan", Tan, std::tan}.set_tolerance(0.05, 0.05),
    }));

}  // namespace
}  // namespace xla
