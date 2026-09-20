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
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

#include "xla/error_spec.h"
#include "xla/fp_util.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/complex_unary_op_samples.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

template <class>
constexpr bool dependent_false = false;

class ComplexUnaryOpTest : public ClientLibraryTestRunnerMixin<
                               HloInterpreterReferenceMixin<HloTestBase>> {
 protected:
  // Disable constant folding to ensure we test the actual backend
  // implementation. Otherwise, constant folding pre-computes results using
  // HloEvaluator's reference implementation (std c++), not the backend under
  // test.
  void SetUp() override {
    ClientLibraryTestRunnerMixin::SetUp();
    mutable_debug_options()->add_xla_disable_hlo_passes("constant_folding");
  }
  template <typename T, size_t index, typename... Types>
  std::vector<T> get_column(const std::vector<std::tuple<Types...>>& table) {
    std::vector<T> column;
    absl::c_transform(table, std::back_inserter(column), [](const auto& item) {
      return static_cast<T>(std::get<index>(item));
    });
    return column;
  }

  template <typename T, typename S>
  void scale_column(std::vector<T>& column, const std::vector<S>& scales) {
    absl::c_transform(column, scales, column.begin(),
                      [](const T& lhs, const S& rhs) { return lhs * rhs; });
  }

  template <typename C>
  void UnaryTestHelper(XlaOp (*Op)(const XlaOp operand)) {
    using InputType = typename C::InputType;
    using OutputType = typename C::OutputType;
    using FloatType = typename C::FloatType;

    float atol;
    // log(10)/log(2) = 3.3219...
    constexpr int precision_deficiency =
        static_cast<int>(C::dps_deficiency * 3.3219280948873626);
    // precision_deficiency defines a slack allowed when comparing a
    // result value against expected value that is known to be
    // inaccurate to some extent.
    if constexpr (std::is_same_v<FloatType, float>) {
      atol = std::ldexp(1e-6f, precision_deficiency);
    } else if constexpr (std::is_same_v<FloatType, double>) {
      atol = std::ldexp(1e-15f, precision_deficiency);
    } else {
      static_assert(dependent_false<FloatType>);
    }

    XlaBuilder builder(TestName());
    auto table = C().get();
    auto inputs_vec = get_column<InputType, 0>(table);
    auto expected_vec = get_column<OutputType, 1>(table);
    auto scales_vec = get_column<FloatType, 2>(table);
    scale_column(expected_vec, scales_vec);

    auto inputs = ConstantR1<InputType>(&builder, inputs_vec);
    auto scales = ConstantR1<FloatType>(&builder, scales_vec);
    Literal expected = LiteralUtil::CreateR1<OutputType>(expected_vec);

    if constexpr (std::is_same_v<OutputType, FloatType>) {
      auto results = Op(inputs);
      Mul(results, scales);
      ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(atol));
    } else {
      auto results = Op(inputs);
      auto re = Mul(Real(results), scales);
      auto im = Mul(Imag(results), scales);
      Complex(re, im);
      ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(atol));
    }
  }

  template <typename ComplexT>
  void TestComplexTanhUlps();
};

TEST_F(ComplexUnaryOpTest, Log1pTest) {
  UnaryTestHelper<complex_unary_op_samples::Log1p<float>>(
      [](XlaOp x) { return Log1p(x); });
  UnaryTestHelper<complex_unary_op_samples::Log1p<double>>(
      [](XlaOp x) { return Log1p(x); });
}

TEST_F(ComplexUnaryOpTest, TanTest) {
  UnaryTestHelper<complex_unary_op_samples::Tan<float>>(
      [](XlaOp x) { return Tan(x); });
  UnaryTestHelper<complex_unary_op_samples::Tan<double>>(
      [](XlaOp x) { return Tan(x); });
}

TEST_F(ComplexUnaryOpTest, AsinTest) {
  UnaryTestHelper<complex_unary_op_samples::Asin<float>>(
      [](XlaOp x) { return Asin(x); });
  UnaryTestHelper<complex_unary_op_samples::Asin<double>>(
      [](XlaOp x) { return Asin(x); });
}

TEST_F(ComplexUnaryOpTest, AsinhTest) {
  UnaryTestHelper<complex_unary_op_samples::Asinh<float>>(
      [](XlaOp x) { return Asinh(x); });
  UnaryTestHelper<complex_unary_op_samples::Asinh<double>>(
      [](XlaOp x) { return Asinh(x); });
}

TEST_F(ComplexUnaryOpTest, ExpTest) {
  UnaryTestHelper<complex_unary_op_samples::Exp<float>>(
      [](XlaOp x) { return Exp(x); });
  UnaryTestHelper<complex_unary_op_samples::Exp<double>>(
      [](XlaOp x) { return Exp(x); });
}

template <typename T>
struct ComplexTanhTestCase {
  std::complex<T> input;
  // Maximum allowed ULP error on CPU and GPU.
  // TODO(phawkins): Tighten near-pole cases to 2 ULPs once MLIR's
  // ComplexToStandard pass is fixed.
  // tanh(a + bi) = sinh(a + bi) / cosh(a + bi). Multiplying numerator and
  // denominator by 2*cosh(a - bi) gives denominator
  // 2*cosh(a + bi)*cosh(a - bi) = cosh(2a) + cos(2b), using the identity
  // 2*cosh(u)*cosh(v) = cosh(u + v) + cosh(u - v) and cosh(2bi) = cos(2b).
  // Using cos(2b) = 2*cos^2(b) - 1,
  // cosh(2a) + cos(2b) = (cosh(2a) - 1) + 2*cos^2(b).
  // Doubling numerator and denominator (so the real numerator is
  // 2*sinh(2a) = expm1(2a) - expm1(-2a)) gives denominator:
  //   2*(cosh(2a) - 1) + 4*cos^2(b)
  // which avoids cancellation near poles (where cos(2b) ≈ -1).
  // However, evaluating 2*(cosh(2a) - 1) as expm1(2a) + expm1(-2a) cancels
  // as a -> 0. Using (e^(2a) - 1)*(e^(-2a) - 1) = 2 - 2*cosh(2a), that term
  // can be computed via multiplication without cancellation:
  //   denom = -expm1(2a) * expm1(-2a) + 4 * cos(b)^2.
  int64_t cpu_gpu_max_ulps = 2;
  // Maximum allowed ULP error across all TPU generations (including older TPUs
  // v2-v5p).
  // TODO(phawkins): Lower the default bound to 4 (or 2) on newer TPUs (TPU v6e,
  // TPU 7x, and later).
  int64_t tpu_max_ulps = 30;
};

template <typename T>
std::vector<ComplexTanhTestCase<T>> GetComplexTanhTestInputs() {
  return {
      // Small inputs where (exp(a))^2 - (exp(-a))^2 suffered precision loss
      // due to exp2 cancellation on older TPUs:
      {{T(0.0017180424), T(0.0017180424)}},
      {{T(-0.0017180424), T(0.0017180424)}},
      {{T(0.0017180424), T(-0.0017180424)}},
      {{T(-0.0017180424), T(-0.0017180424)}},

      // Other small magnitudes:
      {{T(1e-5), T(1e-5)}},
      {{T(-1e-5), T(1e-5)}},
      {{T(1e-4), T(1e-4)}},
      {{T(-1e-4), T(-1e-4)}},
      {{T(1e-3), T(1e-3)}},
      {{T(-1e-3), T(-1e-3)}},
      // (0.01, 0.01): Older TPUs (v2-v5p) experience up to 602 ULPs due to
      // expm1 approximation error. TPU v6+ achieves <= 2 ULPs.
      {{T(1e-2), T(1e-2)}, /*cpu_gpu=*/2, /*tpu=*/650},
      // (0.05, 0.05): Older TPUs achieve <= 80 ULPs (72 observed). TPU v6+
      // achieves <= 2 ULPs.
      {{T(0.05), T(0.05)}, /*cpu_gpu=*/2, /*tpu=*/80},

      // Points near or on axes:
      {{T(0.0), T(0.0)}},
      {{T(1e-3), T(0.0)}},
      {{T(0.0), T(1e-3)}},
      {{T(-1e-3), T(0.0)}},
      {{T(0.0), T(-1e-3)}},

      // Moderate inputs:
      {{T(0.5), T(0.5)}},
      {{T(-0.5), T(0.5)}},
      {{T(1.0), T(1.0)}},
      {{T(2.0), T(-1.5)}},

      // Inputs near poles (b ≈ ±pi/2) where cos(b) ≈ 0 and
      // expm1(2a) + expm1(-2a) cancellation causes up to 131 ULPs on CPU/GPU.
      // TODO(phawkins): Tighten cpu_gpu bounds to 2 ULPs once MLIR's
      // ComplexToStandard pass is fixed. The fix is to compute the
      // denominator's
      // exponential term as (expm1(a) - expm1(-a))^2 instead of
      // expm1(2a) + expm1(-2a), avoiding catastrophic cancellation when a -> 0.
      // On older TPUs (v2-v5p), 0.002 achieves <= 22 ULPs, 0.005 achieves <=
      // 125 ULPs.
      // On TPU v6+, all achieve <= 4 ULPs.
      {{T(0.002), T(1.5707963267948966)}, /*cpu_gpu=*/150},
      {{T(-0.002), T(1.5707963267948966)}, /*cpu_gpu=*/150},
      {{T(0.002), T(-1.5707963267948966)}, /*cpu_gpu=*/150},
      {{T(0.005), T(1.5707963267948966)}, /*cpu_gpu=*/150, /*tpu=*/150},
      // Input near pole (b ≈ pi/2):
      // Older TPUs achieve <= 130 ULPs (111 observed); TPU v6+ achieves <= 2
      // ULPs; H100 and B200 achieve <= 3 ULPs.
      {{T(0.11656774), T(1.7330385)}, /*cpu_gpu=*/4, /*tpu=*/130},

      // Large inputs (overflow handling, Re(z) > 15 region where tanh(z) -> +/-
      // 1):
      // Older TPUs achieve <= 80 ULPs (74 observed); TPU v6+ achieves <= 16
      // ULPs.
      {{T(15.0), T(0.5)}, /*cpu_gpu=*/2, /*tpu=*/80},
      {{T(-15.0), T(0.5)}, /*cpu_gpu=*/2, /*tpu=*/80},
      {{T(20.0), T(1.0)}},
      {{T(-20.0), T(1.0)}},
  };
}

template <typename ComplexT>
void ComplexUnaryOpTest::TestComplexTanhUlps() {
  using RealT = typename ComplexT::value_type;
  std::vector<ComplexTanhTestCase<RealT>> cases =
      GetComplexTanhTestInputs<RealT>();
  std::vector<ComplexT> xs;
  xs.reserve(cases.size());
  for (const auto& c : cases) {
    xs.push_back(c.input);
  }
  XlaBuilder builder(TestName());
  Literal input_literal = LiteralUtil::CreateR1<ComplexT>(xs);
  auto a = Parameter(&builder, 0, input_literal.shape(), "a");
  Tanh(a);
  std::vector<const Literal*> args = {&input_literal};
  ASSERT_OK_AND_ASSIGN(Literal actual,
                       this->ExecuteAndTransfer(&builder, args));
  const bool is_tpu = test::DeviceTypeIs(test::kTpu);
  for (int64_t i = 0; i < cases.size(); ++i) {
    ComplexT act = actual.Get<ComplexT>({i});
    std::complex<double> zd(static_cast<double>(xs[i].real()),
                            static_cast<double>(xs[i].imag()));
    std::complex<double> ref = std::tanh(zd);
    ComplexT exp(static_cast<RealT>(ref.real()),
                 static_cast<RealT>(ref.imag()));

    auto real_ulps = UlpDistance(act.real(), exp.real());
    auto imag_ulps = UlpDistance(act.imag(), exp.imag());
    ASSERT_TRUE(real_ulps.has_value())
        << "NaN/Inf mismatch on real part for input " << xs[i];
    ASSERT_TRUE(imag_ulps.has_value())
        << "NaN/Inf mismatch on imag part for input " << xs[i];

    int64_t max_ulps;
    if constexpr (std::is_same_v<ComplexT, complex128>) {
      // Host libc++ std::complex::tanh suffers from catastrophic cancellation
      // near poles (b ≈ ±pi/2). For complex64, computing the reference in
      // double avoids this, but for complex128 there is no higher precision
      // type to use for the reference, leading to large observed ULP errors.
      max_ulps = 100000;
    } else {
      max_ulps = is_tpu ? cases[i].tpu_max_ulps : cases[i].cpu_gpu_max_ulps;
    }

    if (test::DeviceTypeIs(test::kInterpreter)) {
      // Similarly, the interpreter evaluates tanh via host libc++
      // std::complex<float>, suffering from the same cancellation near poles.
      max_ulps = std::max(max_ulps, int64_t{30000});
    }

    EXPECT_LE(*real_ulps, max_ulps)
        << "Real part ULP error exceeded for input " << xs[i]
        << ": actual=" << act.real() << ", expected=" << exp.real()
        << ", ulp_distance=" << *real_ulps;
    EXPECT_LE(*imag_ulps, max_ulps)
        << "Imag part ULP error exceeded for input " << xs[i]
        << ": actual=" << act.imag() << ", expected=" << exp.imag()
        << ", ulp_distance=" << *imag_ulps;
  }
}

TEST_F(ComplexUnaryOpTest, TanhC64s) { TestComplexTanhUlps<complex64>(); }

TEST_F(ComplexUnaryOpTest, TanhC128s) { TestComplexTanhUlps<complex128>(); }

}  // namespace
}  // namespace xla
