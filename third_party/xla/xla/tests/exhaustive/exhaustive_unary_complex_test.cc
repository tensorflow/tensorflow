/* Copyright 2020 The OpenXLA Authors.

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

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/tests/exhaustive/error_spec.h"
#include "xla/tests/exhaustive/exhaustive_op_test.h"
#include "xla/tests/exhaustive/exhaustive_op_test_base.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "tsl/platform/test.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

// T is the Primitive Type of the complex number
// Test parameter is a tuple containing
//   - primitive type under test,
//   - two FpValues representing the values for the real and imaginary
//     components. The complex numbers for the test input is the cartesian
//     product of the values represented by the two FpValues.
template <PrimitiveType T>
class ExhaustiveComplexUnaryTestBase
    : public ExhaustiveUnaryTest<T>,
      public ::testing::WithParamInterface<std::tuple<FpValues, FpValues>> {
 protected:
  using typename ExhaustiveUnaryTest<T>::NativeT;

 private:
  // Generates the input complex literal given the FpValues representation for
  // the real and imaginary components.
  void FillInput(std::array<Literal, 1>* input_literal) override {
    FpValues real_values = std::get<0>(GetParam());
    FpValues imag_values = std::get<1>(GetParam());
    if (VLOG_IS_ON(2)) {
      LOG(INFO) << this->SuiteName() << this->TestName() << " Values:";
      LOG(INFO) << "\treal values=" << real_values.ToString();
      LOG(INFO) << "\timag values=" << imag_values.ToString();
      LOG(INFO) << "\ttotal values to test="
                << real_values.GetTotalNumValues() *
                       imag_values.GetTotalNumValues();
    }

    absl::Span<NativeT> input_arr = (*input_literal)[0].data<NativeT>();
    uint64_t i = 0;
    for (auto real : real_values) {
      for (auto imag : imag_values) {
        input_arr[i] =
            NativeT(this->ConvertValue(real), this->ConvertValue(imag));
        ++i;
      }
    }
  }

  int64_t GetInputSize() override {
    FpValues real_values = std::get<0>(GetParam());
    FpValues imag_values = std::get<1>(GetParam());
    return real_values.GetTotalNumValues() * imag_values.GetTotalNumValues();
  }
};

using ExhaustiveC64UnaryTest = ExhaustiveComplexUnaryTestBase<C64>;

using ExhaustiveC128UnaryTest = ExhaustiveComplexUnaryTestBase<C128>;

#define UNARY_TEST_COMPLEX_64(test_name, ...)   \
  XLA_TEST_P(ExhaustiveC64UnaryTest, test_name) \
  __VA_ARGS__

UNARY_TEST_COMPLEX_64(Log, {
  ErrorSpecGen error_spec_gen = +[](complex64 x) {
    double abs_err, rel_err;
    // The reference implementation overflows to infinity for arguments near
    // FLT_MAX.
    if (std::abs(x) >= std::numeric_limits<float>::max()) {
      abs_err = rel_err = std::numeric_limits<float>::infinity();
    } else {
      double eps = std::numeric_limits<float>::epsilon();
      abs_err = 2 * eps;
      rel_err = 50 * eps;
    }

    // TODO(rmlarsen): see b/162664705 and b/138578594
    bool should_skip = std::isnan(x.real()) || std::isnan(x.imag());

    return ErrorSpec::Builder()
        .abs_err(abs_err)
        .rel_err(rel_err)
        .skip_comparison(should_skip)
        .build();
  };
  Run([](XlaOp x) { return Log(x); }, [](complex64 x) { return std::log(x); },
      error_spec_gen);
})

UNARY_TEST_COMPLEX_64(Sqrt, {
  ErrorSpecGen error_spec_gen = +[](complex64 x) {
    // The reference implementation overflows to infinity for arguments near
    // FLT_MAX.
    if (std::abs(x) >= std::numeric_limits<float>::max()) {
      float inf = std::numeric_limits<float>::infinity();
      return ErrorSpec::Builder().abs_err(inf).rel_err(inf).build();
    }
    // The reference implementation appears to be very poor on inputs with
    // subnormal entries. Allowing an absolute error of ~sqrt(FLT_DENORM_MIN)
    // allows such cases to pass, effectively letting the relative error
    // increase gradually until it reaches 100% at abs(x) == FLT_DENORM_MIN.
    return ErrorSpec::Builder()
        .abs_err(std::sqrt(std::numeric_limits<float>::denorm_min()))
        .rel_err(50 * std::numeric_limits<float>::epsilon())
        .build();
  };
  Run([](XlaOp x) { return Sqrt(x); }, [](complex64 x) { return std::sqrt(x); },
      error_spec_gen);
})

template <typename NativeT, typename ComponentNativeT>
double RsqrtCpuGpuAbsErr(NativeT x) {
  return std::sqrt(std::numeric_limits<ComponentNativeT>::min());
}

template <typename NativeT, typename ComponentNativeT>
double RsqrtCpuGpuRelErr(NativeT x) {
  // As noted above for Sqrt, the accuracy of sqrt degrades severely for
  // inputs with inputs with subnormals entries.
  constexpr double eps = std::numeric_limits<ComponentNativeT>::epsilon();
  constexpr double norm_min = std::numeric_limits<ComponentNativeT>::min();
  constexpr double denorm_min =
      std::numeric_limits<ComponentNativeT>::denorm_min();
  if (std::abs(x) < norm_min) {
    // Gradually loosen the relative tolerance as abs(x) becomes smaller
    // than norm_min, letting it reach 100% when abs(x) = 10 * denorm_min.
    return 10 * denorm_min / std::abs(x);
  }
  return 50 * eps;
}

UNARY_TEST_COMPLEX_64(Rsqrt, {
  ErrorSpecGen error_spec_gen = +[](complex64) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu()) {
    error_spec_gen = +[](complex64 x) {
      return ErrorSpec::Builder()
          .abs_err(RsqrtCpuGpuAbsErr<NativeT, ComponentNativeT>(x))
          .rel_err(RsqrtCpuGpuRelErr<NativeT, ComponentNativeT>(x))
          .skip_comparison(x.real() == 0.0f)
          .strict_signed_zeros(false)
          .build();
    };
  }

  if (IsGpu()) {
    error_spec_gen = +[](complex64 x) {
      return ErrorSpec::Builder()
          .abs_err(RsqrtCpuGpuAbsErr<NativeT, ComponentNativeT>(x))
          .rel_err(RsqrtCpuGpuRelErr<NativeT, ComponentNativeT>(x))
          .strict_signed_zeros(false)
          .build();
    };
  }

  Run([](XlaOp x) { return Rsqrt(x); },
      [](complex64 x) { return complex64(1, 0) / std::sqrt(x); },
      error_spec_gen);
})

UNARY_TEST_COMPLEX_64(Tanh, {
  ErrorSpecGen error_spec_gen = +[](complex64 x) {
    // This implementation of Tanh becomes less accurate when the denominator
    // is small.
    if (std::cosh(2 * x.real()) + std::cos(2 * x.imag()) < 1e-4) {
      return ErrorSpec::Builder().abs_err(5e-2).rel_err(5e-2).build();
    }

    return GetDefaultSpecGenerator()(x);
  };
  Run([](XlaOp x) { return Tanh(x); },
      +[](complex64 x) {
        // The current libc++ implementation of the complex tanh function
        // provides less accurate results when the denominator of a complex tanh
        // is small, due to floating point precision loss. To avoid this issue
        // for complex64 numbers, we cast it to and from a complex128 when
        // computing tanh.
        return static_cast<complex64>(std::tanh(static_cast<complex128>(x)));
      },
      error_spec_gen);
})

INSTANTIATE_TEST_SUITE_P(
    F32SpecialValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    F32SpecialAndNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::Values(GetNormals<float>(10000))));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndSpecialValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<float>(10000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(::testing::Values(GetNormals<float>(10000)),
                       ::testing::Values(GetNormals<float>(10000))));

// Tests a total of 40000 ^ 2 inputs, with 4000 ^ 2 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    F32LargeAndSmallMagnitudeNormalValues, ExhaustiveC64UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<float>(40000,
                                                                         4000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<float>(40000, 4000))));

#define UNARY_TEST_COMPLEX_128(test_name, ...)   \
  XLA_TEST_P(ExhaustiveC128UnaryTest, test_name) \
  __VA_ARGS__

UNARY_TEST_COMPLEX_128(Log, {
  ErrorSpecGen error_spec_gen = +[](complex128 x) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu() || IsGpu()) {
    error_spec_gen = +[](complex128 x) {
      // TODO(rmlarsen): see b/162664705 and b/138578594
      bool should_skip = std::isnan(x.real()) || std::isnan(x.imag());
      return ErrorSpec::Builder()
          .distance_err(1)
          .skip_comparison(should_skip)
          .strict_signed_zeros()
          .build();
    };
  }

  Run([](XlaOp x) { return Log(x); }, [](complex128 x) { return std::log(x); },
      error_spec_gen);
})

UNARY_TEST_COMPLEX_128(Sqrt, {
  ErrorSpecGen error_spec_gen = +[](complex128 x) {
    // TODO(rmlarsen): Investigate why we only get ~float accuracy here.

    // The reference implementation appears to be very poor on inputs with
    // subnormal entries. Allowing an absolute error of ~sqrt(DBL_DENORM_MIN)
    // allows such cases to pass, effectively letting the relative error
    // increase gradually until it reaches 100% at abs(x) == DBL_DENORM_MIN.
    return ErrorSpec::Builder()
        .abs_err(std::sqrt(std::numeric_limits<double>::denorm_min()))
        .rel_err(50 * std::numeric_limits<double>::epsilon())
        // TODO(b/138126045): Similar to the Tanh bug.
        .skip_comparison(std::abs(x) > std::numeric_limits<double>::max() / 2)
        .build();
  };
  Run([](XlaOp x) { return Sqrt(x); },
      [](complex128 x) { return std::sqrt(x); }, error_spec_gen);
})

UNARY_TEST_COMPLEX_128(Rsqrt, {
  ErrorSpecGen error_spec_gen = +[](complex128) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu()) {
    error_spec_gen = +[](complex128 x) {
      return ErrorSpec::Builder()
          .abs_err(RsqrtCpuGpuAbsErr<NativeT, ComponentNativeT>(x))
          .rel_err(RsqrtCpuGpuRelErr<NativeT, ComponentNativeT>(x))
#ifdef __aarch64__
          // TODO(b/365620546): ARM and x86 handle complex(inf, nan)
          // differently.
          .skip_comparison(x.real() == 0.0f ||
                           (std::isinf(x.real()) && std::isnan(x.imag())))
#else
          .skip_comparison(x.real() == 0.0f)
#endif
          .strict_signed_zeros(false)
          .build();
    };
  }

  if (IsGpu()) {
    error_spec_gen = +[](complex128 x) {
      return ErrorSpec::Builder()
          .abs_err(RsqrtCpuGpuAbsErr<NativeT, ComponentNativeT>(x))
          .rel_err(RsqrtCpuGpuRelErr<NativeT, ComponentNativeT>(x))
          .strict_signed_zeros(false)
          .build();
    };
  }

  Run([](XlaOp x) { return Rsqrt(x); },
      [](complex128 x) { return complex128(1, 0) / std::sqrt(x); },
      error_spec_gen);
})

UNARY_TEST_COMPLEX_128(Tanh, {
  ErrorSpecGen error_spec_gen = +[](complex128 x) {
    // TODO(b/138126045): Current libc++ implementation of the complex tanh
    //                    function returns (NaN, NaN) when the imaginary
    //                    component is more than half of the max value.
    // TODO(b/138750327): Current libc++ implementation of the complex tanh
    //                    function returns (1, 0) when the real component is
    //                    negative infinity, when it should return (-1, 0).
    bool should_skip = std::abs(x) > std::numeric_limits<double>::max() / 2;

    // TODO(rmlarsen): Investigate why we only get slightly better than
    // float accuracy here.
    return ErrorSpec::Builder()
        .abs_err(2 * std::numeric_limits<double>::min())
        .rel_err(2e-8)
        .skip_comparison(should_skip)
        .build();
  };

  Run([](XlaOp x) { return Tanh(x); },
      +[](complex128 x) { return std::tanh(x); }, error_spec_gen);
})

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::Values(GetNormals<double>(10000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<double>(10000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    F32NormalAndNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(::testing::Values(GetNormals<double>(10000)),
                       ::testing::Values(GetNormals<double>(10000))));

// Tests a total of 40000 ^ 2 inputs, with 2000 ^ 2 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveC128UnaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000))));

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
