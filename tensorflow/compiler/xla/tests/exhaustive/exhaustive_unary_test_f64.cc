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

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "tensorflow/compiler/xla/util.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {

// Exhaustive test for unary operations for double.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - FpValues representing a set of double values.

class ExhaustiveF64UnaryTest : public ExhaustiveUnaryTest<F64>,
                               public ::testing::WithParamInterface<FpValues> {
 private:
  int64_t GetInputSize() override {
    FpValues values = GetParam();
    return values.GetTotalNumValues();
  }

  void FillInput(std::array<Literal, 1>* input_literal) override {
    FpValues fp_values = GetParam();
    int64_t input_size = (*input_literal)[0].element_count();
    LOG(INFO) << "Checking fp values " << fp_values.ToString() << ", "
              << input_size;
    absl::Span<double> input_arr = (*input_literal)[0].data<double>();

    uint64_t i = 0;
    for (auto bits : fp_values) {
      input_arr[i] = this->ConvertAndReplaceKnownIncorrectValueWith(bits, 1);
      ++i;
    }
    CHECK_EQ(i, input_size);
  }
};

#define UNARY_TEST_FLOAT_64(test_name, ...)     \
  XLA_TEST_P(ExhaustiveF64UnaryTest, test_name) \
  __VA_ARGS__

UNARY_TEST_FLOAT_64(Log, { Run(Log, std::log); })

UNARY_TEST_FLOAT_64(Log1p, { Run(Log1p, std::log1p); })

UNARY_TEST_FLOAT_64(Exp, { Run(Exp, std::exp); })

UNARY_TEST_FLOAT_64(Expm1, { Run(Expm1, std::expm1); })

// TODO(b/138385863): Turn on the test for GPU after fixing the bug.
UNARY_TEST_FLOAT_64(DISABLED_ON_GPU(PowOneHalf), {
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); },
      +[](double x) { return std::pow(x, 0.5); });
})

UNARY_TEST_FLOAT_64(Rsqrt, {
  Run(
      Rsqrt, +[](double x) { return 1 / std::sqrt(x); });
})

UNARY_TEST_FLOAT_64(Sqrt, { Run(Sqrt, std::sqrt); })

UNARY_TEST_FLOAT_64(Acosh, { Run(Acosh, std::acosh); })

UNARY_TEST_FLOAT_64(Asinh, { Run(Asinh, std::asinh); })

UNARY_TEST_FLOAT_64(Atanh, { Run(Atanh, std::atanh); })

UNARY_TEST_FLOAT_64(Acos, { Run(Acos, std::acos); })

UNARY_TEST_FLOAT_64(Asin, { Run(Asin, std::asin); })

UNARY_TEST_FLOAT_64(Cosh, { Run(Cosh, std::cosh); })

UNARY_TEST_FLOAT_64(Sinh, { Run(Sinh, std::sinh); })

UNARY_TEST_FLOAT_64(Tanh, {
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

UNARY_TEST_FLOAT_64(Cos, { Run(Cos, std::cos); })

UNARY_TEST_FLOAT_64(Sin, { Run(Sin, std::sin); })

UNARY_TEST_FLOAT_64(Tan, { Run(Tan, std::tan); })

UNARY_TEST_FLOAT_64(Round, { Run(Round, std::round); })

UNARY_TEST_FLOAT_64(Erf, {
  Run(Erf, std::erf, [](NativeT x) { return ErrorSpec{1e-20, 1e-20}; });
})

UNARY_TEST_FLOAT_64(Erfc, {
  Run(Erfc, std::erfc, [](NativeT x) { return ErrorSpec{1e-20, 1e-20}; });
})

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()));

INSTANTIATE_TEST_SUITE_P(NormalValues, ExhaustiveF64UnaryTest,
                         ::testing::Values(GetNormals<double>(1000)));

// Tests a total of 4000000000 inputs, with 16000000 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<double>(
        4000000000ull, 16000000)));

}  // namespace exhaustive_op_test
}  // namespace xla
