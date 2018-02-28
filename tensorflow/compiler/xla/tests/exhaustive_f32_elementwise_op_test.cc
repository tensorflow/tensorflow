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

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/casts.h"

namespace xla {
namespace {
class ExhaustiveF32ElementwiseOpTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::pair<int64, int64>> {
 protected:
  ErrorSpec error_spec_{0.0001, 0.0001, /*relaxed_nans=*/true};

  template <typename EnqueueOpTy>
  void ExhaustivelyTestF32Op(EnqueueOpTy enqueue_op,
                             float (*evaluate_op)(float),
                             std::pair<int64, int64> known_incorrect_range) {
    int64 begin, end;
    std::tie(begin, end) = GetParam();
    int64 input_size = end - begin;
    LOG(INFO) << "Checking range [" << begin << ", " << end << ")";

    ComputationBuilder builder(client_, TestName());

    std::unique_ptr<Literal> input_literal =
        Literal::CreateFromDimensions(F32, {input_size});
    for (int64 i = begin; i < end; i++) {
      if (i >= known_incorrect_range.first &&
          i < known_incorrect_range.second) {
        // If the operation is known to be buggy on a specific input clamp that
        // input to 0 under the assumption that the op is at least correct on 0.
        input_literal->Set({i - begin}, 0.0f);
      } else {
        input_literal->Set({i - begin}, tensorflow::bit_cast<float, int>(i));
      }
    }

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                            client_->TransferToServer(*input_literal));

    auto input = builder.Parameter(0, input_literal->shape(), "input");
    enqueue_op(&builder, input);

    std::vector<float> expected_result;
    expected_result.reserve(input_size);
    for (int64 i = 0; i < input_size; i++) {
      expected_result.push_back(evaluate_op(input_literal->Get<float>({i})));
    }

    ComputeAndCompareR1<float>(&builder, expected_result, {input_data.get()},
                               error_spec_);
  }
};

XLA_TEST_P(ExhaustiveF32ElementwiseOpTest, LogF32) {
#ifdef XLA_TEST_BACKEND_CPU
  // TODO(b/73141998): The vectorized Log implementation gives results outside
  // our error spec in this range (these numbers are bitwise representations of
  // floats expressed as a zero extended int64):
  std::pair<int64, int64> known_incorrect_range = {1, 8315654};
#else
  std::pair<int64, int64> known_incorrect_range = {0, 0};
#endif

  ExhaustivelyTestF32Op(
      [](ComputationBuilder* builder, const ComputationDataHandle& input) {
        builder->Log(input);
      },
      std::log, known_incorrect_range);
}

XLA_TEST_P(ExhaustiveF32ElementwiseOpTest, ExpF32) {
#ifdef XLA_TEST_BACKEND_CPU
  // TODO(b/73142289): The vectorized Exp implementation gives results outside
  // our error spec in this range (these numbers are bitwise representations of
  // floats expressed as a zero extended int64):
  std::pair<int64, int64> known_incorrect_range = {1107296256 + 11583654,
                                                   1107296256 + 11629080};
#else
  std::pair<int64, int64> known_incorrect_range = {0, 0};
#endif

  ExhaustivelyTestF32Op(
      [](ComputationBuilder* builder, const ComputationDataHandle& input) {
        builder->Exp(input);
      },
      std::exp, known_incorrect_range);
}

XLA_TEST_P(ExhaustiveF32ElementwiseOpTest, TanhF32) {
  ExhaustivelyTestF32Op(
      [](ComputationBuilder* builder, const ComputationDataHandle& input) {
        builder->Tanh(input);
      },
      std::tanh, /*known_incorrect_range=*/{0, 0});
}

std::vector<std::pair<int64, int64>> CreateExhaustiveParameters() {
  // We break up the 2^32-element space into small'ish chunks to keep peak
  // memory usage low.
  std::vector<std::pair<int64, int64>> result;
  const int64 step = 1 << 25;
  for (int64 i = 0; i < (1l << 32); i += step) {
    result.push_back({i, i + step});
  }
  return result;
}

INSTANTIATE_TEST_CASE_P(ExhaustiveF32ElementwiseOpTestInstance,
                        ExhaustiveF32ElementwiseOpTest,
                        ::testing::ValuesIn(CreateExhaustiveParameters()));
}  // namespace
}  // namespace xla
