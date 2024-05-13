/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <vector>

#include "absl/log/absl_check.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/ops/benchmark_util.h"
#include "tensorflow/lite/experimental/shlo/ops/is_finite.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
#include "tensorflow/lite/experimental/shlo/tensor_with_data.h"

namespace shlo_ref {
namespace {

void BM_IsFinite(benchmark::State& state, DimensionSize num_elements,
                 const Tensor& operand) {
  IsFiniteOp op = Create(IsFiniteOp::Attributes{});

  Tensor result = Tensor{.type = TensorType{.shape = Shape{{num_elements}},
                                            .element_type = DataType::kI1}};
  ABSL_CHECK_OK(Prepare(op, operand, result));

  std::vector<std::byte> result_values(result.SizeInBytes());
  result.data = result_values.data();

  for (auto _ : state) {
    ABSL_CHECK_OK(Evaluate(op, operand, result));
  }
}

template <DataType data_type>
void BM_IsFinite(benchmark::State& state) {
  const DimensionSize num_elements = state.range(0);

  auto operand_values = GenerateRandomVector<data_type>(num_elements);
  auto operand =
      TensorWithData::Create<data_type>(Shape{{num_elements}}, operand_values);

  BM_IsFinite(state, num_elements, operand.tensor());
}

template <DataType storage_type, DataType expressed_type>
void BM_IsFiniteQuantized(benchmark::State& state) {
  const DimensionSize num_elements = state.range(0);

  auto operand_values = GenerateRandomVector<expressed_type>(num_elements);
  auto operand = TensorWithData::Create<storage_type, expressed_type>(
      Shape{{num_elements}}, operand_values, 0.1, 0);

  BM_IsFinite(state, num_elements, operand.tensor());
}

BENCHMARK(BM_IsFinite<DataType::kBF16>)
    ->RangeMultiplier(2)
    ->Range(KiB(8), KiB(64));
BENCHMARK(BM_IsFinite<DataType::kF16>)
    ->RangeMultiplier(2)
    ->Range(KiB(8), KiB(64));
BENCHMARK(BM_IsFinite<DataType::kF32>)
    ->RangeMultiplier(2)
    ->Range(KiB(8), KiB(64));

// IsFinite will be the same regardless of quantization parameters, so only
// benchmark one combination.
BENCHMARK(BM_IsFiniteQuantized<DataType::kSI16, DataType::kF32>)
    ->RangeMultiplier(2)
    ->Range(KiB(8), KiB(64));

}  // namespace
}  // namespace shlo_ref

BENCHMARK_MAIN();
