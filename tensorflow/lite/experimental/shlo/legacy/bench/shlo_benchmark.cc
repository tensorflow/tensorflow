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

#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/lite/experimental/shlo/legacy/bench/util.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/test/util.h"

namespace stablehlo {
namespace benchmark {
namespace {

template <absl::Status (*op)(const Tensor&, Tensor&), ElementType element_type,
          int size>
void BM_SHLO(::benchmark::State& state) {
  Shape shape = {size};

  using ST = typename Storage<element_type>::Type;
  auto operand_values = GenerateRandomVector<ST>(size);
  decltype(operand_values) result_values(operand_values.size());

  Tensor operand(TensorType(Shape(shape), element_type),
                 std::data(operand_values));
  Tensor result(TensorType(Shape(shape), element_type),
                std::data(result_values));

  for (auto _ : state) {
    auto res = op(operand, result);
    CHECK_OK(res);
  }
}

template <absl::Status (*op)(const Tensor&, const Tensor&, Tensor&),
          ElementType element_type, int size>
void BM_SHLO(::benchmark::State& state) {
  Shape shape = {size};

  using ST = typename Storage<element_type>::Type;
  auto lhs_values = GenerateRandomVector<ST>(size);
  auto rhs_values = GenerateRandomVector<ST>(size);
  decltype(rhs_values) result_values(rhs_values.size());

  Tensor lhs(TensorType(Shape(shape), element_type), std::data(lhs_values));
  Tensor rhs(TensorType(Shape(shape), element_type), std::data(rhs_values));
  Tensor result(TensorType(Shape(shape), element_type),
                std::data(result_values));

  for (auto _ : state) {
    auto res = op(lhs, rhs, result);
    CHECK_OK(res);
  }
}

template <absl::Status (*op)(const QuantizedTensor&, QuantizedTensor&),
          ElementType storage_type, ElementType expressed_type, int size>
void BM_SHLO(::benchmark::State& state) {
  Shape shape = {size};
  QuantizedParameter quantized_parameter = {.scale = 0.1, .zero_point = 0};

  using ET = typename Storage<expressed_type>::Type;

  auto operand_values = GenerateRandomVector<ET>(size);

  auto operand_quant_values =
      testing::QuantizeVector<storage_type, expressed_type>(
          operand_values, quantized_parameter);
  decltype(operand_quant_values) result_quant_values(
      operand_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));

  QuantizedTensor operand(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      operand_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  for (auto _ : state) {
    auto res = op(operand, result);
    CHECK_OK(res);
  }
}

template <absl::Status (*op)(const QuantizedTensor&, const QuantizedTensor&,
                             QuantizedTensor&),
          ElementType storage_type, ElementType expressed_type, int size>
void BM_SHLO(::benchmark::State& state) {
  Shape shape = {size};
  QuantizedParameter quantized_parameter = {.scale = 0.1, .zero_point = 0};

  using ET = typename Storage<expressed_type>::Type;

  auto lhs_values = GenerateRandomVector<ET>(size);
  auto rhs_values = GenerateRandomVector<ET>(size);

  auto lhs_quant_values = testing::QuantizeVector<storage_type, expressed_type>(
      lhs_values, quantized_parameter);
  auto rhs_quant_values = testing::QuantizeVector<storage_type, expressed_type>(
      rhs_values, quantized_parameter);
  decltype(rhs_quant_values) result_quant_values(rhs_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));

  QuantizedTensor lhs(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      lhs_quant_values.data());
  QuantizedTensor rhs(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      rhs_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  for (auto _ : state) {
    auto res = op(lhs, rhs, result);
    CHECK_OK(res);
  }
}

#define BENCHMARK_OP_HELPER(Op, ElementType)    \
  BENCHMARK(BM_SHLO<Op, ElementType, 8 * KB>);  \
  BENCHMARK(BM_SHLO<Op, ElementType, 16 * KB>); \
  BENCHMARK(BM_SHLO<Op, ElementType, 32 * KB>); \
  BENCHMARK(BM_SHLO<Op, ElementType, 64 * KB>);

#define BENCHMARK_QOP_HELPER(Op, StorageType, ExpressedType)   \
  BENCHMARK(BM_SHLO<Op, StorageType, ExpressedType, 8 * KB>);  \
  BENCHMARK(BM_SHLO<Op, StorageType, ExpressedType, 16 * KB>); \
  BENCHMARK(BM_SHLO<Op, StorageType, ExpressedType, 32 * KB>); \
  BENCHMARK(BM_SHLO<Op, StorageType, ExpressedType, 64 * KB>);

#define BENCHMARK_OP(Op)                                            \
  BENCHMARK_OP_HELPER(Op, ElementType::kF32);                       \
  BENCHMARK_OP_HELPER(Op, ElementType::kF16);                       \
  BENCHMARK_OP_HELPER(Op, ElementType::kBF16);                      \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI8, ElementType::kF32);   \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI16, ElementType::kF32);  \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI32, ElementType::kF32);  \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI8, ElementType::kF16);   \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI16, ElementType::kF16);  \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI32, ElementType::kF16);  \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI8, ElementType::kBF16);  \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI16, ElementType::kBF16); \
  BENCHMARK_QOP_HELPER(Op, ElementType::kSI32, ElementType::kBF16);

BENCHMARK_OP(Abs);
BENCHMARK_OP(Add);

}  // namespace
}  // namespace benchmark
}  // namespace stablehlo

// Run the benchmark
BENCHMARK_MAIN();
