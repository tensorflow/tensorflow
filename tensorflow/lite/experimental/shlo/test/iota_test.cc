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

#include <initializer_list>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "tensorflow/lite/experimental/shlo/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/src/debug.h"  // IWYU pragma: keep, b/321245930
#include "tensorflow/lite/experimental/shlo/src/storage.h"
#include "tensorflow/lite/experimental/shlo/test/util.h"

namespace stablehlo {
namespace testing {

template <ElementType element_type>
void test(std::initializer_list<DimensionSize>&& shape,
          DimensionSize iota_dimension,
          std::vector<typename Storage<element_type>::Type>&& expected_values) {
  Tensor expected(TensorType(Shape(shape), element_type),
                  expected_values.data());

  std::vector<typename Storage<element_type>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), element_type), result_values.data());

  auto res = Iota(iota_dimension, result);
  if (!res.ok()) {
    ABSL_LOG(INFO) << "Failure: " << res;
  }
  ASSERT_EQ(res.ok(), true);

  if (result != expected) {
    ABSL_LOG(INFO) << "iota_dimension=" << iota_dimension;
    LOG(INFO) << "expected=" << expected;
    LOG(INFO) << "result=" << result;
  }

  ASSERT_EQ(result, expected);
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    QuantizedParameter&& quantized_parameter,
    std::initializer_list<DimensionSize>&& shape, DimensionSize iota_dimension,
    std::vector<typename Storage<expressed_type>::Type>&& expected_values) {
  auto expected_quant_values = QuantizeVector<storage_type, expressed_type>(
      expected_values, quantized_parameter);
  decltype(expected_quant_values) result_quant_values(
      expected_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));
  QuantizedTensor expected(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      expected_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  auto res = Iota(iota_dimension, result);
  if (!res.ok()) {
    LOG(INFO) << "Failure: " << res;
  }
  ASSERT_EQ(res.ok(), true);

  if (result != expected) {
    LOG(INFO) << "iota_dimension=" << iota_dimension;
    LOG(INFO) << "expected=" << expected;
    LOG(INFO) << "result=" << result;
  }

  ASSERT_EQ(result, expected);
}

TEST(Iota, Unquantized) {
  test<ElementType::kSI8>(
      {4, 5}, 0, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI8>(
      {4, 5}, 1, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});

  test<ElementType::kSI16>(
      {4, 5}, 0, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI16>(
      {4, 5}, 1, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});

  test<ElementType::kSI32>(
      {4, 5}, 0, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI32>(
      {4, 5}, 1, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});

  test<ElementType::kBF16>(
      {4, 5}, 0, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kBF16>(
      {4, 5}, 1, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});

  test<ElementType::kF16>(
      {4, 5}, 0, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kF16>(
      {4, 5}, 1, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});

  test<ElementType::kF32>(
      {4, 5}, 0, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kF32>(
      {4, 5}, 1, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
}

TEST(Iota, Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});

  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 1e-3, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 1e-3, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});

  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 1e-2, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 1e-3, .zero_point = 0}, {4, 5}, 0,
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 1e-3, .zero_point = 0}, {4, 5}, 1,
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4});
}

}  // namespace testing
}  // namespace stablehlo
