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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/debug.h"  // IWYU pragma: keep, b/321245930
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/test/util.h"

namespace stablehlo {
namespace testing {

template <ElementType element_type>
void test(std::initializer_list<DimensionSize>&& operand_shape,
          std::vector<typename Storage<element_type>::Type>&& operand_values,
          std::initializer_list<DimensionSize>&& broadcast_dimensions_values,
          std::initializer_list<DimensionSize>&& result_shape,
          std::vector<typename Storage<element_type>::Type>&& expected_values) {
  Tensor operand(TensorType(Shape(operand_shape), element_type),
                 operand_values.data());
  Tensor expected(TensorType(Shape(result_shape), element_type),
                  expected_values.data());

  std::vector<typename Storage<element_type>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(result_shape), element_type),
                result_values.data());

  absl::Span<const DimensionSize> broadcast_dimensions(
      broadcast_dimensions_values);

  ASSERT_OK(BroadcastInDim(operand, broadcast_dimensions, result));
  EXPECT_EQ(result, expected)
      << "operand: " << operand
      << "\nbroadcast_dimensions: " << ToString(broadcast_dimensions);
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    QuantizedParameter&& quantized_parameter,
    std::initializer_list<DimensionSize>&& operand_shape,
    std::vector<typename Storage<expressed_type>::Type>&& operand_values,
    std::initializer_list<DimensionSize>&& broadcast_dimensions_values,
    std::initializer_list<DimensionSize>&& result_shape,
    std::vector<typename Storage<expressed_type>::Type>&& expected_values) {
  auto operand_quant_values = QuantizeVector<storage_type, expressed_type>(
      operand_values, quantized_parameter);
  auto expected_quant_values = QuantizeVector<storage_type, expressed_type>(
      expected_values, quantized_parameter);
  std::vector<typename Storage<storage_type>::Type> result_quant_values(
      expected_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));
  QuantizedTensor operand(
      QuantizedTensorType(Shape(operand_shape),
                          QuantizedTensorElementType(element_type)),
      operand_quant_values.data());
  QuantizedTensor expected(
      QuantizedTensorType(Shape(result_shape),
                          QuantizedTensorElementType(element_type)),
      expected_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(result_shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  absl::Span<const DimensionSize> broadcast_dimensions(
      broadcast_dimensions_values);
  auto res = BroadcastInDim(operand, broadcast_dimensions, result);

  ASSERT_OK(BroadcastInDim(operand, broadcast_dimensions, result));
  EXPECT_EQ(result, expected)
      << "operand: " << operand
      << "\nbroadcast_dimensions: " << ToString(broadcast_dimensions);
}

TEST(BroadcastInDim, Unquantized) {
  test<ElementType::kI1>({1, 3}, {true, false, true}, {2, 1}, {2, 3, 2},
                         {true, true, false, false, true, true, true, true,
                          false, false, true, true});
  test<ElementType::kSI8>({1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
                          {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI16>({1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
                           {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI32>({1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
                           {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kBF16>({1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
                           {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kF16>({1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
                          {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kF32>({1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
                          {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
}

TEST(BroadcastInDim, Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});

  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});

  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {1, 3}, {1, 2, 3}, {2, 1}, {2, 3, 2},
      {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3});
}

}  // namespace testing
}  // namespace stablehlo
