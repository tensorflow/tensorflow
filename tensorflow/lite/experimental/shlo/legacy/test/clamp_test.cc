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
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/debug.h"  // IWYU pragma: keep, b/321245930
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/test/util.h"

namespace stablehlo {
namespace testing {

template <ElementType element_type>
void test(std::initializer_list<DimensionSize>&& shape,
          std::vector<typename Storage<element_type>::Type>&& min_values,
          std::vector<typename Storage<element_type>::Type>&& operand_values,
          std::vector<typename Storage<element_type>::Type>&& max_values,
          std::vector<typename Storage<element_type>::Type>&& expected_values) {
  Shape min_shape = (min_values.size() > 1) ? Shape(shape) : Shape();
  Tensor min(TensorType(std::move(min_shape), element_type), min_values.data());
  Shape max_shape = (max_values.size() > 1) ? Shape(shape) : Shape();
  Tensor max(TensorType(std::move(max_shape), element_type), max_values.data());
  Tensor operand(TensorType(Shape(shape), element_type), operand_values.data());
  Tensor expected(TensorType(Shape(shape), element_type),
                  expected_values.data());

  std::vector<typename Storage<element_type>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), element_type), result_values.data());

  ASSERT_OK(Clamp(min, operand, max, result));
  EXPECT_EQ(result, expected)
      << "min: " << min << "\nmax: " << max << "\noperand: " << operand;
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    QuantizedParameter&& quantized_parameter,
    std::initializer_list<DimensionSize>&& shape,
    std::vector<typename Storage<expressed_type>::Type>&& min_values,
    std::vector<typename Storage<expressed_type>::Type>&& operand_values,
    std::vector<typename Storage<expressed_type>::Type>&& max_values,
    std::vector<typename Storage<expressed_type>::Type>&& expected_values) {
  auto min_quant_values = QuantizeVector<storage_type, expressed_type>(
      min_values, quantized_parameter);
  auto operand_quant_values = QuantizeVector<storage_type, expressed_type>(
      operand_values, quantized_parameter);
  auto max_quant_values = QuantizeVector<storage_type, expressed_type>(
      max_values, quantized_parameter);
  auto expected_quant_values = QuantizeVector<storage_type, expressed_type>(
      expected_values, quantized_parameter);
  std::vector<typename Storage<storage_type>::Type> result_quant_values(
      expected_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));

  Shape min_shape = (min_values.size() > 1) ? Shape(shape) : Shape();
  QuantizedTensor min(
      QuantizedTensorType(std::move(min_shape),
                          QuantizedTensorElementType(element_type)),
      min_quant_values.data());
  Shape max_shape = (max_values.size() > 1) ? Shape(shape) : Shape();
  QuantizedTensor max(
      QuantizedTensorType(std::move(max_shape),
                          QuantizedTensorElementType(element_type)),
      max_quant_values.data());
  QuantizedTensor operand(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      operand_quant_values.data());
  QuantizedTensor expected(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      expected_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  ASSERT_OK(Clamp(min, operand, max, result));
  EXPECT_EQ(result, expected)
      << "min: " << min << "\nmax: " << max << "\noperand: " << operand;
}

TEST(Clamp, Unquantized) {
  test<ElementType::kSI8>({3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI16>({3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI32>({3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kBF16>({3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kF16>({3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kF32>({3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});

  test<ElementType::kSI8>({3}, {0, 1, 1}, {-3, 0, 3}, {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI16>({3}, {0, 1, 1}, {-3, 0, 3}, {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI32>({3}, {0, 1, 1}, {-3, 0, 3}, {1, 1, 2}, {0, 1, 2});
  test<ElementType::kBF16>({3}, {0, 1, 1}, {-3, 0, 3}, {1, 1, 2}, {0, 1, 2});
  test<ElementType::kF16>({3}, {0, 1, 1}, {-3, 0, 3}, {1, 1, 2}, {0, 1, 2});
  test<ElementType::kF32>({3}, {0, 1, 1}, {-3, 0, 3}, {1, 1, 2}, {0, 1, 2});
}

TEST(Clamp, Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});

  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});

  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {3}, {0}, {-2, 0, 2}, {1}, {0, 0, 1});

  test<ElementType::kSI8, ElementType::kBF16>({.scale = 0.1, .zero_point = 0},
                                              {3}, {0, 1, 1}, {-3, 0, 3},
                                              {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI8, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                             {3}, {0, 1, 1}, {-3, 0, 3},
                                             {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI8, ElementType::kF32>({.scale = 0.1, .zero_point = 0},
                                             {3}, {0, 1, 1}, {-3, 0, 3},
                                             {1, 1, 2}, {0, 1, 2});

  test<ElementType::kSI16, ElementType::kBF16>({.scale = 0.1, .zero_point = 0},
                                               {3}, {0, 1, 1}, {-3, 0, 3},
                                               {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI16, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                              {3}, {0, 1, 1}, {-3, 0, 3},
                                              {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI16, ElementType::kF32>({.scale = 0.1, .zero_point = 0},
                                              {3}, {0, 1, 1}, {-3, 0, 3},
                                              {1, 1, 2}, {0, 1, 2});

  test<ElementType::kSI32, ElementType::kBF16>({.scale = 0.1, .zero_point = 0},
                                               {3}, {0, 1, 1}, {-3, 0, 3},
                                               {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI32, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                              {3}, {0, 1, 1}, {-3, 0, 3},
                                              {1, 1, 2}, {0, 1, 2});
  test<ElementType::kSI32, ElementType::kF32>({.scale = 0.1, .zero_point = 0},
                                              {3}, {0, 1, 1}, {-3, 0, 3},
                                              {1, 1, 2}, {0, 1, 2});
}

}  // namespace testing
}  // namespace stablehlo
