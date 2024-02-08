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
          std::vector<typename Storage<ElementType::kI1>::Type>&& pred_values,
          std::vector<typename Storage<element_type>::Type>&& on_true_values,
          std::vector<typename Storage<element_type>::Type>&& on_false_values,
          std::vector<typename Storage<element_type>::Type>&& expected_values) {
  Shape pred_shape = (pred_values.size() > 1) ? Shape(shape) : Shape();
  Tensor pred(TensorType(std::move(pred_shape), ElementType::kI1),
              pred_values.data());
  Tensor on_true(TensorType(Shape(shape), element_type), on_true_values.data());
  Tensor on_false(TensorType(Shape(shape), element_type),
                  on_false_values.data());
  Tensor expected(TensorType(Shape(shape), element_type),
                  expected_values.data());

  std::vector<typename Storage<element_type>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), element_type), result_values.data());

  ASSERT_OK(Select(pred, on_true, on_false, result));
  EXPECT_EQ(result, expected) << "pred: " << pred << "\non_true: " << on_true
                              << "\nnon_false: " << on_false;
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    QuantizedParameter&& quantized_parameter,
    std::initializer_list<DimensionSize>&& shape,
    std::vector<typename Storage<ElementType::kI1>::Type>&& pred_values,
    std::vector<typename Storage<expressed_type>::Type>&& on_true_values,
    std::vector<typename Storage<expressed_type>::Type>&& on_false_values,
    std::vector<typename Storage<expressed_type>::Type>&& expected_values) {
  Shape pred_shape = (pred_values.size() > 1) ? Shape(shape) : Shape();
  Tensor pred(TensorType(std::move(pred_shape), ElementType::kI1),
              pred_values.data());

  auto on_true_quant_values = QuantizeVector<storage_type, expressed_type>(
      on_true_values, quantized_parameter);
  auto on_false_quant_values = QuantizeVector<storage_type, expressed_type>(
      on_false_values, quantized_parameter);
  auto expected_quant_values = QuantizeVector<storage_type, expressed_type>(
      expected_values, quantized_parameter);
  std::vector<typename Storage<storage_type>::Type> result_quant_values(
      expected_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));
  QuantizedTensor on_true(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      on_true_quant_values.data());
  QuantizedTensor on_false(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      on_false_quant_values.data());
  QuantizedTensor expected(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      expected_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  ASSERT_OK(Select(pred, on_true, on_false, result));
  EXPECT_EQ(result, expected) << "pred: " << pred << "\non_true: " << on_true
                              << "\nnon_false: " << on_false;
}

TEST(Select, Unquantized) {
  test<ElementType::kI1>({2}, {true}, {true, false}, {false, true},
                         {true, false});
  test<ElementType::kSI8>({2}, {false}, {1, 2}, {-1, -2}, {-1, -2});
  test<ElementType::kSI16>({2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kSI32>({2}, {false}, {1, 2}, {-1, -2}, {-1, -2});
  test<ElementType::kBF16>({2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kF16>({2}, {false}, {1, 2}, {-1, -2}, {-1, -2});
  test<ElementType::kF32>({2}, {true}, {1, 2}, {-1, -2}, {1, 2});

  test<ElementType::kI1>({2}, {true, false}, {true, true}, {false, false},
                         {true, false});
  test<ElementType::kSI8>({2}, {true, false}, {1, 2}, {-1, -2}, {1, -2});
  test<ElementType::kSI16>({2}, {true, false}, {1, 2}, {-1, -2}, {1, -2});
  test<ElementType::kSI32>({2}, {true, false}, {1, 2}, {-1, -2}, {1, -2});
  test<ElementType::kBF16>({2}, {true, false}, {1, 2}, {-1, -2}, {1, -2});
  test<ElementType::kF16>({2}, {true, false}, {1, 2}, {-1, -2}, {1, -2});
  test<ElementType::kF32>({2}, {true, false}, {1, 2}, {-1, -2}, {1, -2});
}

TEST(Select, Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kSI8, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                             {2}, {false}, {1, 2}, {-1, -2},
                                             {-1, -2});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kSI8, ElementType::kBF16>({.scale = 0.1, .zero_point = 0},
                                              {2}, {true, false}, {1, 2},
                                              {-1, -2}, {1, -2});
  test<ElementType::kSI8, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                             {2}, {true, false}, {1, 2},
                                             {-1, -2}, {1, -2});
  test<ElementType::kSI8, ElementType::kF32>({.scale = 0.1, .zero_point = 0},
                                             {2}, {true, false}, {1, 2},
                                             {-1, -2}, {1, -2});

  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kSI16, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                              {2}, {false}, {1, 2}, {-1, -2},
                                              {-1, -2});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kSI16, ElementType::kBF16>({.scale = 0.1, .zero_point = 0},
                                               {2}, {true, false}, {1, 2},
                                               {-1, -2}, {1, -2});
  test<ElementType::kSI16, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                              {2}, {true, false}, {1, 2},
                                              {-1, -2}, {1, -2});
  test<ElementType::kSI16, ElementType::kF32>({.scale = 0.1, .zero_point = 0},
                                              {2}, {true, false}, {1, 2},
                                              {-1, -2}, {1, -2});

  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kSI32, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                              {2}, {false}, {1, 2}, {-1, -2},
                                              {-1, -2});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {2}, {true}, {1, 2}, {-1, -2}, {1, 2});
  test<ElementType::kSI32, ElementType::kBF16>({.scale = 0.1, .zero_point = 0},
                                               {2}, {true, false}, {1, 2},
                                               {-1, -2}, {1, -2});
  test<ElementType::kSI32, ElementType::kF16>({.scale = 0.1, .zero_point = 0},
                                              {2}, {true, false}, {1, 2},
                                              {-1, -2}, {1, -2});
  test<ElementType::kSI32, ElementType::kF32>({.scale = 0.1, .zero_point = 0},
                                              {2}, {true, false}, {1, 2},
                                              {-1, -2}, {1, -2});
}

}  // namespace testing
}  // namespace stablehlo
