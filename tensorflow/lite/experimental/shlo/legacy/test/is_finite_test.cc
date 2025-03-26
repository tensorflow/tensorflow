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

#include <cmath>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/debug.h"  // IWYU pragma: keep, b/321245930
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"

namespace stablehlo {
namespace testing {

template <ElementType element_type>
void test(
    std::initializer_list<DimensionSize>&& shape,
    std::vector<typename Storage<element_type>::Type>&& input_values,
    std::vector<typename Storage<ElementType::kI1>::Type>&& expected_values) {
  Tensor input(TensorType(Shape(shape), element_type), input_values.data());
  Tensor expected(TensorType(Shape(shape), ElementType::kI1),
                  expected_values.data());

  std::vector<typename Storage<ElementType::kI1>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), ElementType::kI1),
                result_values.data());

  ASSERT_OK(IsFinite(input, result));
  EXPECT_EQ(result, expected) << "input: " << input;
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    QuantizedParameter&& quantized_parameter,
    std::initializer_list<DimensionSize>&& shape,
    std::vector<typename Storage<expressed_type>::Type>&& input_values,
    std::vector<typename Storage<ElementType::kI1>::Type>&& expected_values) {
  Tensor input(TensorType(Shape(shape), expressed_type), input_values.data());
  Tensor expected(TensorType(Shape(shape), ElementType::kI1),
                  expected_values.data());

  std::vector<typename Storage<ElementType::kI1>::Type> result_values(
      input_values.size());
  Tensor result(TensorType(Shape(shape), ElementType::kI1),
                result_values.data());

  ASSERT_OK(IsFinite(input, result));
  EXPECT_EQ(result, expected) << "input: " << input;
}

TEST(IsFinite, Unquantized) {
  test<ElementType::kBF16>({7}, {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
                           {false, false, false, false, true, true, true});
  test<ElementType::kF16>({7}, {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
                          {false, false, false, false, true, true, true});
  test<ElementType::kF32>({7}, {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
                          {false, false, false, false, true, true, true});
}

TEST(IsFinite, Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});

  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});

  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, {7},
      {+NAN, -NAN, -INFINITY, +INFINITY, -1, 0, 1},
      {false, false, false, false, true, true, true});
}

}  // namespace testing
}  // namespace stablehlo
