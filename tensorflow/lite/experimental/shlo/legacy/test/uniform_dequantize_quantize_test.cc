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
#include "tensorflow/lite/experimental/shlo/legacy/test/matchers.h"

namespace stablehlo {
namespace testing {

template <ElementType storage_type, ElementType expressed_type>
void test(std::initializer_list<DimensionSize>&& shape,
          QuantizedParameter&& quantized_parameter,
          std::vector<typename Storage<expressed_type>::Type>&& input_values) {
  Tensor input(TensorType(Shape(shape), expressed_type), input_values.data());

  std::vector<typename Storage<storage_type>::Type> quant_values(
      input_values.size());
  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));
  QuantizedTensor quant(
      QuantizedTensorType(Shape(shape), std::move(element_type)),
      quant_values.data());

  std::vector<typename Storage<expressed_type>::Type> result_values(
      input_values.size());
  Tensor result(TensorType(Shape(shape), expressed_type), result_values.data());

  ASSERT_OK(UniformQuantize(input, quant));
  ASSERT_OK(UniformDequantize(quant, result));
  EXPECT_THAT(result, IsAlmostSame(input));
}

TEST(QuantizeDequantize, All) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {4}, {.scale = 1, .zero_point = 0}, {-2, -1, 0, 1, 2});
  test<ElementType::kSI8, ElementType::kBF16>(
      {4}, {.scale = 1, .zero_point = 0}, {-2, -1, 0, 1, 2});
  test<ElementType::kSI8, ElementType::kBF16>(
      {4}, {.scale = 1e-1, .zero_point = -5}, {-2.2, -1.1, 0, 1.1, 2.2});
  test<ElementType::kSI8, ElementType::kF16>({4}, {.scale = 1, .zero_point = 5},
                                             {-2, -1, 0, 1, 2});
  test<ElementType::kSI8, ElementType::kF16>(
      {4}, {.scale = 1e-1, .zero_point = -10}, {-2.2, -1.1, 0, 1.1, 2.2});
  test<ElementType::kSI8, ElementType::kF32>({4}, {.scale = 1, .zero_point = 5},
                                             {-2, -1, 0, 1, 2});
  test<ElementType::kSI8, ElementType::kF32>(
      {4}, {.scale = 1e-1, .zero_point = +10}, {-2.2, -1.1, 0, 1.1, 2.2});

  test<ElementType::kSI16, ElementType::kBF16>(
      {4}, {.scale = 1, .zero_point = 0}, {-2, -1, 0, 1, 2});
  test<ElementType::kSI16, ElementType::kBF16>(
      {4}, {.scale = 1e-1, .zero_point = 5}, {-2.2, -1.1, 0, 1.1, 2.2});
  test<ElementType::kSI16, ElementType::kBF16>(
      {4}, {.scale = 1e-2, .zero_point = -5}, {-2.22, -1.11, 0, 1.11, 2.22});
  test<ElementType::kSI16, ElementType::kF16>(
      {4}, {.scale = 1, .zero_point = 0}, {-2, -1, 0, 1, 2});
  test<ElementType::kSI16, ElementType::kF16>(
      {4}, {.scale = 1e-1, .zero_point = -10}, {-2.2, -1.1, 0, 1.1, 2.2});
  test<ElementType::kSI16, ElementType::kF16>(
      {4}, {.scale = 1e-2, .zero_point = 10}, {-2.22, -1.11, 0, 1.11, 2.22});

  test<ElementType::kSI32, ElementType::kBF16>(
      {4}, {.scale = 1, .zero_point = +7}, {-2, -1, 0, 1, 2});
  test<ElementType::kSI32, ElementType::kBF16>(
      {4}, {.scale = 1e-1, .zero_point = -7}, {-2.2, -1.1, 0, 1.1, 2.2});
  test<ElementType::kSI32, ElementType::kBF16>(
      {4}, {.scale = 1e-2, .zero_point = 0}, {-2.22, -1.11, 0, 1.11, 2.22});
  test<ElementType::kSI32, ElementType::kBF16>(
      {4}, {.scale = 1e-3, .zero_point = 0}, {-2.222, -1.111, 0, 1.111, 2.222});
  test<ElementType::kSI32, ElementType::kF16>(
      {4}, {.scale = 1, .zero_point = +7}, {-2, -1, 0, 1, 2});
  test<ElementType::kSI32, ElementType::kF16>(
      {4}, {.scale = 1e-1, .zero_point = -7}, {-2.2, -1.1, 0, 1.1, 2.2});
  test<ElementType::kSI32, ElementType::kF16>(
      {4}, {.scale = 1e-2, .zero_point = 10}, {-2.22, -1.11, 0, 1.11, 2.22});
  test<ElementType::kSI32, ElementType::kF16>(
      {4}, {.scale = 1e-3, .zero_point = -0},
      {-2.222, -1.111, 0, 1.111, 2.222});
  test<ElementType::kSI32, ElementType::kF32>(
      {4}, {.scale = 1, .zero_point = +7}, {-2, -1, 0, 1, 2});
  test<ElementType::kSI32, ElementType::kF32>(
      {4}, {.scale = 1e-1, .zero_point = -7}, {-2.2, -1.1, 0, 1.1, 2.2});
  test<ElementType::kSI32, ElementType::kF32>(
      {4}, {.scale = 1e-2, .zero_point = 10}, {-2.22, -1.11, 0, 1.11, 2.22});
  test<ElementType::kSI32, ElementType::kF32>(
      {4}, {.scale = 1e-3, .zero_point = -0},
      {-2.222, -1.111, 0, 1.111, 2.222});
}

}  // namespace testing
}  // namespace stablehlo
