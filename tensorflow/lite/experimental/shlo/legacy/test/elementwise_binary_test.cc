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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/debug.h"  // IWYU pragma: keep, b/321245930
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/test/matchers.h"
#include "tensorflow/lite/experimental/shlo/legacy/test/util.h"

namespace stablehlo {
namespace testing {

template <ElementType element_type>
void test(absl::Status (*op)(const Tensor&, const Tensor&, Tensor&),
          std::initializer_list<DimensionSize>&& shape,
          std::vector<typename Storage<element_type>::Type>&& input1_values,
          std::vector<typename Storage<element_type>::Type>&& input2_values,
          std::vector<typename Storage<element_type>::Type>&& expected_values) {
  Tensor input1(TensorType(Shape(shape), element_type),
                std::data(input1_values));
  Tensor input2(TensorType(Shape(shape), element_type),
                std::data(input2_values));
  Tensor expected(TensorType(Shape(shape), element_type),
                  std::data(expected_values));

  std::vector<typename Storage<element_type>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), element_type), result_values.data());

  ASSERT_OK(op(input1, input2, result));
  EXPECT_THAT(result, IsAlmostSame(expected))
      << "input1: " << input1 << "\ninput2: " << input2;
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    absl::Status (*op)(const QuantizedTensor&, const QuantizedTensor&,
                       QuantizedTensor&),
    std::initializer_list<DimensionSize>&& shape,
    QuantizedParameter&& quantized_parameter,
    std::vector<typename Storage<expressed_type>::Type>&& input1_values,
    std::vector<typename Storage<expressed_type>::Type>&& input2_values,
    std::vector<typename Storage<expressed_type>::Type>&& expected_values) {
  auto input1_quant_values = QuantizeVector<storage_type, expressed_type>(
      input1_values, quantized_parameter);
  auto input2_quant_values = QuantizeVector<storage_type, expressed_type>(
      input2_values, quantized_parameter);
  auto expected_quant_values = QuantizeVector<storage_type, expressed_type>(
      expected_values, quantized_parameter);
  std::vector<typename Storage<storage_type>::Type> result_quant_values(
      expected_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));

  QuantizedTensor input1(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      input1_quant_values.data());
  QuantizedTensor input2(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      input2_quant_values.data());
  QuantizedTensor expected(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      expected_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  ASSERT_OK(op(input1, input2, result));
  EXPECT_THAT(result, IsAlmostSame(expected))
      << "input1: " << input1 << "\ninput2: " << input2;
}

TEST(ElementwiseBinary, Add) {
  test<ElementType::kI1>(Add, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 1, 1});
  test<ElementType::kSI8>(Add, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 2, 1});
  test<ElementType::kSI16>(Add, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 2, 1});
  test<ElementType::kSI32>(Add, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 2, 1});
  test<ElementType::kBF16>(Add, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 2, 1});
  test<ElementType::kF16>(Add, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 2, 1});
  test<ElementType::kF32>(Add, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 2, 1});
}

TEST(ElementwiseBinary, AddQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Add, {4}, {.scale = 1, .zero_point = 0}, {10, 0, 20, 0}, {0, 0, 10, -10},
      {10, 0, 30, -10});
  test<ElementType::kSI16, ElementType::kBF16>(
      Add, {4}, {.scale = 2, .zero_point = 2}, {10, 0, 20, 0}, {0, 0, 10, -10},
      {10, 0, 30, -10});
  test<ElementType::kSI32, ElementType::kBF16>(
      Add, {4}, {.scale = 0.5, .zero_point = -10}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 30, -10});
  test<ElementType::kSI8, ElementType::kF16>(
      Add, {4}, {.scale = 1, .zero_point = 0}, {10, 0, 20, 0}, {0, 0, 10, -10},
      {10, 0, 30, -10});
  test<ElementType::kSI16, ElementType::kF16>(
      Add, {4}, {.scale = 2, .zero_point = 2}, {10, 0, 20, 0}, {0, 0, 10, -10},
      {10, 0, 30, -10});
  test<ElementType::kSI32, ElementType::kF16>(
      Add, {4}, {.scale = 0.5, .zero_point = -10}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 30, -10});
  test<ElementType::kSI8, ElementType::kF32>(
      Add, {4}, {.scale = 1, .zero_point = 0}, {10, 0, 20, 0}, {0, 0, 10, -10},
      {10, 0, 30, -10});
  test<ElementType::kSI16, ElementType::kF32>(
      Add, {4}, {.scale = 2, .zero_point = 2}, {10, 0, 20, 0}, {0, 0, 10, -10},
      {10, 0, 30, -10});
  test<ElementType::kSI32, ElementType::kF32>(
      Add, {4}, {.scale = 0.5, .zero_point = -10}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 30, -10});
}

TEST(ElementwiseBinary, And) {
  test<ElementType::kI1>(And, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {0, 0, 1, 0});
  test<ElementType::kSI8>(And, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {1, 0, 4, 0});
  test<ElementType::kSI16>(And, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {1, 0, 4, 0});
  test<ElementType::kSI32>(And, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {1, 0, 4, 0});
}

TEST(ElementwiseBinary, Atan2) {
  test<ElementType::kBF16>(Atan2, {4}, {3, 0, 5, 3}, {1, 1, 4, 1},
                           {1.24904577239825442582f, 0, 0.89605538457134395617f,
                            1.24904577239825442582f});
  test<ElementType::kF16>(Atan2, {4}, {3, 0, 5, 3}, {1, 1, 4, 1},
                          {1.24904577239825442582f, 0, 0.89605538457134395617f,
                           1.24904577239825442582f});
  test<ElementType::kF32>(Atan2, {4}, {3, 0, 5, 3}, {1, 1, 4, 1},
                          {1.24904577239825442582f, 0, 0.89605538457134395617f,
                           1.24904577239825442582f});
}

TEST(ElementwiseBinary, Atan2Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Atan2, {4}, {.scale = 1e-1, .zero_point = 0}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI8, ElementType::kF16>(
      Atan2, {4}, {.scale = 1e-1, .zero_point = 2}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI8, ElementType::kF32>(
      Atan2, {4}, {.scale = 1e-1, .zero_point = -2}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI16, ElementType::kBF16>(
      Atan2, {4}, {.scale = 1e-2, .zero_point = 0}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI16, ElementType::kF16>(
      Atan2, {4}, {.scale = 1e-2, .zero_point = 2}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI16, ElementType::kF32>(
      Atan2, {4}, {.scale = 1e-3, .zero_point = -2}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI32, ElementType::kBF16>(
      Atan2, {4}, {.scale = 1e-2, .zero_point = 0}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI32, ElementType::kF16>(
      Atan2, {4}, {.scale = 1e-2, .zero_point = 2}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
  test<ElementType::kSI32, ElementType::kF32>(
      Atan2, {4}, {.scale = 1e-3, .zero_point = -2}, {3, 0, 5, 3}, {1, 1, 4, 1},
      {1.24904577239825442582f, 0, 0.89605538457134395617f,
       1.24904577239825442582f});
}

TEST(ElementwiseBinary, Divide) {
  test<ElementType::kSI8>(Divide, {4}, {2, 5, -3, -7}, {2, 2, 3, 3},
                          {1, 2, -1, -2});
  test<ElementType::kSI16>(Divide, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {11, 18, -8, 15});
  test<ElementType::kSI32>(Divide, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {11, 18, -8, 15});
  test<ElementType::kBF16>(Divide, {4}, {22, 53, -33, -77}, {2, 4, 4, -5},
                           {11, 13.25, -8.25, 15.4});
  test<ElementType::kF16>(Divide, {4}, {22, 53, -33, -77}, {2, 4, 4, -5},
                          {11, 13.25, -8.25, 15.4});
  test<ElementType::kF32>(Divide, {4}, {22, 53, -33, -77}, {2, 4, 4, -5},
                          {11, 13.25, -8.25, 15.4});
}

TEST(ElementwiseBinary, DivideQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Divide, {4}, {.scale = 1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {11, 13.25, -8.25, 15.4});
  test<ElementType::kSI8, ElementType::kF16>(
      Divide, {4}, {.scale = 1, .zero_point = 5}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {11, 13.25, -8.25, 15.4});
  test<ElementType::kSI8, ElementType::kF32>(
      Divide, {4}, {.scale = 1, .zero_point = -5}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {11, 13.25, -8.25, 15.4});
  test<ElementType::kSI16, ElementType::kBF16>(
      Divide, {4}, {.scale = 5e-1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {11, 13.25, -8.25, 15.4});
  test<ElementType::kSI16, ElementType::kF16>(
      Divide, {4}, {.scale = 1e-1, .zero_point = 10}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {11, 13.25, -8.25, 15.4});
  test<ElementType::kSI16, ElementType::kF32>(
      Divide, {4}, {.scale = 5e-2, .zero_point = -10}, {222, 533, -333, -777},
      {2, 4, 4, -5}, {111, 133.25, -83.25, 155.4});
  test<ElementType::kSI32, ElementType::kBF16>(
      Divide, {4}, {.scale = 5e-1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {11, 13.25, -8.25, 15.4});
  test<ElementType::kSI32, ElementType::kF16>(
      Divide, {4}, {.scale = 1e-1, .zero_point = 10}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {11, 13.25, -8.25, 15.4});
  test<ElementType::kSI32, ElementType::kF32>(
      Divide, {4}, {.scale = 5e-2, .zero_point = -10}, {222, 533, -333, -777},
      {2, 4, 4, -5}, {111, 133.25, -83.25, 155.4});
}

TEST(ElementwiseBinary, Maximum) {
  test<ElementType::kI1>(Maximum, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                         {1, 0, 1, 1});
  test<ElementType::kSI8>(Maximum, {4}, {2, 5, -3, -7}, {2, 2, 3, 3},
                          {2, 5, 3, 3});
  test<ElementType::kSI16>(Maximum, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {22, 55, 4, -5});
  test<ElementType::kSI32>(Maximum, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {22, 55, 4, -5});
  test<ElementType::kBF16>(Maximum, {4}, {2.2, 5.3, -3.3, -7.7},
                           {2.2, 4.4, 4.4, -5.5}, {2.2, 5.3, 4.4, -5.5});
  test<ElementType::kF16>(Maximum, {4}, {22, 55, -33, -77},
                          {2.5, 3.5, 4.5, -5.5}, {22, 55, 4.5, -5.5});
  test<ElementType::kF32>(Maximum, {4}, {2.2, 5.3, -3.3, -7.7},
                          {2.2, 4.4, 4.4, -5.5}, {2.2, 5.3, 4.4, -5.5});
}

TEST(ElementwiseBinary, MaximumQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Maximum, {4}, {.scale = 1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {22, 53, 4, -5});
  test<ElementType::kSI8, ElementType::kF16>(
      Maximum, {4}, {.scale = 1, .zero_point = 5}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {22, 53, 4, -5});
  test<ElementType::kSI8, ElementType::kF32>(
      Maximum, {4}, {.scale = 1, .zero_point = -5}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {22, 53, 4, -5});
  test<ElementType::kSI16, ElementType::kBF16>(
      Maximum, {4}, {.scale = 5e-1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {22, 53, 4, -5});
  test<ElementType::kSI16, ElementType::kF16>(
      Maximum, {4}, {.scale = 1e-1, .zero_point = 10}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {22, 53, 4, -5});
  test<ElementType::kSI16, ElementType::kF32>(
      Maximum, {4}, {.scale = 5e-2, .zero_point = -10}, {222, 533, -333, -777},
      {2, 4, 4, -5}, {222, 533, 4, -5});
  test<ElementType::kSI32, ElementType::kBF16>(
      Maximum, {4}, {.scale = 5e-1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {22, 53, 4, -5});
  test<ElementType::kSI32, ElementType::kF16>(
      Maximum, {4}, {.scale = 1e-1, .zero_point = 10}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {22, 53, 4, -5});
  test<ElementType::kSI32, ElementType::kF32>(
      Maximum, {4}, {.scale = 5e-2, .zero_point = -10}, {222, 533, -333, -777},
      {2, 4, 4, -5}, {222, 533, 4, -5});
}

TEST(ElementwiseBinary, Minimum) {
  test<ElementType::kI1>(Minimum, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                         {0, 0, 1, 0});
  test<ElementType::kSI8>(Minimum, {4}, {2, 5, -3, -7}, {2, 2, 3, 3},
                          {2, 2, -3, -7});
  test<ElementType::kSI16>(Minimum, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {2, 3, -33, -77});
  test<ElementType::kSI32>(Minimum, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {2, 3, -33, -77});
  test<ElementType::kBF16>(Minimum, {4}, {2.2, 5.3, -3.3, -7.7},
                           {2.2, 4.4, 4.4, -5.5}, {2.2, 4.4, -3.3, -7.7});
  test<ElementType::kF16>(Minimum, {4}, {22, 55, -33, -77},
                          {2.5, 3.5, 4.5, -5.5}, {2.5, 3.5, -33, -77});
  test<ElementType::kF32>(Minimum, {4}, {2.2, 5.3, -3.3, -7.7},
                          {2.2, 4.4, 4.4, -5.5}, {2.2, 4.4, -3.3, -7.7});
}

TEST(ElementwiseBinary, MinimumQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Minimum, {4}, {.scale = 1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {2, 4, -33, -77});
  test<ElementType::kSI8, ElementType::kF16>(
      Minimum, {4}, {.scale = 1, .zero_point = 5}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {2, 4, -33, -77});
  test<ElementType::kSI8, ElementType::kF32>(
      Minimum, {4}, {.scale = 1, .zero_point = -5}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {2, 4, -33, -77});
  test<ElementType::kSI16, ElementType::kBF16>(
      Minimum, {4}, {.scale = 5e-1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {2, 4, -33, -77});
  test<ElementType::kSI16, ElementType::kF16>(
      Minimum, {4}, {.scale = 1e-1, .zero_point = 10}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {2, 4, -33, -77});
  test<ElementType::kSI16, ElementType::kF32>(
      Minimum, {4}, {.scale = 5e-2, .zero_point = -10}, {222, 533, -333, -777},
      {2, 4, 4, -5}, {2, 4, -333, -777});
  test<ElementType::kSI32, ElementType::kBF16>(
      Minimum, {4}, {.scale = 5e-1, .zero_point = 0}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {2, 4, -33, -77});
  test<ElementType::kSI32, ElementType::kF16>(
      Minimum, {4}, {.scale = 1e-1, .zero_point = 10}, {22, 53, -33, -77},
      {2, 4, 4, -5}, {2, 4, -33, -77});
  test<ElementType::kSI32, ElementType::kF32>(
      Minimum, {4}, {.scale = 5e-2, .zero_point = -10}, {222, 533, -333, -777},
      {2, 4, 4, -5}, {2, 4, -333, -777});
}

TEST(ElementwiseBinary, Multiply) {
  test<ElementType::kI1>(Multiply, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                         {0, 0, 1, 0});
  test<ElementType::kSI8>(Multiply, {4}, {2, 5, -3, -7}, {2, 2, 3, 3},
                          {4, 10, -9, -21});
  test<ElementType::kSI16>(Multiply, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {44, 165, -132, 385});
  test<ElementType::kSI32>(Multiply, {4}, {22, 55, -33, -77}, {2, 3, 4, -5},
                           {44, 165, -132, 385});
  test<ElementType::kBF16>(Multiply, {4}, {2.2, 5.3, -3.3, -7.7},
                           {2.2, 4.4, 4.4, -5.5}, {4.84, 23.32, -14.52, 42.35});
  test<ElementType::kF16>(Multiply, {4}, {22, 55, -33, -77},
                          {2.5, 3.5, 4.5, -5.5}, {55, 192.5, -148.5, 423.5});
  test<ElementType::kF32>(Multiply, {4}, {2.2, 5.3, -3.3, -7.7},
                          {2.2, 4.4, 4.4, -5.5}, {4.84, 23.32, -14.52, 42.35});
}

TEST(ElementwiseBinary, MultiplyQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI8, ElementType::kF16>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI8, ElementType::kF32>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI16, ElementType::kBF16>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI16, ElementType::kF16>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI16, ElementType::kF32>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI32, ElementType::kBF16>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI32, ElementType::kF16>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
  test<ElementType::kSI32, ElementType::kF32>(
      Multiply, {4}, {.scale = 1e-1, .zero_point = 0}, {1.1, 2.2, -3.3, -4.4},
      {0.1, 1, 0.5, 2.5}, {0.11, 2.2, -1.7, -11});
}

TEST(ElementwiseBinary, Or) {
  test<ElementType::kI1>(Or, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 1, 1});
  test<ElementType::kSI8>(Or, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {3, 0, 5, 1});
  test<ElementType::kSI16>(Or, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {3, 0, 5, 1});
  test<ElementType::kSI32>(Or, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {3, 0, 5, 1});
}

TEST(ElementwiseBinary, Power) {
  test<ElementType::kSI8>(Power, {6}, {-2, 1, -3, 5, -3, 4}, {0, 1, 2, 3, 3, 2},
                          {1, 1, 9, 125, -27, 16});
  test<ElementType::kSI16>(Power, {6}, {-2, 1, -36, 5, 3, 5},
                           {0, 1, 2, 3, 4, 5}, {1, 1, 1296, 125, 81, 3125});
  test<ElementType::kSI32>(Power, {6}, {-2, 1, -36, 5, 3, 10},
                           {0, 1, 2, 3, 4, 5}, {1, 1, 1296, 125, 81, 100000});
  test<ElementType::kBF16>(Power, {6}, {-2, -0, -36, 5, 3, 1000},
                           {2, 2, 1.1, 2, -1, 10},
                           {4, 0, -NAN, 25, 0.3333333333333333f, 1e+30});
  test<ElementType::kF16>(Power, {6}, {-2, -0, -36, 5, 3, 10000},
                          {2, 2, 1.1, 2, -1, 10},
                          {4, 0, -NAN, 25, 0.3333333333333333f, INFINITY});
  test<ElementType::kF32>(Power, {6}, {-2, -0, -36, 5, 3, 10000},
                          {2, 2, 1.1, 2, -1, 10},
                          {4, 0, -NAN, 25, 0.3333333333333333f, INFINITY});
}

TEST(ElementwiseBinary, Remainder) {
  test<ElementType::kSI8>(Remainder, {4}, {17, 18, 19, 20}, {3, 4, 5, 7},
                          {2, 2, 4, 6});
  test<ElementType::kSI16>(Remainder, {4}, {17, 18, 19, 20}, {3, 4, 5, 7},
                           {2, 2, 4, 6});
  test<ElementType::kSI32>(Remainder, {4}, {17, -17, 17, -17}, {3, 3, -3, -3},
                           {2, -2, 2, -2});
  test<ElementType::kBF16>(Remainder, {4}, {17, 18, 19, 20}, {3, 4, 5, 7},
                           {2, 2, 4, 6});
  test<ElementType::kF16>(Remainder, {4}, {17, -17, 17, -17}, {3, 3, -3, -3},
                          {2, -2, 2, -2});
  test<ElementType::kF32>(Remainder, {4}, {17.1, -17.1, 17.1, -17.1},
                          {3, 3, -3, -3}, {2.1, -2.1, 2.1, -2.1});
}

TEST(ElementwiseBinary, RemainderQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Remainder, {4}, {.scale = 1e-1, .zero_point = 0}, {7.1, -7.1, 7.1, -7.1},
      {3, 3, -3, -3}, {1.1, -1.1, 1.1, -1.1});
  test<ElementType::kSI8, ElementType::kF16>(
      Remainder, {4}, {.scale = 1e-1, .zero_point = 0}, {7.1, -7.1, 7.1, -7.1},
      {3, 3, -3, -3}, {1.1, -1.1, 1.1, -1.1});
  test<ElementType::kSI8, ElementType::kF32>(
      Remainder, {4}, {.scale = 1e-1, .zero_point = 0}, {7.1, -7.1, 7.1, -7.1},
      {3, 3, -3, -3}, {1.1, -1.1, 1.1, -1.1});

  test<ElementType::kSI16, ElementType::kBF16>(
      Remainder, {4}, {.scale = 1e-1, .zero_point = 4}, {17, 18, 19, 20},
      {3, 4, 5, 7}, {2, 2, 4, 6});
  test<ElementType::kSI16, ElementType::kF16>(
      Remainder, {4}, {.scale = 1e-1, .zero_point = 0}, {17, -17, 17, -17},
      {3, 3, -3, -3}, {2, -2, 2, -2});
  test<ElementType::kSI16, ElementType::kF32>(
      Remainder, {4}, {.scale = 1e-2, .zero_point = -10},
      {17.1, -17.1, 17.1, -17.1}, {3, 3, -3, -3}, {2.1, -2.1, 2.1, -2.1});
  test<ElementType::kSI32, ElementType::kBF16>(
      Remainder, {4}, {.scale = 1e-1, .zero_point = 4}, {17, 18, 19, 20},
      {3, 4, 5, 7}, {2, 2, 4, 6});
  test<ElementType::kSI32, ElementType::kF16>(
      Remainder, {4}, {.scale = 1e-1, .zero_point = 0}, {17, -17, 17, -17},
      {3, 3, -3, -3}, {2, -2, 2, -2});
  test<ElementType::kSI32, ElementType::kF32>(
      Remainder, {4}, {.scale = 1e-2, .zero_point = -10},
      {17.1, -17.1, 17.1, -17.1}, {3, 3, -3, -3}, {2.1, -2.1, 2.1, -2.1});
}

TEST(ElementwiseBinary, ShiftLeft) {
  test<ElementType::kSI8>(ShiftLeft, {3}, {-1, 0, 1}, {1, 2, 3}, {-2, 0, 8});
  test<ElementType::kSI16>(ShiftLeft, {3}, {-1, 0, 1}, {1, 2, 3}, {-2, 0, 8});
  test<ElementType::kSI32>(ShiftLeft, {3}, {-1, 0, 1}, {1, 2, 3}, {-2, 0, 8});
}

TEST(ElementwiseBinary, ShiftRightArithmetic) {
  test<ElementType::kSI8>(ShiftRightArithmetic, {3}, {-1, 0, 8}, {1, 2, 3},
                          {-1, 0, 1});
  test<ElementType::kSI16>(ShiftRightArithmetic, {3}, {-1, 0, 8}, {1, 2, 3},
                           {-1, 0, 1});
  test<ElementType::kSI32>(ShiftRightArithmetic, {3}, {-1, 0, 8}, {1, 2, 3},
                           {-1, 0, 1});
}

TEST(ElementwiseBinary, ShiftRightLogical) {
  test<ElementType::kSI8>(ShiftRightLogical, {3}, {-1, 0, 8}, {1, 2, 3},
                          {0x7F, 0, 1});
  test<ElementType::kSI16>(ShiftRightLogical, {3}, {-1, 0, 8}, {1, 2, 3},
                           {0x7FFF, 0, 1});
  test<ElementType::kSI32>(ShiftRightLogical, {3}, {-1, 0, 8}, {1, 2, 3},
                           {0x7FFFFFFF, 0, 1});
}

TEST(ElementwiseBinary, Subtract) {
  test<ElementType::kSI8>(Subtract, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                          {1, 0, 0, -1});
  test<ElementType::kSI16>(Subtract, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                           {1, 0, 0, -1});
  test<ElementType::kSI32>(Subtract, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                           {1, 0, 0, -1});
  test<ElementType::kBF16>(Subtract, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                           {1, 0, 0, -1});
  test<ElementType::kF16>(Subtract, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                          {1, 0, 0, -1});
  test<ElementType::kF32>(Subtract, {4}, {1, 0, 1, 0}, {0, 0, 1, 1},
                          {1, 0, 0, -1});
}

TEST(ElementwiseBinary, SubtractQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Subtract, {4}, {.scale = 1, .zero_point = 0}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI8, ElementType::kF16>(
      Subtract, {4}, {.scale = 1, .zero_point = 2}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI8, ElementType::kF32>(
      Subtract, {4}, {.scale = 1, .zero_point = -10}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI16, ElementType::kBF16>(
      Subtract, {4}, {.scale = 1e-1, .zero_point = 0}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI16, ElementType::kF16>(
      Subtract, {4}, {.scale = 1e-1, .zero_point = 2}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI16, ElementType::kF32>(
      Subtract, {4}, {.scale = 1e-1, .zero_point = -10}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI32, ElementType::kBF16>(
      Subtract, {4}, {.scale = 1e-3, .zero_point = 0}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI32, ElementType::kF16>(
      Subtract, {4}, {.scale = 1e-3, .zero_point = 2}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
  test<ElementType::kSI32, ElementType::kF32>(
      Subtract, {4}, {.scale = 1e-3, .zero_point = -10}, {10, 0, 20, 0},
      {0, 0, 10, -10}, {10, 0, 10, 10});
}

TEST(ElementwiseBinary, Xor) {
  test<ElementType::kI1>(Xor, {4}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, 1});
  test<ElementType::kSI8>(Xor, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {2, 0, 1, 1});
  test<ElementType::kSI16>(Xor, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {2, 0, 1, 1});
  test<ElementType::kSI32>(Xor, {4}, {3, 0, 5, 0}, {1, 0, 4, 1}, {2, 0, 1, 1});
}

}  // namespace testing
}  // namespace stablehlo
