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
#include <cstdint>
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
void test(absl::Status (*op)(const Tensor&, Tensor&),
          std::initializer_list<DimensionSize>&& shape,
          std::vector<typename Storage<element_type>::Type>&& input_values,
          std::vector<typename Storage<element_type>::Type>&& expected_values) {
  Tensor input(TensorType(Shape(shape), element_type), std::data(input_values));
  Tensor expected(TensorType(Shape(shape), element_type),
                  std::data(expected_values));

  std::vector<typename Storage<element_type>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), element_type), result_values.data());

  ASSERT_OK(op(input, result));
  EXPECT_THAT(result, IsAlmostSame(expected)) << "input: " << input;
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    absl::Status (*op)(const QuantizedTensor&, QuantizedTensor&),
    std::initializer_list<DimensionSize>&& shape,
    QuantizedParameter&& quantized_parameter,
    std::vector<typename Storage<expressed_type>::Type>&& input_values,
    std::vector<typename Storage<expressed_type>::Type>&& expected_values) {
  auto input_quant_values = QuantizeVector<storage_type, expressed_type>(
      input_values, quantized_parameter);
  auto expected_quant_values = QuantizeVector<storage_type, expressed_type>(
      expected_values, quantized_parameter);
  std::vector<typename Storage<storage_type>::Type> result_quant_values(
      expected_quant_values.size());

  QuantizedTensorElementType element_type(storage_type, expressed_type,
                                          std::move(quantized_parameter));

  QuantizedTensor input(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      input_quant_values.data());
  QuantizedTensor expected(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      expected_quant_values.data());
  QuantizedTensor result(
      QuantizedTensorType(Shape(shape),
                          QuantizedTensorElementType(element_type)),
      result_quant_values.data());

  ASSERT_OK(op(input, result));
  EXPECT_THAT(result, IsAlmostSame(expected)) << "input: " << input;
}

TEST(ElementwiseUnary, Abs) {
  test<ElementType::kSI8>(Abs, {5}, {0, 1, -2, 3, -4}, {0, 1, 2, 3, 4});
  test<ElementType::kSI16>(Abs, {5}, {0, 1, -2, 3, -4}, {0, 1, 2, 3, 4});
  test<ElementType::kSI32>(Abs, {5}, {0, 1, -2, 3, -4}, {0, 1, 2, 3, 4});
  test<ElementType::kBF16>(Abs, {5}, {0, 1, -2, 3, -4}, {0, 1, 2, 3, 4});
  test<ElementType::kF16>(Abs, {5}, {0, 1, -2, 3, -4}, {0, 1, 2, 3, 4});
  test<ElementType::kF32>(Abs, {5}, {0, 1, -2, 3, -4}, {0, 1, 2, 3, 4});
}

TEST(ElementwiseBinary, AbsQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Abs, {5}, {.scale = 1, .zero_point = 0}, {0, 1, -2, 3, -4},
      {0, 1, 2, 3, 4});
  test<ElementType::kSI8, ElementType::kF16>(
      Abs, {5}, {.scale = 1e-1, .zero_point = 1}, {0, 1, -2, 3, -4},
      {0, 1, 2, 3, 4});
  test<ElementType::kSI8, ElementType::kF32>(
      Abs, {5}, {.scale = 1e-1, .zero_point = -1}, {0, 1, -2, 3, -4},
      {0, 1, 2, 3, 4});
  test<ElementType::kSI16, ElementType::kF32>(
      Abs, {5}, {.scale = 1e-3, .zero_point = -1}, {0, 1, -2, 3, -4},
      {0, 1, 2, 3, 4});
}

TEST(ElementwiseUnary, Cbrt) {
  test<ElementType::kBF16>(
      Cbrt, {4}, {0, 1, -2, 3},
      {0, 1, -1.25992104989487316476f, 1.44224957030740838232f});
  test<ElementType::kF16>(
      Cbrt, {4}, {0, 1, -2, 3},
      {0, 1, -1.25992104989487316476f, 1.44224957030740838232f});
  test<ElementType::kF32>(
      Cbrt, {4}, {0, 1, -2, 3},
      {0, 1, -1.25992104989487316476f, 1.44224957030740838232f});
}

TEST(ElementwiseUnary, CbrtQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Cbrt, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1, -2, 3},
      {0, 1, -1.25992104989487316476f, 1.44224957030740838232f});
  test<ElementType::kSI8, ElementType::kF16>(
      Cbrt, {4}, {.scale = 1e-1, .zero_point = -2}, {0, 1, -2, 3},
      {0, 1, -1.25992104989487316476f, 1.44224957030740838232f});
  test<ElementType::kSI8, ElementType::kF32>(
      Cbrt, {4}, {.scale = 1e-1, .zero_point = 4}, {0, 1, -2, 3},
      {0, 1, -1.25992104989487316476f, 1.44224957030740838232f});
  test<ElementType::kSI16, ElementType::kF32>(
      Cbrt, {4}, {.scale = 1e-1, .zero_point = 4}, {0, 1, -2, 3},
      {0, 1, -1.25992104989487316476f, 1.44224957030740838232f});
}

TEST(ElementwiseUnary, Ceil) {
  test<ElementType::kBF16>(Ceil, {4}, {0, 1.1, -2.7, 3.5}, {0, 2, -2, 4});
  test<ElementType::kF16>(Ceil, {4}, {0, 1.1, -2.7, 3.5}, {0, 2, -2, 4});
  test<ElementType::kF32>(Ceil, {4}, {0, 1.1, -2.7, 3.5}, {0, 2, -2, 4});
}

TEST(ElementwiseUnary, CeilQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Ceil, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1.1, -2.7, 3.5},
      {0, 2, -2, 4});
  test<ElementType::kSI8, ElementType::kF16>(
      Ceil, {4}, {.scale = 1e-1, .zero_point = 4}, {0, 1.1, -2.7, 3.5},
      {0, 2, -2, 4});
  test<ElementType::kSI8, ElementType::kF32>(
      Ceil, {4}, {.scale = 1e-1, .zero_point = -4}, {0, 1.1, -2.7, 3.5},
      {0, 2, -2, 4});
  test<ElementType::kSI16, ElementType::kF32>(
      Ceil, {4}, {.scale = 1e-2, .zero_point = -4}, {0, 1.11, -2.77, 3.55},
      {0, 2, -2, 4});
}

TEST(ElementwiseUnary, Cosine) {
  test<ElementType::kBF16>(Cosine, {4}, {0, 1.1, -1.1, 2.3},
                           {1, 0.45359612142557738777f, 0.45359612142557738777f,
                            -0.66627602127982419331f});
  test<ElementType::kF16>(Cosine, {4}, {0, 1.1, -1.1, 2.3},
                          {1, 0.45359612142557738777f, 0.45359612142557738777f,
                           -0.66627602127982419331f});
  test<ElementType::kF32>(Cosine, {4}, {0, 1.1, -1.1, 2.3},
                          {1, 0.45359612142557738777f, 0.45359612142557738777f,
                           -0.66627602127982419331f});
}

TEST(ElementwiseUnary, CosineQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Cosine, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1.1, -1.1, 2.3},
      {1, 0.45359612142557738777f, 0.45359612142557738777f,
       -0.66627602127982419331f});
  test<ElementType::kSI8, ElementType::kF16>(
      Cosine, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1.1, -1.1, 2.3},
      {1, 0.45359612142557738777f, 0.45359612142557738777f,
       -0.66627602127982419331f});
  test<ElementType::kSI8, ElementType::kF32>(
      Cosine, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1.1, -1.1, 2.3},
      {1, 0.45359612142557738777f, 0.45359612142557738777f,
       -0.66627602127982419331f});
  test<ElementType::kSI16, ElementType::kF32>(
      Cosine, {4}, {.scale = 1e-4, .zero_point = 0}, {0, 1.1, -1.1, 2.3},
      {1, 0.45359612142557738777f, 0.45359612142557738777f,
       -0.66627602127982419331f});
}

TEST(ElementwiseUnary, CountLeadingZeros) {
  test<ElementType::kSI8>(CountLeadingZeros, {4}, {0, 1, 127, -1},
                          {8, 7, 1, 0});
  test<ElementType::kSI16>(CountLeadingZeros, {4}, {0, 1, 32767, -1},
                           {16, 15, 1, 0});
  test<ElementType::kSI32>(CountLeadingZeros, {4}, {0, 1, 2147483647, -1},
                           {32, 31, 1, 0});
}

TEST(ElementwiseUnary, Exponential) {
  test<ElementType::kBF16>(Exponential, {4}, {0, 0.5, 1, 1.5},
                           {1, 1.64872127070012814684f, 2.71828182845904523536f,
                            4.48168907033806482260f});
  test<ElementType::kF16>(Exponential, {4}, {0, 0.5, 1, 1.5},
                          {1, 1.64872127070012814684f, 2.71828182845904523536f,
                           4.48168907033806482260f});
  test<ElementType::kF32>(Exponential, {4}, {0, 0.5, 1, 1.5},
                          {1, 1.64872127070012814684f, 2.71828182845904523536f,
                           4.48168907033806482260f});
}

TEST(ElementwiseUnary, ExponentialQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Exponential, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 0.5, 1, 1.5},
      {1, 1.64872127070012814684f, 2.71828182845904523536f,
       4.48168907033806482260f});
  test<ElementType::kSI8, ElementType::kF16>(
      Exponential, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 0.5, 1, 1.5},
      {1, 1.64872127070012814684f, 2.71828182845904523536f,
       4.48168907033806482260f});
  test<ElementType::kSI8, ElementType::kF32>(
      Exponential, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 0.5, 1, 1.5},
      {1, 1.64872127070012814684f, 2.71828182845904523536f,
       4.48168907033806482260f});
  test<ElementType::kSI16, ElementType::kF32>(
      Exponential, {4}, {.scale = 1e-2, .zero_point = 0}, {0, 0.5, 1, 1.5},
      {1, 1.64872127070012814684f, 2.71828182845904523536f,
       4.48168907033806482260f});
}

TEST(ElementwiseUnary, ExponentialMinusOne) {
  test<ElementType::kBF16>(ExponentialMinusOne, {4}, {0, 0.5, 1, 1.5},
                           {0, 0.64872127070012814684f, 1.71828182845904523536f,
                            3.48168907033806482260f});
  test<ElementType::kF16>(ExponentialMinusOne, {4}, {0, 0.5, 1, 1.5},
                          {0, 0.64872127070012814684f, 1.71828182845904523536f,
                           3.48168907033806482260f});
  test<ElementType::kF32>(ExponentialMinusOne, {4}, {0, 0.5, 1, 1.5},
                          {0, 0.64872127070012814684f, 1.71828182845904523536f,
                           3.48168907033806482260f});
}

TEST(ElementwiseUnary, ExponentialMinusOneQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      ExponentialMinusOne, {4}, {.scale = 1e-1, .zero_point = 0},
      {0, 0.5, 1, 1.5},
      {0, 0.64872127070012814684f, 1.71828182845904523536f,
       3.48168907033806482260f});
  test<ElementType::kSI8, ElementType::kF16>(
      ExponentialMinusOne, {4}, {.scale = 1e-1, .zero_point = 0},
      {0, 0.5, 1, 1.5},
      {0, 0.64872127070012814684f, 1.71828182845904523536f,
       3.48168907033806482260f});
  test<ElementType::kSI8, ElementType::kF32>(
      ExponentialMinusOne, {4}, {.scale = 1e-1, .zero_point = 0},
      {0, 0.5, 1, 1.5},
      {0, 0.64872127070012814684f, 1.71828182845904523536f,
       3.48168907033806482260f});
  test<ElementType::kSI16, ElementType::kF32>(
      ExponentialMinusOne, {4}, {.scale = 1e-2, .zero_point = 0},
      {0, 0.5, 1, 1.5},
      {0, 0.64872127070012814684f, 1.71828182845904523536f,
       3.48168907033806482260f});
}

TEST(ElementwiseUnary, Floor) {
  test<ElementType::kBF16>(Floor, {4}, {0, 1.1, -2.7, 3.5}, {0, 1, -3, 3});
  test<ElementType::kF16>(Floor, {4}, {0, 1.1, -2.7, 3.5}, {0, 1, -3, 3});
  test<ElementType::kF32>(Floor, {4}, {0, 1.1, -2.7, 3.5}, {0, 1, -3, 3});
}

TEST(ElementwiseUnary, FloorQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Floor, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1.1, -2.7, 3.5},
      {0, 1, -3, 3});
  test<ElementType::kSI8, ElementType::kF16>(
      Floor, {4}, {.scale = 1e-1, .zero_point = 4}, {0, 1.1, -2.7, 3.5},
      {0, 1, -3, 3});
  test<ElementType::kSI8, ElementType::kF32>(
      Floor, {4}, {.scale = 1e-1, .zero_point = -4}, {0, 1.1, -2.7, 3.5},
      {0, 1, -3, 3});
  test<ElementType::kSI16, ElementType::kF32>(
      Floor, {4}, {.scale = 1e-2, .zero_point = -4}, {0, 1.11, -2.77, 3.55},
      {0, 1, -3, 3});
}

TEST(ElementwiseUnary, Log) {
  test<ElementType::kBF16>(Log, {4}, {0.1, 0.5, 1, 1.5},
                           {-2.30258509299404568401f, -0.69314718055994530941f,
                            0, 0.40546510810816438197f});
  test<ElementType::kF16>(Log, {4}, {0.1, 0.5, 1, 1.5},
                          {-2.30258509299404568401f, -0.69314718055994530941f,
                           0, 0.40546510810816438197f});
  test<ElementType::kF32>(Log, {4}, {0.1, 0.5, 1, 1.5},
                          {-2.30258509299404568401f, -0.69314718055994530941f,
                           0, 0.40546510810816438197f});
}

TEST(ElementwiseUnary, LogQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Log, {4}, {.scale = 1e-1, .zero_point = -4}, {0.1, 0.5, 1, 1.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
  test<ElementType::kSI8, ElementType::kF16>(
      Log, {4}, {.scale = 1e-1, .zero_point = -4}, {0.1, 0.5, 1, 1.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
  test<ElementType::kSI8, ElementType::kF32>(
      Log, {4}, {.scale = 1e-1, .zero_point = -4}, {0.1, 0.5, 1, 1.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
  test<ElementType::kSI16, ElementType::kF32>(
      Log, {4}, {.scale = 1e-3, .zero_point = -4}, {0.1, 0.5, 1, 1.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
}

TEST(ElementwiseUnary, LogPlusOne) {
  test<ElementType::kBF16>(LogPlusOne, {4}, {-0.9, -0.5, 0, 0.5},
                           {-2.30258509299404568401f, -0.69314718055994530941f,
                            0, 0.40546510810816438197f});
  test<ElementType::kF16>(LogPlusOne, {4}, {-0.9, -0.5, 0, 0.5},
                          {-2.30258509299404568401f, -0.69314718055994530941f,
                           0, 0.40546510810816438197f});
  test<ElementType::kF32>(LogPlusOne, {4}, {-0.9, -0.5, 0, 0.5},
                          {-2.30258509299404568401f, -0.69314718055994530941f,
                           0, 0.40546510810816438197f});
}

TEST(ElementwiseUnary, LogPlusOneQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      LogPlusOne, {4}, {.scale = 1e-1, .zero_point = 0}, {-0.9, -0.5, 0, 0.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
  test<ElementType::kSI8, ElementType::kF16>(
      LogPlusOne, {4}, {.scale = 1e-1, .zero_point = 0}, {-0.9, -0.5, 0, 0.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
  test<ElementType::kSI8, ElementType::kF32>(
      LogPlusOne, {4}, {.scale = 1e-1, .zero_point = 0}, {-0.9, -0.5, 0, 0.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
  test<ElementType::kSI16, ElementType::kF32>(
      LogPlusOne, {4}, {.scale = 1e-4, .zero_point = 0}, {-0.9, -0.5, 0, 0.5},
      {-2.30258509299404568401f, -0.69314718055994530941f, 0,
       0.40546510810816438197f});
}

TEST(ElementwiseUnary, Logistic) {
  test<ElementType::kBF16>(Logistic, {4}, {-1, -0.5, 0, 0.5},
                           {0.26894142136999512074f, 0.37754066879814543536f,
                            0.5, 0.62245933120185456464f});
  test<ElementType::kF16>(Logistic, {4}, {-1, -0.5, 0, 0.5},
                          {0.26894142136999512074f, 0.37754066879814543536f,
                           0.5, 0.62245933120185456464f});
  test<ElementType::kF32>(Logistic, {4}, {-1, -0.5, 0, 0.5},
                          {0.26894142136999512074f, 0.37754066879814543536f,
                           0.5, 0.62245933120185456464f});
}

TEST(ElementwiseUnary, LogisticQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Logistic, {4}, {.scale = 1e-1, .zero_point = 0}, {-1, -0.5, 0, 0.5},
      {0.26894142136999512074f, 0.37754066879814543536f, 0.5,
       0.62245933120185456464f});
  test<ElementType::kSI8, ElementType::kF16>(
      Logistic, {4}, {.scale = 1e-1, .zero_point = 0}, {-1, -0.5, 0, 0.5},
      {0.26894142136999512074f, 0.37754066879814543536f, 0.5,
       0.62245933120185456464f});
  test<ElementType::kSI8, ElementType::kF32>(
      Logistic, {4}, {.scale = 1e-1, .zero_point = 0}, {-1, -0.5, 0, 0.5},
      {0.26894142136999512074f, 0.37754066879814543536f, 0.5,
       0.62245933120185456464f});
  test<ElementType::kSI16, ElementType::kF32>(
      Logistic, {4}, {.scale = 1e-3, .zero_point = 0}, {-1, -0.5, 0, 0.5},
      {0.26894142136999512074f, 0.37754066879814543536f, 0.5,
       0.62245933120185456464f});
}

TEST(ElementwiseUnary, Negate) {
  test<ElementType::kSI8>(Negate, {5}, {0, 1, -2, 3, -4}, {0, -1, 2, -3, 4});
  test<ElementType::kSI16>(Negate, {5}, {0, 1, -2, 3, -4}, {0, -1, 2, -3, 4});
  test<ElementType::kSI32>(Negate, {5}, {0, 1, -2, 3, -4}, {0, -1, 2, -3, 4});
  test<ElementType::kBF16>(Negate, {5}, {0, 1, -2, 3, -4}, {0, -1, 2, -3, 4});
  test<ElementType::kF16>(Negate, {5}, {0, 1, -2, 3, -4}, {0, -1, 2, -3, 4});
  test<ElementType::kF32>(Negate, {5}, {0, 1, -2, 3, -4}, {0, -1, 2, -3, 4});
}

TEST(ElementwiseBinary, NegateQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Negate, {5}, {.scale = 1, .zero_point = 0}, {0, 1, -2, 3, -4},
      {0, -1, 2, -3, 4});
  test<ElementType::kSI8, ElementType::kF16>(
      Negate, {5}, {.scale = 1e-1, .zero_point = 1}, {0, 1, -2, 3, -4},
      {0, -1, 2, -3, 4});
  test<ElementType::kSI8, ElementType::kF32>(
      Negate, {5}, {.scale = 1e-1, .zero_point = -1}, {0, 1, -2, 3, -4},
      {0, -1, 2, -3, 4});
  test<ElementType::kSI16, ElementType::kF32>(
      Negate, {5}, {.scale = 1e-3, .zero_point = -1}, {0, 1, -2, 3, -4},
      {0, -1, 2, -3, 4});
}

TEST(ElementwiseUnary, Not) {
  test<ElementType::kI1>(Not, {2}, {0, 1}, {1, 0});
  test<ElementType::kSI8>(Not, {5}, {-2, -1, 0, 1, 2},
                          {1, 0, int8_t(0xFF), int8_t(0xFE), int8_t(0xFD)});
  test<ElementType::kSI16>(
      Not, {5}, {-2, -1, 0, 1, 2},
      {1, 0, int16_t(0xFFFF), int16_t(0xFFFE), int16_t(0xFFFD)});
  test<ElementType::kSI32>(
      Not, {5}, {-2, -1, 0, 1, 2},
      {1, 0, int32_t(0xFFFFFFFFU), int32_t(0xFFFFFFFEU), int32_t(0xFFFFFFFDU)});
}

TEST(ElementwiseUnary, Popcnt) {
  test<ElementType::kSI8>(Popcnt, {4}, {0, 1, 2, 127}, {0, 1, 1, 7});
  test<ElementType::kSI16>(Popcnt, {4}, {0, 1, 2, 127}, {0, 1, 1, 7});
  test<ElementType::kSI32>(Popcnt, {4}, {0, 1, 2, 127}, {0, 1, 1, 7});
}

TEST(ElementwiseUnary, RoundNearestAfz) {
  test<ElementType::kBF16>(RoundNearestAfz, {5}, {-2.5, 0.4, 0.5, 0.6, 2.5},
                           {-3.0, 0.0, 1.0, 1.0, 3.0});
  test<ElementType::kF16>(RoundNearestAfz, {5}, {-2.5, 0.4, 0.5, 0.6, 2.5},
                          {-3.0, 0.0, 1.0, 1.0, 3.0});
  test<ElementType::kF32>(RoundNearestAfz, {5}, {-2.5, 0.4, 0.5, 0.6, 2.5},
                          {-3.0, 0.0, 1.0, 1.0, 3.0});
}

TEST(ElementwiseBinary, RoundNearestAfzQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      RoundNearestAfz, {5}, {.scale = 1e-1, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-3.0, 0.0, 1.0, 1.0, 3.0});
  test<ElementType::kSI8, ElementType::kF16>(
      RoundNearestAfz, {5}, {.scale = 1e-1, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-3.0, 0.0, 1.0, 1.0, 3.0});
  test<ElementType::kSI8, ElementType::kF32>(
      RoundNearestAfz, {5}, {.scale = 1e-1, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-3.0, 0.0, 1.0, 1.0, 3.0});
  test<ElementType::kSI16, ElementType::kF32>(
      RoundNearestAfz, {5}, {.scale = 1e-2, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-3.0, 0.0, 1.0, 1.0, 3.0});
}

TEST(ElementwiseUnary, RoundNearestEven) {
  test<ElementType::kBF16>(RoundNearestEven, {5}, {-2.5, 0.4, 0.5, 0.6, 2.5},
                           {-2.0, 0.0, 0.0, 1.0, 2.0});
  test<ElementType::kF16>(RoundNearestEven, {5}, {-2.5, 0.4, 0.5, 0.6, 2.5},
                          {-2.0, 0.0, 0.0, 1.0, 2.0});
  test<ElementType::kF32>(RoundNearestEven, {5}, {-2.5, 0.4, 0.5, 0.6, 2.5},
                          {-2.0, 0.0, 0.0, 1.0, 2.0});
}

TEST(ElementwiseBinary, RoundNearestEvenQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      RoundNearestEven, {5}, {.scale = 1e-1, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-2.0, 0.0, 0.0, 1.0, 2.0});
  test<ElementType::kSI8, ElementType::kF16>(
      RoundNearestEven, {5}, {.scale = 1e-1, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-2.0, 0.0, 0.0, 1.0, 2.0});
  test<ElementType::kSI8, ElementType::kF32>(
      RoundNearestEven, {5}, {.scale = 1e-1, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-2.0, 0.0, 0.0, 1.0, 2.0});
  test<ElementType::kSI16, ElementType::kF32>(
      RoundNearestEven, {5}, {.scale = 1e-2, .zero_point = 0},
      {-2.5, 0.4, 0.5, 0.6, 2.5}, {-2.0, 0.0, 0.0, 1.0, 2.0});
}

TEST(ElementwiseUnary, Rsqrt) {
  test<ElementType::kBF16>(Rsqrt, {4}, {1.0, 4.0, 9.0, 25.0},
                           {1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 5.0});
  test<ElementType::kF16>(Rsqrt, {4}, {1.0, 4.0, 9.0, 25.0},
                          {1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 5.0});
  test<ElementType::kF32>(Rsqrt, {4}, {1.0, 4.0, 9.0, 25.0},
                          {1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 5.0});
}

TEST(ElementwiseUnary, RsqrtQuantized) {
  test<ElementType::kSI16, ElementType::kF32>(
      Rsqrt, {4}, {.scale = 1e-3, .zero_point = 0}, {1.0, 4.0, 9.0, 25.0},
      {1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 5.0});
}

TEST(ElementwiseUnary, Sign) {
  test<ElementType::kSI8>(Sign, {3}, {-2, 0, 2}, {-1, 0, 1});
  test<ElementType::kSI16>(Sign, {3}, {-2, 0, 2}, {-1, 0, 1});
  test<ElementType::kSI32>(Sign, {3}, {-2, 0, 2}, {-1, 0, 1});
  test<ElementType::kBF16>(
      Sign, {8}, {+NAN, -NAN, +INFINITY, -INFINITY, -2.0, -0.0, +0.0, 2.0},
      {NAN, NAN, 1, -1, -1, 0, 0, 1});
  test<ElementType::kF16>(
      Sign, {8}, {+NAN, -NAN, +INFINITY, -INFINITY, -2.0, -0.0, +0.0, 2.0},
      {NAN, NAN, 1, -1, -1, 0, 0, 1});
  test<ElementType::kF32>(
      Sign, {8}, {+NAN, -NAN, +INFINITY, -INFINITY, -2.0, -0.0, +0.0, 2.0},
      {NAN, NAN, 1, -1, -1, 0, 0, 1});
}

TEST(ElementwiseUnary, SignQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Sign, {4}, {.scale = 1e-1, .zero_point = 0}, {-2.0, -0.0, +0.0, 2.0},
      {-1, 0, 0, 1});
  test<ElementType::kSI8, ElementType::kF16>(
      Sign, {4}, {.scale = 1e-1, .zero_point = 0}, {-2.0, -0.0, +0.0, 2.0},
      {-1, 0, 0, 1});
  test<ElementType::kSI8, ElementType::kF32>(
      Sign, {4}, {.scale = 1e-1, .zero_point = 0}, {-2.0, -0.0, +0.0, 2.0},
      {-1, 0, 0, 1});
  test<ElementType::kSI16, ElementType::kF32>(
      Sign, {4}, {.scale = 1e-2, .zero_point = 0}, {-2.0, -0.0, +0.0, 2.0},
      {-1, 0, 0, 1});
}

TEST(ElementwiseUnary, Sine) {
  test<ElementType::kBF16>(Sine, {5}, {0, M_PI_2, M_PI, 3 * M_PI_2, 2 * M_PI},
                           {0, 1, 0, -1, 0});
  test<ElementType::kF16>(Sine, {5}, {0, M_PI_2, M_PI, 3 * M_PI_2, 2 * M_PI},
                          {0, 1, 0, -1, 0});
  test<ElementType::kF32>(Sine, {5}, {0, M_PI_2, M_PI, 3 * M_PI_2, 2 * M_PI},
                          {0, 1, 0, -1, 0});
}

TEST(ElementwiseUnary, SineQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Sine, {5}, {.scale = 1e-1, .zero_point = 0},
      {0, M_PI_2, M_PI, 3 * M_PI_2, 2 * M_PI}, {0, 1, 0, -1, 0});
  test<ElementType::kSI8, ElementType::kF16>(
      Sine, {5}, {.scale = 1e-1, .zero_point = 0},
      {0, M_PI_2, M_PI, 3 * M_PI_2, 2 * M_PI}, {0, 1, 0, -1, 0});
  test<ElementType::kSI8, ElementType::kF32>(
      Sine, {5}, {.scale = 1e-1, .zero_point = 0},
      {0, M_PI_2, M_PI, 3 * M_PI_2, 2 * M_PI}, {0, 1, 0, -1, 0});
  test<ElementType::kSI16, ElementType::kF32>(
      Sine, {5}, {.scale = 1e-2, .zero_point = 0},
      {0, M_PI_2, M_PI, 3 * M_PI_2, 2 * M_PI}, {0, 1, 0, -1, 0});
}

TEST(ElementwiseUnary, Sqrt) {
  test<ElementType::kBF16>(Sqrt, {4}, {0, 1, 4, 9}, {0, 1, 2, 3});
  test<ElementType::kF16>(Sqrt, {4}, {0, 1, 4, 9}, {0, 1, 2, 3});
  test<ElementType::kF32>(Sqrt, {4}, {0, 1, 4, 9}, {0, 1, 2, 3});
}

TEST(ElementwiseUnary, SqrtQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Sqrt, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1, 4, 9}, {0, 1, 2, 3});
  test<ElementType::kSI8, ElementType::kF16>(
      Sqrt, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1, 4, 9}, {0, 1, 2, 3});
  test<ElementType::kSI8, ElementType::kF32>(
      Sqrt, {4}, {.scale = 1e-1, .zero_point = 0}, {0, 1, 4, 9}, {0, 1, 2, 3});
  test<ElementType::kSI16, ElementType::kF32>(
      Sqrt, {4}, {.scale = 1e-2, .zero_point = 0}, {0, 1, 4, 9}, {0, 1, 2, 3});
}

TEST(ElementwiseUnary, Tanh) {
  test<ElementType::kBF16>(Tanh, {3}, {-1, 0, 1},
                           {-0.76159416, 0.0, 0.76159416});
  test<ElementType::kF16>(Tanh, {3}, {-1, 0, 1},
                          {-0.76159416, 0.0, 0.76159416});
  test<ElementType::kF32>(Tanh, {3}, {-1, 0, 1},
                          {-0.76159416, 0.0, 0.76159416});
}

TEST(ElementwiseUnary, TanhQuantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      Tanh, {3}, {.scale = 1e-1, .zero_point = 0}, {-1, 0, 1},
      {-0.76159416, 0.0, 0.76159416});
  test<ElementType::kSI8, ElementType::kF16>(
      Tanh, {3}, {.scale = 1e-1, .zero_point = 0}, {-1, 0, 1},
      {-0.76159416, 0.0, 0.76159416});
  test<ElementType::kSI8, ElementType::kF32>(
      Tanh, {3}, {.scale = 1e-1, .zero_point = 0}, {-1, 0, 1},
      {-0.76159416, 0.0, 0.76159416});
  test<ElementType::kSI16, ElementType::kF32>(
      Tanh, {3}, {.scale = 1e-2, .zero_point = 0}, {-1, 0, 1},
      {-0.76159416, 0.0, 0.76159416});
}

}  // namespace testing
}  // namespace stablehlo
