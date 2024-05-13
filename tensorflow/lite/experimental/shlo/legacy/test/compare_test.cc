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
void test(
    ComparisonDirection comparison_direction, CompareType compare_type,
    std::initializer_list<DimensionSize>&& shape,
    std::vector<typename Storage<element_type>::Type>&& lhs_values,
    std::vector<typename Storage<element_type>::Type>&& rhs_values,
    std::vector<typename Storage<ElementType::kI1>::Type>&& expected_values) {
  Tensor lhs(TensorType(Shape(shape), element_type), lhs_values.data());
  Tensor rhs(TensorType(Shape(shape), element_type), rhs_values.data());
  Tensor expected(TensorType(Shape(shape), ElementType::kI1),
                  expected_values.data());

  std::vector<typename Storage<ElementType::kI1>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), ElementType::kI1),
                result_values.data());

  ASSERT_OK(Compare(lhs, rhs, comparison_direction, compare_type, result));
  EXPECT_EQ(result, expected)
      << "comparison_direction: " << comparison_direction
      << "\ncompare_type: " << compare_type << "\nlhs: " << lhs
      << "\nrhs: " << rhs;
}

template <ElementType storage_type, ElementType expressed_type>
void test(
    QuantizedParameter&& quantized_parameter,
    ComparisonDirection comparison_direction, CompareType compare_type,
    std::initializer_list<DimensionSize>&& shape,
    std::vector<typename Storage<expressed_type>::Type>&& lhs_values,
    std::vector<typename Storage<expressed_type>::Type>&& rhs_values,
    std::vector<typename Storage<ElementType::kI1>::Type>&& expected_values) {
  auto lhs_quant_values = QuantizeVector<storage_type, expressed_type>(
      lhs_values, quantized_parameter);
  auto rhs_quant_values = QuantizeVector<storage_type, expressed_type>(
      rhs_values, quantized_parameter);

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

  Tensor expected(TensorType(Shape(shape), ElementType::kI1),
                  expected_values.data());

  std::vector<typename Storage<ElementType::kI1>::Type> result_values(
      expected_values.size());
  Tensor result(TensorType(Shape(shape), ElementType::kI1),
                result_values.data());

  ASSERT_OK(Compare(lhs, rhs, comparison_direction, compare_type, result));
  EXPECT_EQ(result, expected)
      << "comparison_direction: " << comparison_direction
      << "\ncompare_type: " << compare_type << "\nlhs: " << lhs
      << "\nrhs: " << rhs;
}

TEST(Compare, Unquantized) {
  test<ElementType::kI1>(ComparisonDirection::kEQ, CompareType::kUnsigned, {4},
                         {true, false, true, false}, {true, true, false, false},
                         {true, false, false, true});
  test<ElementType::kSI8>(ComparisonDirection::kEQ, CompareType::kSigned, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, false, false, true});
  test<ElementType::kSI16>(ComparisonDirection::kEQ, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, false, false, true});
  test<ElementType::kSI32>(ComparisonDirection::kEQ, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, false, false, true});
  test<ElementType::kBF16>(ComparisonDirection::kEQ, CompareType::kFloat, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, false, false, true});
  test<ElementType::kF16>(ComparisonDirection::kEQ, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, false, false, true});
  test<ElementType::kF32>(ComparisonDirection::kEQ, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, false, false, true});

  test<ElementType::kI1>(ComparisonDirection::kNE, CompareType::kUnsigned, {4},
                         {true, false, true, false}, {true, true, false, false},
                         {false, true, true, false});
  test<ElementType::kSI8>(ComparisonDirection::kNE, CompareType::kSigned, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, true, true, false});
  test<ElementType::kSI16>(ComparisonDirection::kNE, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, true, true, false});
  test<ElementType::kSI32>(ComparisonDirection::kNE, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, true, true, false});
  test<ElementType::kBF16>(ComparisonDirection::kNE, CompareType::kFloat, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, true, true, false});
  test<ElementType::kF16>(ComparisonDirection::kNE, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, true, true, false});
  test<ElementType::kF32>(ComparisonDirection::kNE, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, true, true, false});

  test<ElementType::kI1>(ComparisonDirection::kGE, CompareType::kUnsigned, {4},
                         {true, false, true, false}, {true, true, false, false},
                         {true, false, true, true});
  test<ElementType::kSI8>(ComparisonDirection::kGE, CompareType::kSigned, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, false, true, true});
  test<ElementType::kSI16>(ComparisonDirection::kGE, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, false, true, true});
  test<ElementType::kSI32>(ComparisonDirection::kGE, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, false, true, true});
  test<ElementType::kBF16>(ComparisonDirection::kGE, CompareType::kFloat, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, false, true, true});
  test<ElementType::kF16>(ComparisonDirection::kGE, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, false, true, true});
  test<ElementType::kF32>(ComparisonDirection::kGE, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, false, true, true});

  test<ElementType::kI1>(ComparisonDirection::kGT, CompareType::kUnsigned, {4},
                         {true, false, true, false}, {true, true, false, false},
                         {false, false, true, false});
  test<ElementType::kSI8>(ComparisonDirection::kGT, CompareType::kSigned, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, false, true, false});
  test<ElementType::kSI16>(ComparisonDirection::kGT, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, false, true, false});
  test<ElementType::kSI32>(ComparisonDirection::kGT, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, false, true, false});
  test<ElementType::kBF16>(ComparisonDirection::kGT, CompareType::kFloat, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, false, true, false});
  test<ElementType::kF16>(ComparisonDirection::kGT, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, false, true, false});
  test<ElementType::kF32>(ComparisonDirection::kGT, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, false, true, false});

  test<ElementType::kI1>(ComparisonDirection::kLE, CompareType::kUnsigned, {4},
                         {true, false, true, false}, {true, true, false, false},
                         {true, true, false, true});
  test<ElementType::kSI8>(ComparisonDirection::kLE, CompareType::kSigned, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, true, false, true});
  test<ElementType::kSI16>(ComparisonDirection::kLE, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, true, false, true});
  test<ElementType::kSI32>(ComparisonDirection::kLE, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, true, false, true});
  test<ElementType::kBF16>(ComparisonDirection::kLE, CompareType::kFloat, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {true, true, false, true});
  test<ElementType::kF16>(ComparisonDirection::kLE, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, true, false, true});
  test<ElementType::kF32>(ComparisonDirection::kLE, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {true, true, false, true});

  test<ElementType::kI1>(ComparisonDirection::kLT, CompareType::kUnsigned, {4},
                         {true, false, true, false}, {true, true, false, false},
                         {false, true, false, false});
  test<ElementType::kSI8>(ComparisonDirection::kLT, CompareType::kSigned, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, true, false, false});
  test<ElementType::kSI16>(ComparisonDirection::kLT, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, true, false, false});
  test<ElementType::kSI32>(ComparisonDirection::kLT, CompareType::kSigned, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, true, false, false});
  test<ElementType::kBF16>(ComparisonDirection::kLT, CompareType::kFloat, {4},
                           {1, 0, 1, 0}, {1, 1, 0, 0},
                           {false, true, false, false});
  test<ElementType::kF16>(ComparisonDirection::kLT, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, true, false, false});
  test<ElementType::kF32>(ComparisonDirection::kLT, CompareType::kFloat, {4},
                          {1, 0, 1, 0}, {1, 1, 0, 0},
                          {false, true, false, false});
}

TEST(Compare, Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kEQ,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, false, true});

  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kNE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, true, false});

  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, false, true, true});

  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kGT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, false, true, false});

  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLE,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {true, true, false, true});

  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0}, ComparisonDirection::kLT,
      CompareType::kFloat, {4}, {1, 0, 1, 0}, {1, 1, 0, 0},
      {false, true, false, false});
}

}  // namespace testing
}  // namespace stablehlo
