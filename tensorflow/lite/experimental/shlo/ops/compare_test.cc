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

#include "tensorflow/lite/experimental/shlo/ops/compare.h"

#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise_test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::FloatEq;
using testing::Pointwise;

namespace shlo_ref {

template <>
struct ParamName<CompareOp> {
  static std::string Get() { return "Compare"; }
};

struct Compare {
  template <class T>
  constexpr bool operator()(const T a, const T b) const {
    switch (comparison_direction) {
      case CompareOp::ComparisonDirection::kEq:
        return a == b;
      case CompareOp::ComparisonDirection::kNe:
        return a != b;
      case CompareOp::ComparisonDirection::kGe:
        return a >= b;
      case CompareOp::ComparisonDirection::kGt:
        return a > b;
      case CompareOp::ComparisonDirection::kLe:
        return a <= b;
      case CompareOp::ComparisonDirection::kLt:
        return a < b;
    }
    return false;
  }

  CompareOp::ComparisonDirection comparison_direction;
};

const char* ToString(CompareOp::ComparisonDirection comparison_direction) {
  switch (comparison_direction) {
    case CompareOp::ComparisonDirection::kEq:
      return "eq";
    case CompareOp::ComparisonDirection::kNe:
      return "ne";
    case CompareOp::ComparisonDirection::kGe:
      return "ge";
    case CompareOp::ComparisonDirection::kGt:
      return "gt";
    case CompareOp::ComparisonDirection::kLe:
      return "le";
    case CompareOp::ComparisonDirection::kLt:
      return "lt";
  }
}

template <>
struct SupportedOpAttributes<CompareOp> {
  static CompareOp::Attributes Get() {
    return {.comparison_direction = CompareOp::ComparisonDirection::kEq};
  }
};

template <>
struct SupportedOpOutputDataType<CompareOp> {
  static constexpr DataType kStorageType = DataType::kI1;
};

namespace {
INSTANTIATE_TYPED_TEST_SUITE_P(Compare, BinaryElementwiseOpShapePropagationTest,
                               CompareOp, TestParamNames);

// Tests that the baseline element type of the input and output tensors is the
// same.
//
// Compare has input/output constraints that are different from the other binary
// element wise ops. Thus the specific test suite.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    BinaryElementwiseSameBaselineElementTypeConstraintTest);

template <class Op, class SupportedTypes>
using CompareBaselineConstraintTypesCrossProduct =
    MapTypes<OpTupleFactory<Op>::template WithOp,
             FilterTypes<NegatePred<SameTypes>::template Predicate,
                         CrossProductTypes<SupportedTypes, SupportedTypes>>>;

using CompareBaselineContraintTypes =
    CompareBaselineConstraintTypesCrossProduct<
        CompareOp, ConcatTypes<BoolTestType, BaselineConstraintIntTypes,
                               BaselineConstraintFloatTypes,
                               BaselineConstraintQuantizedPerTensorTypes>>;

template <class T>
class CompareSameBaselineElementTypeConstraintTest : public ::testing::Test {};

TYPED_TEST_SUITE(CompareSameBaselineElementTypeConstraintTest,
                 CompareBaselineContraintTypes, TestParamNames);

TYPED_TEST(CompareSameBaselineElementTypeConstraintTest,
           DifferentInputOutputStorageTypesRaiseAnError) {
  using Op = std::tuple_element_t<0, TypeParam>;
  using LhsTypeDesc = std::tuple_element_t<1, TypeParam>;
  using RhsTypeDesc = std::tuple_element_t<2, TypeParam>;
  const Shape shape({2, 3, 4});
  Tensor lhs_tensor{.type = TensorTypeFor(LhsTypeDesc{}, shape),
                    .data = nullptr};
  Tensor rhs_tensor{.type = TensorTypeFor(RhsTypeDesc{}, shape),
                    .data = nullptr};
  Tensor output_tensor{.type = TensorTypeFor(TestParam<DataType::kI1>{}, shape),
                       .data = nullptr};
  auto op = Create(typename Op::Attributes{});
  const absl::Status status =
      Prepare(op, lhs_tensor, rhs_tensor, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      status.message(),
      ::testing::ContainsRegex(
          "stablehlo.[_a-z]+: baseline type constraint is not satisfied"));
}

using UnsupportedTypes =
    WithOpTypes<CompareOp, ConcatTypes<PerAxisQuantizedTestTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Compare, BinaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

using SupportedTypes = ConcatTypes<BoolTestType, ArithmeticTestTypes>;

template <class T>
struct CompareTest : ::testing::Test {};

TYPED_TEST_SUITE(CompareTest, SupportedTypes, TestParamNames);

TYPED_TEST(CompareTest, SupportedTestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  absl::BitGen bit_gen;
  const Shape shape({2, 3, 4});
  Vector<StorageT> lhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-50, /*max=*/50);
  Vector<StorageT> rhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/1, /*max=*/5);
  Vector<StorageT> output_data(shape.NumElements());
  Tensor lhs_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = lhs_data.data()};
  Tensor rhs_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = DataType::kI1},
      .data = output_data.data()};

  const CompareOp::ComparisonDirection comparison_direction =
      static_cast<CompareOp::ComparisonDirection>(absl::Uniform(bit_gen, 0, 6));

  Compare compare_ref{comparison_direction};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(lhs_data, rhs_data, expected_data.begin(), compare_ref);

  auto op = Create(
      CompareOp::Attributes{.comparison_direction = comparison_direction});
  ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct QuantizedCompareTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedCompareTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedCompareTest, PerTensorWorks) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  absl::BitGen bit_gen;
  const Shape shape({2, 2, 2});
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(2);
  Vector<StorageT> lhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-50, /*max=*/50);
  Vector<StorageT> rhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/zero_point + 1,
                                        /*max=*/zero_point + 5);
  Vector<StorageType<DataType::kI1>> output_data(shape.NumElements());
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  Tensor lhs_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
      .data = lhs_data.data()};
  Tensor rhs_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
      .data = rhs_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = DataType::kI1},
      .data = output_data.data()};

  const CompareOp::ComparisonDirection comparison_direction =
      CompareOp::ComparisonDirection::kEq;
  // static_cast<CompareOp::ComparisonDirection>(absl::Uniform(bit_gen,
  // 0, 6));

  Compare compare_ref{comparison_direction};
  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      lhs_data, rhs_data, expected_data.begin(),
      [zero_point, scale, compare_ref](auto lhs, auto rhs) {
        const ExpressedT dequantized_lhs = Dequantize(lhs, zero_point, scale);
        const ExpressedT dequantized_rhs = Dequantize(rhs, zero_point, scale);
        return compare_ref(dequantized_lhs, dequantized_rhs);
      });

  auto op = Create(
      CompareOp::Attributes{.comparison_direction = comparison_direction});
  ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data))
      << "lhs " << ::testing::PrintToString(lhs_data) << "\n"
      << "rhs " << ::testing::PrintToString(rhs_data) << "\n"
      << "dir " << ToString(comparison_direction) << "\n";
}
}  // namespace
}  // namespace shlo_ref
