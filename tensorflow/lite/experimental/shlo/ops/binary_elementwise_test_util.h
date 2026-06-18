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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BINARY_ELEMENTWISE_TEST_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BINARY_ELEMENTWISE_TEST_UTIL_H_

#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <class Op, class List>
struct OpTuple;

template <class Op, class... Ts>
struct OpTuple<Op, ::testing::Types<Ts...>> {
  using Type = std::tuple<Op, Ts...>;
};

template <class Op>
struct OpTupleFactory {
  template <class T>
  using WithOp = typename OpTuple<Op, T>::Type;
};

template <class Op, class SupportedTypes>
using BinaryElementwiseBaselineConstraintTypes =
    MapTypes<OpTupleFactory<Op>::template WithOp,
             FilterTypes<NegatePred<SameTypes>::template Predicate,
                         CrossProductTypes<SupportedTypes, SupportedTypes,
                                           SupportedTypes>>>;

using BaselineConstraintIntTypes = ::testing::Types<TestParam<DataType::kSI32>>;

using BaselineConstraintFloatTypes =
    ::testing::Types<TestParam<DataType::kF32>>;

using BaselineConstraintQuantizedPerTensorTypes =
    ::testing::Types<PerTensor<TestParam<DataType::kSI8, DataType::kF32>>,
                     PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>>;

template <class Op>
class BinaryElementwiseOpShapePropagationTest : public ::testing::Test {
 protected:
  void SetRhsShape(Shape shape) { rhs_tensor_.shape() = std::move(shape); }
  void SetOutputShape(Shape shape) {
    output_tensor_.shape() = std::move(shape);
  }
  bool LhsAndOutputShapesAreEqual() const {
    return lhs_tensor_.shape() == output_tensor_.shape();
  }

  Op op_ = Create(SupportedOpAttributes<Op>::Get());
  Tensor lhs_tensor_ = {
      .type = TensorType{.shape = Shape({2, 3, 4}),
                         .element_type = SupportedOpDataType<Op>::kStorageType},
      .data = nullptr};
  Tensor rhs_tensor_ = {
      .type = TensorType{.shape = Shape({2, 3, 4}),
                         .element_type = SupportedOpDataType<Op>::kStorageType},
      .data = nullptr};
  Tensor output_tensor_ = {
      .type = TensorType{.shape = Shape(),
                         .element_type =
                             SupportedOpOutputDataType<Op>::kStorageType},
      .data = nullptr};
};

TYPED_TEST_SUITE_P(BinaryElementwiseOpShapePropagationTest);

TYPED_TEST_P(BinaryElementwiseOpShapePropagationTest, ShapePropagationWorks) {
  ASSERT_TRUE(this->output_tensor_.shape().empty());
  EXPECT_OK(Prepare(this->op_, this->lhs_tensor_, this->rhs_tensor_,
                    this->output_tensor_));
  EXPECT_THAT(this->output_tensor_.shape(),
              ::testing::ElementsAreArray(this->lhs_tensor_.shape()));
}

TYPED_TEST_P(BinaryElementwiseOpShapePropagationTest,
             SmallerOutputShapeRaisesAnError) {
  this->SetOutputShape(Shape({2, 3}));
  ASSERT_FALSE(this->LhsAndOutputShapesAreEqual());
  EXPECT_EQ(
      Prepare(this->op_, this->lhs_tensor_, this->rhs_tensor_,
              this->output_tensor_),
      absl::FailedPreconditionError("The specified output tensor shape is not "
                                    "compatible with the input shapes."));
}

TYPED_TEST_P(BinaryElementwiseOpShapePropagationTest,
             BiggerOutputShapeRaisesAnError) {
  this->SetOutputShape(Shape({2, 3, 4, 5}));
  ASSERT_FALSE(this->LhsAndOutputShapesAreEqual());
  EXPECT_EQ(
      Prepare(this->op_, this->lhs_tensor_, this->rhs_tensor_,
              this->output_tensor_),
      absl::FailedPreconditionError("The specified output tensor shape is not "
                                    "compatible with the input shapes."));
}

TYPED_TEST_P(BinaryElementwiseOpShapePropagationTest,
             IncompatibleOutputShapeRaisesAnError) {
  this->SetOutputShape(Shape({2, 3, 5}));
  ASSERT_FALSE(this->LhsAndOutputShapesAreEqual());
  EXPECT_EQ(
      Prepare(this->op_, this->lhs_tensor_, this->rhs_tensor_,
              this->output_tensor_),
      absl::FailedPreconditionError("The specified output tensor shape is not "
                                    "compatible with the input shapes."));
}

REGISTER_TYPED_TEST_SUITE_P(BinaryElementwiseOpShapePropagationTest,
                            ShapePropagationWorks,
                            SmallerOutputShapeRaisesAnError,
                            BiggerOutputShapeRaisesAnError,
                            IncompatibleOutputShapeRaisesAnError);

// Tests that the baseline element type of the input and output tensors is the
// same.
template <class T>
class BinaryElementwiseSameBaselineElementTypeConstraintTest
    : public ::testing::Test {};

TYPED_TEST_SUITE_P(BinaryElementwiseSameBaselineElementTypeConstraintTest);

TYPED_TEST_P(BinaryElementwiseSameBaselineElementTypeConstraintTest,
             DifferentInputOutputStorageTypesRaiseAnError) {
  using Op = std::tuple_element_t<0, TypeParam>;
  using LhsTypeDesc = std::tuple_element_t<1, TypeParam>;
  using RhsTypeDesc = std::tuple_element_t<2, TypeParam>;
  using ResultTypeDesc = std::tuple_element_t<3, TypeParam>;
  const Shape shape({2, 3, 4});
  Tensor lhs_tensor{.type = TensorTypeFor(LhsTypeDesc{}, shape),
                    .data = nullptr};
  Tensor rhs_tensor{.type = TensorTypeFor(RhsTypeDesc{}, shape),
                    .data = nullptr};
  Tensor output_tensor{.type = TensorTypeFor(ResultTypeDesc{}, shape),
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

REGISTER_TYPED_TEST_SUITE_P(
    BinaryElementwiseSameBaselineElementTypeConstraintTest,
    DifferentInputOutputStorageTypesRaiseAnError);

// Tests that unsupported types are detected during when `Prepare` is called.
template <class T>
class BinaryElementwiseUnsupportedTypeTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(BinaryElementwiseUnsupportedTypeTest);

TYPED_TEST_P(BinaryElementwiseUnsupportedTypeTest, PrepareRaisesAnError) {
  using Op = std::tuple_element_t<0, TypeParam>;
  using TypeDesc = std::tuple_element_t<1, TypeParam>;
  Tensor input_tensor{.type = TensorTypeFor(TypeDesc{}, Shape({2, 3, 4})),
                      .data = nullptr};
  Tensor output_tensor = input_tensor;
  auto op = Create(typename Op::Attributes{});
  const absl::Status status =
      Prepare(op, input_tensor, input_tensor, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Unsupported tensor type"));
}

REGISTER_TYPED_TEST_SUITE_P(BinaryElementwiseUnsupportedTypeTest,
                            PrepareRaisesAnError);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BINARY_ELEMENTWISE_TEST_UTIL_H_
