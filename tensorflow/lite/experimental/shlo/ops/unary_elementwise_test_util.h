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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UNARY_ELEMENTWISE_TEST_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UNARY_ELEMENTWISE_TEST_UTIL_H_

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

template <class Op>
using BaselineMismatchSignedIntegerTypes = ::testing::Types<
    std::tuple<Op, TestParam<DataType::kSI4>, TestParam<DataType::kSI8>>,
    std::tuple<Op, TestParam<DataType::kSI4>, TestParam<DataType::kSI16>>,
    std::tuple<Op, TestParam<DataType::kSI8>, TestParam<DataType::kSI4>>,
    std::tuple<Op, TestParam<DataType::kSI8>, TestParam<DataType::kSI16>>,
    std::tuple<Op, TestParam<DataType::kSI16>, TestParam<DataType::kSI4>>,
    std::tuple<Op, TestParam<DataType::kSI16>, TestParam<DataType::kSI8>>>;

// Lists couples of unmatched baseline element types.
template <class Op>
using UnaryElementwiseConstraint1Types = ::testing::Types<
    std::tuple<Op, TestParam<DataType::kF16>, TestParam<DataType::kBF16>>,
    std::tuple<Op, TestParam<DataType::kF16>, TestParam<DataType::kF32>>,
    std::tuple<Op, TestParam<DataType::kBF16>, TestParam<DataType::kF16>>,
    std::tuple<Op, TestParam<DataType::kBF16>, TestParam<DataType::kF32>>,
    std::tuple<Op, TestParam<DataType::kF32>, TestParam<DataType::kF16>>,
    std::tuple<Op, TestParam<DataType::kF32>, TestParam<DataType::kBF16>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI16, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI16, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI4, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI16, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF16>>,
               PerTensor<TestParam<DataType::kSI16, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>,
               PerTensor<TestParam<DataType::kSI16, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI8, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI16, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI16, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI16, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI16, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI4, DataType::kF32>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI16, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI16, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kBF16>>>,
    std::tuple<Op, PerTensor<TestParam<DataType::kSI16, DataType::kF32>>,
               PerTensor<TestParam<DataType::kSI8, DataType::kF32>>>>;

// Tests that the input shape is compared to the output shape and that it is
// propagated if needed.

template <class Op>
class UnaryElementwiseOpShapePropagationTest : public ::testing::Test {
 protected:
  void SetOutputShape(Shape shape) {
    output_tensor_.shape() = std::move(shape);
  }
  bool InputAndOutputShapesAreEqual() const {
    return input_tensor_.shape() == output_tensor_.shape();
  }

  Op op_ = Create(typename Op::Attributes{});
  Tensor input_tensor_ = {
      .type = TensorType{.shape = Shape({2, 3, 4}),
                         .element_type = SupportedOpDataType<Op>::kStorageType},
      .data = nullptr};
  Tensor output_tensor_ = {
      .type = TensorType{.shape = Shape(),
                         .element_type = SupportedOpDataType<Op>::kStorageType},
      .data = nullptr};
};

TYPED_TEST_SUITE_P(UnaryElementwiseOpShapePropagationTest);

TYPED_TEST_P(UnaryElementwiseOpShapePropagationTest, ShapePropagationWorks) {
  ASSERT_TRUE(this->output_tensor_.shape().empty());
  EXPECT_OK(Prepare(this->op_, this->input_tensor_, this->output_tensor_));
  EXPECT_THAT(this->output_tensor_.shape(),
              ::testing::ElementsAreArray(this->input_tensor_.shape()));
}

TYPED_TEST_P(UnaryElementwiseOpShapePropagationTest,
             SmallerOutputShapeRaisesAnError) {
  this->SetOutputShape(Shape({2, 3}));
  ASSERT_FALSE(this->InputAndOutputShapesAreEqual());
  EXPECT_EQ(
      Prepare(this->op_, this->input_tensor_, this->output_tensor_),
      absl::FailedPreconditionError("The specified output tensor shape is not "
                                    "compatible with the input shape."));
}

TYPED_TEST_P(UnaryElementwiseOpShapePropagationTest,
             BiggerOutputShapeRaisesAnError) {
  this->SetOutputShape(Shape({2, 3, 4, 5}));
  ASSERT_FALSE(this->InputAndOutputShapesAreEqual());
  EXPECT_EQ(
      Prepare(this->op_, this->input_tensor_, this->output_tensor_),
      absl::FailedPreconditionError("The specified output tensor shape is not "
                                    "compatible with the input shape."));
}

TYPED_TEST_P(UnaryElementwiseOpShapePropagationTest,
             IncompatibleOutputShapeRaisesAnError) {
  this->SetOutputShape(Shape({2, 3, 5}));
  ASSERT_FALSE(this->InputAndOutputShapesAreEqual());
  EXPECT_EQ(
      Prepare(this->op_, this->input_tensor_, this->output_tensor_),
      absl::FailedPreconditionError("The specified output tensor shape is not "
                                    "compatible with the input shape."));
}

REGISTER_TYPED_TEST_SUITE_P(UnaryElementwiseOpShapePropagationTest,
                            ShapePropagationWorks,
                            SmallerOutputShapeRaisesAnError,
                            BiggerOutputShapeRaisesAnError,
                            IncompatibleOutputShapeRaisesAnError);

// Tests that the baseline element type of the input and output tensors is the
// same.
template <class T>
class UnaryElementwiseSameBaselineElementTypeConstraintTest
    : public ::testing::Test {};

TYPED_TEST_SUITE_P(UnaryElementwiseSameBaselineElementTypeConstraintTest);

TYPED_TEST_P(UnaryElementwiseSameBaselineElementTypeConstraintTest,
             DifferentInputOutputStorageTypesRaiseAnError) {
  using Op = std::tuple_element_t<0, TypeParam>;
  using OperandTypeDesc = std::tuple_element_t<1, TypeParam>;
  using ResultTypeDesc = std::tuple_element_t<2, TypeParam>;
  const Shape shape({2, 3, 4});
  Tensor input_tensor{.type = TensorTypeFor(OperandTypeDesc{}, shape),
                      .data = nullptr};
  Tensor output_tensor{.type = TensorTypeFor(ResultTypeDesc{}, shape),
                       .data = nullptr};
  auto op = Create(typename Op::Attributes{});
  const absl::Status status = Prepare(op, input_tensor, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      status.message(),
      ::testing::ContainsRegex(
          "stablehlo.[_a-z]+: baseline type constraint is not satisfied"));
}

REGISTER_TYPED_TEST_SUITE_P(
    UnaryElementwiseSameBaselineElementTypeConstraintTest,
    DifferentInputOutputStorageTypesRaiseAnError);

// Tests that unsupported types are detected during when `Prepare` is called.
template <class T>
class UnaryElementwiseUnsupportedTypeTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(UnaryElementwiseUnsupportedTypeTest);

TYPED_TEST_P(UnaryElementwiseUnsupportedTypeTest, PrepareRaisesAnError) {
  using Op = std::tuple_element_t<0, TypeParam>;
  using TypeDesc = std::tuple_element_t<1, TypeParam>;
  Tensor input_tensor{.type = TensorTypeFor(TypeDesc{}, Shape({2, 3, 4})),
                      .data = nullptr};
  Tensor output_tensor = input_tensor;
  auto op = Create(typename Op::Attributes{});
  const absl::Status status = Prepare(op, input_tensor, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Unsupported tensor type"));
}

REGISTER_TYPED_TEST_SUITE_P(UnaryElementwiseUnsupportedTypeTest,
                            PrepareRaisesAnError);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UNARY_ELEMENTWISE_TEST_UTIL_H_
