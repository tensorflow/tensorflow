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

#include "tensorflow/lite/experimental/shlo/ops/not.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise_test_util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::NanSensitiveFloatEq;
using testing::Pointwise;

namespace shlo_ref {

template <>
struct ParamName<NotOp> {
  static std::string Get() { return "Not"; }
};

template <>
struct SupportedOpDataType<NotOp> {
  static constexpr DataType kStorageType = DataType::kSI32;
};

namespace {

struct Not {
  template <class T>
  T operator()(T v) const {
    return ~v;
  }
} not_ref;

template <>
bool Not::operator()(bool v) const {
  return !v;
}

INSTANTIATE_TYPED_TEST_SUITE_P(Not, UnaryElementwiseOpShapePropagationTest,
                               NotOp, TestParamNames);

INSTANTIATE_TYPED_TEST_SUITE_P(
    Not, UnaryElementwiseSameBaselineElementTypeConstraintTest,
    BaselineMismatchSignedIntegerTypes<NotOp>, TestParamNames);

using UnsupportedTypes =
    WithOpTypes<NotOp, ConcatTypes<FloatTestTypes, PerTensorQuantizedTestTypes,
                                   PerAxisQuantizedTestTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Not, UnaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

template <class T>
struct BoolAndIntNotTest : ::testing::Test {};

using SupportedTypes = ConcatTypes<BoolTestType, IntTestTypes>;

TYPED_TEST_SUITE(BoolAndIntNotTest, SupportedTypes, TestParamNames);

TYPED_TEST(BoolAndIntNotTest, BoolAndIntTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());

  Tensor input_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(input_data, expected_data.begin(), not_ref);

  auto op = Create(NotOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(NanSensitiveFloatEq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref
