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

#include "tensorflow/lite/experimental/shlo/ops/exponential_minus_one.h"

#include <cmath>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise_test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::ElementsAreArray;
using testing::NanSensitiveFloatEq;
using testing::Pointwise;

namespace shlo_ref {

template <>
struct ParamName<ExponentialMinusOneOp> {
  static std::string Get() { return "ExponentialMinusOne"; }
};

namespace {

struct ExponentialMinusOne {
  template <class T>
  T operator()(T v) const {
    return std::expm1(v);
  }
} exponential_minus_one_ref;

template <>
F16 ExponentialMinusOne::operator()(F16 v) const {
  return F16(operator()(static_cast<float>(v)));
}

template <>
BF16 ExponentialMinusOne::operator()(BF16 v) const {
  return BF16(operator()(static_cast<float>(v)));
}

INSTANTIATE_TYPED_TEST_SUITE_P(ExponentialMinusOne,
                               UnaryElementwiseOpShapePropagationTest,
                               ExponentialMinusOneOp, TestParamNames);

INSTANTIATE_TYPED_TEST_SUITE_P(
    ExponentialMinusOne, UnaryElementwiseSameBaselineElementTypeConstraintTest,
    UnaryElementwiseConstraint1Types<ExponentialMinusOneOp>, TestParamNames);

using UnsupportedTypes =
    WithOpTypes<ExponentialMinusOneOp, ConcatTypes<BoolTestType, IntTestTypes,
                                                   PerAxisQuantizedTestTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(ExponentialMinusOneOp,
                               UnaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

template <class T>
struct ExponentialMinusOneTest : ::testing::Test {};

TYPED_TEST_SUITE(ExponentialMinusOneTest, FloatTestTypes, TestParamNames);

TYPED_TEST(ExponentialMinusOneTest, FloatTensorsWork) {
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
  absl::c_transform(input_data, expected_data.begin(),
                    exponential_minus_one_ref);

  auto op = Create(ExponentialMinusOneOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(NanSensitiveFloatEq(), expected_data));
}

template <class T>
struct QuantizedExponentialMinusOneTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedExponentialMinusOneTest, QuantizedTestTypes,
                 TestParamNames);

TYPED_TEST(QuantizedExponentialMinusOneTest, PerTensorWorks) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(5);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  Tensor input_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      input_data, expected_data.begin(), [zero_point, scale](auto v) {
        const ExpressedT dequantized_input = Dequantize(v, zero_point, scale);
        const ExpressedT dequantized_res =
            exponential_minus_one_ref(dequantized_input);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, zero_point, static_cast<ExpressedT>(1.) / scale);
      });

  auto op = Create(ExponentialMinusOneOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

}  // namespace
}  // namespace shlo_ref
