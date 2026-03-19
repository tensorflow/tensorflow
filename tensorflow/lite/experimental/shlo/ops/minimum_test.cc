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

#include "tensorflow/lite/experimental/shlo/ops/minimum.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
struct ParamName<MinimumOp> {
  static std::string Get() { return "Minimum"; }
};

struct Minimum {
  template <class T>
  constexpr auto operator()(const T a, const T b) {
    return a < b ? a : b;
  }
};

namespace {

INSTANTIATE_TYPED_TEST_SUITE_P(Minimum, BinaryElementwiseOpShapePropagationTest,
                               MinimumOp, TestParamNames);

using MinimumBaselineContraintTypes = BinaryElementwiseBaselineConstraintTypes<
    MinimumOp, ConcatTypes<BoolTestType, BaselineConstraintIntTypes,
                           BaselineConstraintFloatTypes,
                           BaselineConstraintQuantizedPerTensorTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    Minimum, BinaryElementwiseSameBaselineElementTypeConstraintTest,
    MinimumBaselineContraintTypes, TestParamNames);

using UnsupportedTypes =
    WithOpTypes<MinimumOp, ConcatTypes<PerAxisQuantizedTestTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Minimum, BinaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

using SupportedTypes = ConcatTypes<BoolTestType, ArithmeticTestTypes>;

template <class T>
struct MinimumTest : ::testing::Test {};

TYPED_TEST_SUITE(MinimumTest, SupportedTypes, TestParamNames);

TYPED_TEST(MinimumTest, ArithmeticTestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

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
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(lhs_data, rhs_data, expected_data.begin(), Minimum());

  auto op = Create(MinimumOp::Attributes{});
  ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct QuantizedMinimumTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedMinimumTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedMinimumTest, PerTensorWorks) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(2);
  Vector<StorageT> lhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-50, /*max=*/50);
  Vector<StorageT> rhs_data = RandomBuffer<TypeParam::kStorage>(
      shape, /*min=*/zero_point + 1, /*max=*/zero_point + 5);
  Vector<StorageT> output_data(shape.NumElements());
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
      .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      lhs_data, rhs_data, expected_data.begin(),
      [zero_point, scale](auto lhs, auto rhs) {
        const ExpressedT dequantized_lhs = Dequantize(lhs, zero_point, scale);
        const ExpressedT dequantized_rhs = Dequantize(rhs, zero_point, scale);
        const ExpressedT dequantized_res =
            Minimum()(dequantized_lhs, dequantized_rhs);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, zero_point, static_cast<ExpressedT>(1.) / scale);
      });

  auto op = Create(MinimumOp::Attributes{});
  ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}
}  // namespace
}  // namespace shlo_ref
