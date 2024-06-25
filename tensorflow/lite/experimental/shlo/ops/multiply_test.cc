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

#include "tensorflow/lite/experimental/shlo/ops/multiply.h"

#include <functional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
struct ParamName<MultiplyOp> {
  static std::string Get() { return "Multiply"; }
};

template <DataType expressed_type>
struct Multiply : std::multiplies<void> {};

template <>
struct Multiply<DataType::kI1> {
  template <class T>
  T operator()(const T& lhs, const T& rhs) const {
    return static_cast<T>(lhs && rhs);
  }
};

namespace {

INSTANTIATE_TYPED_TEST_SUITE_P(Multiply,
                               BinaryElementwiseOpShapePropagationTest,
                               MultiplyOp, TestParamNames);

using MultipyBaselineContraintTypes = BinaryElementwiseBaselineConstraintTypes<
    MultiplyOp, ConcatTypes<BoolTestType, BaselineConstraintIntTypes,
                            BaselineConstraintFloatTypes,
                            BaselineConstraintQuantizedPerTensorTypes>>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    Multiply, BinaryElementwiseSameBaselineElementTypeConstraintTest,
    MultipyBaselineContraintTypes, TestParamNames);

using UnsupportedTypes = WithOpTypes<MultiplyOp, PerAxisQuantizedTestTypes>;

INSTANTIATE_TYPED_TEST_SUITE_P(Multiply, BinaryElementwiseUnsupportedTypeTest,
                               UnsupportedTypes, TestParamNames);

using ArithmeticTypes = ConcatTypes<BoolTestType, ArithmeticTestTypes>;

template <class T>
struct MultiplyTest : ::testing::Test {};

TYPED_TEST_SUITE(MultiplyTest, ArithmeticTypes, TestParamNames);

TYPED_TEST(MultiplyTest, ArithmeticTestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> lhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
  Vector<StorageT> rhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
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
  absl::c_transform(lhs_data, rhs_data, expected_data.begin(),
                    Multiply<TypeParam::kStorage>());

  auto op = Create(MultiplyOp::Attributes{});
  ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct QuantizedMultiplyTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedMultiplyTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedMultiplyTest, PerTensorWorks) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> lhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
  Vector<StorageT> rhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
  Vector<StorageT> output_data(shape.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(5);
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
            Multiply<TypeParam::kExpressed>()(dequantized_lhs, dequantized_rhs);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, zero_point, static_cast<ExpressedT>(1.) / scale);
      });

  auto op = Create(MultiplyOp::Attributes{});
  ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}
}  // namespace
}  // namespace shlo_ref
