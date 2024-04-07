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

#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::ElementsAreArray;

namespace shlo_ref {
namespace {

struct TestOp {
  template <typename T>
  T operator()(const T& lhs, const T& rhs) {
    return lhs + rhs;
  }
};

template <class T>
struct EvaluateNoQuantizationTest : ::testing::Test {};

TYPED_TEST_SUITE(EvaluateNoQuantizationTest, ArithmeticTestTypes,
                 TestParamNames);

TYPED_TEST(EvaluateNoQuantizationTest, ArithmeticTensorsWithTestOp) {
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
  absl::c_transform(lhs_data, rhs_data, expected_data.begin(), TestOp());

  detail::EvaluateNoQuantization<TypeParam::kStorage>(
      TestOp(), lhs_tensor, rhs_tensor, output_tensor);
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

template <class T>
struct DequantizeOpQuantizePerTensor : ::testing::Test {};

TYPED_TEST_SUITE(DequantizeOpQuantizePerTensor, QuantizedTestTypes,
                 TestParamNames);

TYPED_TEST(DequantizeOpQuantizePerTensor, QuantizedPerTensorWithTestOp) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> lhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
  Vector<StorageT> rhs_data =
      RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
  Vector<StorageT> output_data(shape.NumElements());
  const ExpressedT lhs_scale = static_cast<ExpressedT>(1.3);
  const StorageT lhs_zero_point = static_cast<StorageT>(4);
  const ExpressedT rhs_scale = static_cast<ExpressedT>(1.2);
  const StorageT rhs_zero_point = static_cast<StorageT>(5);
  const ExpressedT output_scale = static_cast<ExpressedT>(1.5);
  const StorageT output_zero_point = static_cast<StorageT>(3);
  Tensor lhs_tensor{.type =
                        QuantizedPerTensorTensorType{
                            .shape = shape,
                            .element_type = QuantizedElementTypePerTensor(
                                TypeParam::kStorage, lhs_zero_point,
                                TypeParam::kExpressed, lhs_scale)},
                    .data = lhs_data.data()};
  Tensor rhs_tensor{.type =
                        QuantizedPerTensorTensorType{
                            .shape = shape,
                            .element_type = QuantizedElementTypePerTensor(
                                TypeParam::kStorage, rhs_zero_point,
                                TypeParam::kExpressed, rhs_scale)},
                    .data = rhs_data.data()};
  Tensor output_tensor{.type =
                           QuantizedPerTensorTensorType{
                               .shape = shape,
                               .element_type = QuantizedElementTypePerTensor(
                                   TypeParam::kStorage, output_zero_point,
                                   TypeParam::kExpressed, output_scale)},
                       .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      lhs_data, rhs_data, expected_data.begin(),
      [lhs_zero_point, lhs_scale, rhs_zero_point, rhs_scale, output_zero_point,
       output_scale](auto lhs, auto rhs) {
        const ExpressedT dequantized_lhs =
            Dequantize(lhs, lhs_zero_point, lhs_scale);
        const ExpressedT dequantized_rhs =
            Dequantize(rhs, rhs_zero_point, rhs_scale);
        const ExpressedT dequantized_res =
            TestOp()(dequantized_lhs, dequantized_rhs);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, output_zero_point,
            static_cast<ExpressedT>(1.) / output_scale);
      });

  detail::DequantizeOpQuantizePerTensor<TypeParam::kStorage,
                                        TypeParam::kExpressed>(
      TestOp(), lhs_tensor, rhs_tensor, output_tensor);
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

}  // namespace
}  // namespace shlo_ref
