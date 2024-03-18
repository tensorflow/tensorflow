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
#include "tensorflow/lite/experimental/shlo/ops/abs.h"

#include <cstddef>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using testing::ElementsAreArray;

namespace shlo_ref {

namespace {

constexpr struct AbsRef {
  template <class T>
  T operator()(T v) const {
    return v < 0 ? -v : v;
  }
} abs_ref;

template <class T>
struct AbsTest : ::testing::Test {};

TYPED_TEST_SUITE(AbsTest, NonQuantizedTestTypes, TestParamNames);

TYPED_TEST(AbsTest, NonQuantized) {
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
  absl::c_transform(input_data, expected_data.begin(), abs_ref);

  auto op = Create(AbsOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

template <class T>
struct QuantizedAbsTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedAbsTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedAbsTest, QuantizedPerTensor) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(5);
  const QuantizedTensorElementType tensor_type =
      QuantizedTensorElementType::PerTensor<TypeParam::kStorage,
                                            TypeParam::kExpressed>(scale,
                                                                   zero_point);
  Tensor input_tensor{
      .type = QuantizedTensorType{.shape = shape, .element_type = tensor_type},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = QuantizedTensorType{.shape = shape, .element_type = tensor_type},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      input_data, expected_data.begin(), [zero_point, scale](auto v) {
        const ExpressedT dequantized_input = Dequantize(v, zero_point, scale);
        const ExpressedT dequantized_res = abs_ref(dequantized_input);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, zero_point, static_cast<ExpressedT>(1.) / scale);
      });

  auto op = Create(AbsOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

TYPED_TEST(QuantizedAbsTest, QuantizedPerAxis) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({4, 3, 2});
  const int quantized_dimension = 2;
  const size_t rank = shape.Rank();
  const Axis quantized_dimension_size = shape.Dim(quantized_dimension);
  const size_t quantization_stride = [&] {
    size_t res = 1;
    for (int64_t i = rank - 1; i > quantized_dimension; --i) {
      res *= shape.Dim(i);
    }
    return res;
  }();
  Vector<StorageT> input_data = IotaBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  Vector<StorageT> zero_points_data = RandomBuffer<TypeParam::kStorage>(
      /*shape=*/Shape({shape.Dim(2)}), /*min=*/static_cast<StorageT>(-5),
      /*max=*/static_cast<StorageT>(5));
  Vector<ExpressedT> scales_data = RandomBuffer<TypeParam::kExpressed>(
      /*shape=*/Shape({shape.Dim(2)}), /*min=*/static_cast<ExpressedT>(1),
      /*max=*/static_cast<ExpressedT>(3));
  const QuantizedTensorElementType tensor_type =
      QuantizedTensorElementType::PerAxis<TypeParam::kStorage,
                                          TypeParam::kExpressed>(
          scales_data, zero_points_data, quantized_dimension);
  Tensor input_tensor{
      .type = QuantizedTensorType{.shape = shape, .element_type = tensor_type},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = QuantizedTensorType{.shape = shape, .element_type = tensor_type},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(
      input_data, expected_data.begin(),
      [&, element_index = 0ull, quantization_index = 0ull](auto v) mutable {
        const StorageT zero_point = zero_points_data[quantization_index];
        const ExpressedT scale = scales_data[quantization_index];

        if (++element_index >= quantization_stride) {
          element_index = 0;
          if (++quantization_index >= quantized_dimension_size) {
            quantization_index = 0;
          }
        }
        const ExpressedT dequantized_input = Dequantize(v, zero_point, scale);
        const ExpressedT dequantized_res = abs_ref(dequantized_input);
        return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
            dequantized_res, zero_point, ExpressedT(1) / scale);
      });

  auto op = Create(AbsOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, ElementsAreArray(expected_data));
}

}  // namespace
}  // namespace shlo_ref
