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

#include "tensorflow/lite/experimental/shlo/ops/convolution.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
using testing::Eq;
using testing::FloatEq;
using testing::Pointwise;

namespace shlo_ref {
namespace {
using kBF16TestTypes = ::testing::Types<TestParam<DataType::kBF16>>;
using kF16TestTypes = ::testing::Types<TestParam<DataType::kF16>>;
using kF32TestTypes = ::testing::Types<TestParam<DataType::kF32>>;

template <class T>
struct NonQuantizedkF32ConvolutionTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF32ConvolutionTest, kF32TestTypes,
                 TestParamNames);
TYPED_TEST(NonQuantizedkF32ConvolutionTest, kF32TestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({1, 1, 2, 2});
  const Shape shape_rhs({1, 1, 1, 1});
  const Shape shape_padding({2, 2});
  const Shape shape_parametrs({2});
  const Shape shape_result({1, 1, 2, 2});

  Vector<float> lhs_data_float{1.16, 2.43, 3.81, 4.77};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{2.21};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_result.NumElements());
  Vector<int64_t> window_stride_values({1, 1});
  Vector<int64_t> padding_values({0, 0, 0, 0});
  Vector<int64_t> lhs_dilation_values({1, 1});
  Vector<int64_t> rhs_dilation_values({1, 1});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> input_spatial_dimensions_values({2, 3});
  int64_t kernel_input_feature_dimension = 1;
  int64_t kernel_output_feature_dimension = 0;
  Vector<int64_t> kernel_spatial_dimensions_values({2, 3});
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 1;
  Vector<int64_t> output_spatial_dimensions_values({2, 3});
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  absl::Span<int64_t> window_stride(window_stride_values);
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  absl::Span<int64_t> lhs_dilation(lhs_dilation_values);
  absl::Span<int64_t> rhs_dilation(rhs_dilation_values);
  absl::Span<int64_t> input_spatial_dimensions(input_spatial_dimensions_values);
  absl::Span<int64_t> kernel_spatial_dimensions(
      kernel_spatial_dimensions_values);
  absl::Span<int64_t> output_spatial_dimensions(
      output_spatial_dimensions_values);
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spatial_dimensions = input_spatial_dimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spatial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spatial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data;
  Vector<float> expected_data_float = {2.5636, 5.3703, 8.4201, 10.5417};
  expected_data.assign(expected_data_float.begin(), expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedkBF16ConvolutionTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkBF16ConvolutionTest, kBF16TestTypes,
                 TestParamNames);
TYPED_TEST(NonQuantizedkBF16ConvolutionTest, kBF16TestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({1, 1, 2, 2});
  const Shape shape_rhs({1, 1, 1, 1});
  const Shape shape_padding({2, 2});
  const Shape shape_parametrs({2});
  const Shape shape_result({1, 1, 2, 2});

  Vector<float> lhs_data_float{1.16, 2.43, 3.81, 4.77};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{2.21};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_result.NumElements());
  Vector<int64_t> window_stride_values({1, 1});
  Vector<int64_t> padding_values({0, 0, 0, 0});
  Vector<int64_t> lhs_dilation_values({1, 1});
  Vector<int64_t> rhs_dilation_values({1, 1});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> input_spatial_dimensions_values({2, 3});
  int64_t kernel_input_feature_dimension = 1;
  int64_t kernel_output_feature_dimension = 0;
  Vector<int64_t> kernel_spatial_dimensions_values({2, 3});
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 1;
  Vector<int64_t> output_spatial_dimensions_values({2, 3});
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  absl::Span<int64_t> window_stride(window_stride_values);
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  absl::Span<int64_t> lhs_dilation(lhs_dilation_values);
  absl::Span<int64_t> rhs_dilation(rhs_dilation_values);
  absl::Span<int64_t> input_spatial_dimensions(input_spatial_dimensions_values);
  absl::Span<int64_t> kernel_spatial_dimensions(
      kernel_spatial_dimensions_values);
  absl::Span<int64_t> output_spatial_dimensions(
      output_spatial_dimensions_values);
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spatial_dimensions = input_spatial_dimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spatial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spatial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data;
  Vector<float> expected_data_float = {2.54688, 5.375, 8.375, 10.5625};
  expected_data.assign(expected_data_float.begin(), expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedkF16ConvolutionTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF16ConvolutionTest, kF16TestTypes,
                 TestParamNames);
TYPED_TEST(NonQuantizedkF16ConvolutionTest, kF16TestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({1, 1, 2, 2});
  const Shape shape_rhs({1, 1, 1, 1});
  const Shape shape_padding({2, 2});
  const Shape shape_parametrs({2});
  const Shape shape_result({1, 1, 2, 2});

  Vector<float> lhs_data_float{1.16, 2.43, 3.81, 4.77};
  Vector<StorageT> lhs_data(lhs_data_float.begin(), lhs_data_float.end());
  Vector<float> rhs_data_float{2.21};
  Vector<StorageT> rhs_data(rhs_data_float.begin(), rhs_data_float.end());
  Vector<StorageT> output_data(shape_result.NumElements());
  Vector<int64_t> window_stride_values({1, 1});
  Vector<int64_t> padding_values({0, 0, 0, 0});
  Vector<int64_t> lhs_dilation_values({1, 1});
  Vector<int64_t> rhs_dilation_values({1, 1});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> input_spatial_dimensions_values({2, 3});
  int64_t kernel_input_feature_dimension = 1;
  int64_t kernel_output_feature_dimension = 0;
  Vector<int64_t> kernel_spatial_dimensions_values({2, 3});
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 1;
  Vector<int64_t> output_spatial_dimensions_values({2, 3});
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  absl::Span<int64_t> window_stride(window_stride_values);
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  absl::Span<int64_t> lhs_dilation(lhs_dilation_values);
  absl::Span<int64_t> rhs_dilation(rhs_dilation_values);
  absl::Span<int64_t> input_spatial_dimensions(input_spatial_dimensions_values);
  absl::Span<int64_t> kernel_spatial_dimensions(
      kernel_spatial_dimensions_values);
  absl::Span<int64_t> output_spatial_dimensions(
      output_spatial_dimensions_values);
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spatial_dimensions = input_spatial_dimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spatial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spatial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data;
  Vector<float> expected_data_float = {2.56445, 5.37109, 8.42188, 10.5469};
  expected_data.assign(expected_data_float.begin(), expected_data_float.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedIntConvolutionTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedIntConvolutionTest, IntTestTypes, TestParamNames);
TYPED_TEST(NonQuantizedIntConvolutionTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({1, 1, 10});
  const Shape shape_rhs({1, 1, 1});
  const Shape shape_padding({1, 2});
  const Shape shape_parametrs({1});
  const Shape shape_result({1, 1, 10});

  Vector<int64_t> lhs_data_int{1, 2, 3, 4, 5, 6, 7, 9, 4, 2};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{5};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_result.NumElements());
  Vector<int64_t> window_stride_values({1});
  Vector<int64_t> padding_values({0, 0});
  Vector<int64_t> lhs_dilation_values({1});
  Vector<int64_t> rhs_dilation_values({1});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> input_spatial_dimensions_values({2});
  int64_t kernel_input_feature_dimension = 1;
  int64_t kernel_output_feature_dimension = 0;
  Vector<int64_t> kernel_spatial_dimensions_values({2});
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 1;
  Vector<int64_t> output_spatial_dimensions_values({2});
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  absl::Span<int64_t> window_stride(window_stride_values);
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  absl::Span<int64_t> lhs_dilation(lhs_dilation_values);
  absl::Span<int64_t> rhs_dilation(rhs_dilation_values);
  absl::Span<int64_t> input_spatial_dimensions(input_spatial_dimensions_values);
  absl::Span<int64_t> kernel_spatial_dimensions(
      kernel_spatial_dimensions_values);
  absl::Span<int64_t> output_spatial_dimensions(
      output_spatial_dimensions_values);
  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  std::array<PrecisionTypes, 2> precision_configs = {PrecisionTypes::DEFAULT,
                                                     PrecisionTypes::DEFAULT};

  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spatial_dimensions = input_spatial_dimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spatial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spatial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<int64_t> expected_data_int{5, 10, 15, 20, 25, 30, 35, 45, 20, 10};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref