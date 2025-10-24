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
using testing::FloatEq;
using testing::Pointwise;
namespace shlo_ref {

namespace {
template <class T>
struct NonQuantizedFloatConvolutionTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedFloatConvolutionTest, FloatTestTypes,
                 TestParamNames);

TYPED_TEST(NonQuantizedFloatConvolutionTest, FloatTestTypesTensorsWork) {
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
  Vector<bool> window_reversal_values({false, false});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> inputSpatialDimensions_values({2, 3});
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
  Tensor window_stride{.type = TensorType{.shape = shape_parametrs,
                                          .element_type = DataType::kSI64},
                       .data = window_stride_values.data()};
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  Tensor lhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = lhs_dilation_values.data()};
  Tensor rhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = rhs_dilation_values.data()};
  Tensor window_reversal{.type = TensorType{.shape = shape_parametrs,
                                            .element_type = DataType::kI1},
                         .data = window_reversal_values.data()};
  Tensor inputSpatialDimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = inputSpatialDimensions_values.data()};
  Tensor kernel_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = kernel_spatial_dimensions_values.data()};
  Tensor output_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = output_spatial_dimensions_values.data()};

  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  absl::InlinedVector<PrecisionTypes, 2> precision_configs = {
      PrecisionTypes::DEFAULT, PrecisionTypes::DEFAULT};
  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .window_reversal = window_reversal,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spacial_dimensions = inputSpatialDimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spacial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spacial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data;
  if (std::is_same<StorageT, BF16>::value) {
    Vector<float> expected_data_float = {2.54688, 5.375, 8.375, 10.5625};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else if (std::is_same<StorageT, F16>::value) {
    Vector<float> expected_data_float = {2.56445, 5.37109, 8.42188, 10.5469};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  } else {
    Vector<float> expected_data_float = {2.5636, 5.3703, 8.4201, 10.5417};
    expected_data.assign(expected_data_float.begin(),
                         expected_data_float.end());
  }

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedIntConvolutionTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedIntConvolutionTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntConvolutionTest, IntTestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_lhs({1, 4, 4, 1});
  const Shape shape_rhs({4, 2, 1, 1});
  const Shape shape_padding({2, 2});
  const Shape shape_parametrs({2});
  const Shape shape_result({1, 4, 2, 1});

  Vector<int64_t> lhs_data_int{1, 3, 10, 12, 2, 4, 11, 13,
                               5, 7, 14, 16, 6, 8, 15, 17};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<int64_t> rhs_data_int{1, 1, 1, 1, 1, 1, 1, 1};
  Vector<StorageT> rhs_data(rhs_data_int.begin(), rhs_data_int.end());
  Vector<StorageT> output_data(shape_result.NumElements());
  Vector<int64_t> window_stride_values({4, 4});
  Vector<int64_t> padding_values({0, 0, 0, 0});
  Vector<int64_t> lhs_dilation_values({2, 2});
  Vector<int64_t> rhs_dilation_values({1, 1});
  Vector<bool> window_reversal_values({false, false});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> inputSpatialDimensions_values({2, 3});
  int64_t kernel_input_feature_dimension = 1;
  int64_t kernel_output_feature_dimension = 0;
  Vector<int64_t> kernel_spatial_dimensions_values({2, 3});
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 1;
  Vector<int64_t> output_spatial_dimensions_values({2, 3});
  int64_t feature_group_count = 2;
  int64_t batch_group_count = 1;

  Tensor lhs{.type = TensorType{.shape = shape_lhs,
                                .element_type = TypeParam::kStorage},
             .data = lhs_data.data()};
  Tensor rhs{.type = TensorType{.shape = shape_rhs,
                                .element_type = TypeParam::kStorage},
             .data = rhs_data.data()};
  Tensor window_stride{.type = TensorType{.shape = shape_parametrs,
                                          .element_type = DataType::kSI64},
                       .data = window_stride_values.data()};
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  Tensor lhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = lhs_dilation_values.data()};
  Tensor rhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = rhs_dilation_values.data()};
  Tensor window_reversal{.type = TensorType{.shape = shape_parametrs,
                                            .element_type = DataType::kI1},
                         .data = window_reversal_values.data()};
  Tensor inputSpatialDimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = inputSpatialDimensions_values.data()};
  Tensor kernel_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = kernel_spatial_dimensions_values.data()};
  Tensor output_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = output_spatial_dimensions_values.data()};

  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  absl::InlinedVector<PrecisionTypes, 2> precision_configs = {
      PrecisionTypes::DEFAULT, PrecisionTypes::DEFAULT};
  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .window_reversal = window_reversal,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spacial_dimensions = inputSpatialDimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spacial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spacial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<int64_t> expected_data_int{3, 21, 3, 21, 11, 29, 11, 29};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

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
  Vector<bool> window_reversal_values({false});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> inputSpatialDimensions_values({2});
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
  Tensor window_stride{.type = TensorType{.shape = shape_parametrs,
                                          .element_type = DataType::kSI64},
                       .data = window_stride_values.data()};
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  Tensor lhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = lhs_dilation_values.data()};
  Tensor rhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = rhs_dilation_values.data()};
  Tensor window_reversal{.type = TensorType{.shape = shape_parametrs,
                                            .element_type = DataType::kI1},
                         .data = window_reversal_values.data()};
  Tensor inputSpatialDimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = inputSpatialDimensions_values.data()};
  Tensor kernel_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = kernel_spatial_dimensions_values.data()};
  Tensor output_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = output_spatial_dimensions_values.data()};

  Tensor output_tensor{.type = TensorType{.shape = shape_result,
                                          .element_type = TypeParam::kStorage},
                       .data = output_data.data()};

  absl::InlinedVector<PrecisionTypes, 2> precision_configs = {
      PrecisionTypes::DEFAULT, PrecisionTypes::DEFAULT};
  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .window_reversal = window_reversal,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spacial_dimensions = inputSpatialDimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spacial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spacial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<int64_t> expected_data_int{5, 10, 15, 20, 25, 30, 35, 45, 20, 10};
  Vector<StorageT> expected_data(expected_data_int.begin(),
                                 expected_data_int.end());

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct QuantizedIntConvolutionTest : ::testing::Test {};
TYPED_TEST_SUITE(QuantizedIntConvolutionTest, QuantizedTestTypes,
                 TestParamNames);

TYPED_TEST(QuantizedIntConvolutionTest, PerTensorsRaiseAnError) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({1, 1, 10});
  const Shape shape_rhs({1, 1, 1});
  const Shape shape_padding({1, 2});
  const Shape shape_parametrs({1});
  const Shape shape_result({1, 1, 10});

  Vector<int64_t> lhs_data_int{1, 2, 3, 1, 2, 3, 1, 2, 3, 2};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());
  Vector<StorageT> rhs_data = Vector<StorageT>{1};

  Vector<StorageT> output_data(shape_result.NumElements());
  Vector<int64_t> window_stride_values({1});
  Vector<int64_t> padding_values({0, 0});
  Vector<int64_t> lhs_dilation_values({1});
  Vector<int64_t> rhs_dilation_values({1});
  Vector<bool> window_reversal_values({false});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> inputSpatialDimensions_values({2});
  int64_t kernel_input_feature_dimension = 1;
  int64_t kernel_output_feature_dimension = 0;
  Vector<int64_t> kernel_spatial_dimensions_values({2});
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 1;
  Vector<int64_t> output_spatial_dimensions_values({2});
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;

  const ExpressedT scale = static_cast<ExpressedT>(1);
  const StorageT zero_point = static_cast<StorageT>(0);

  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{.type = QuantizedPerTensorTensorType{.shape = shape_rhs,
                                                  .element_type = tensor_type},
             .data = rhs_data.data()};
  Tensor window_stride{.type = TensorType{.shape = shape_parametrs,
                                          .element_type = DataType::kSI64},
                       .data = window_stride_values.data()};
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  Tensor lhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = lhs_dilation_values.data()};
  Tensor rhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = rhs_dilation_values.data()};
  Tensor window_reversal{.type = TensorType{.shape = shape_parametrs,
                                            .element_type = DataType::kI1},
                         .data = window_reversal_values.data()};
  Tensor inputSpatialDimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = inputSpatialDimensions_values.data()};
  Tensor kernel_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = kernel_spatial_dimensions_values.data()};
  Tensor output_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = output_spatial_dimensions_values.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_result,
                                           .element_type = tensor_type},
      .data = output_data.data()};

  absl::InlinedVector<PrecisionTypes, 2> precision_configs = {
      PrecisionTypes::DEFAULT, PrecisionTypes::DEFAULT};
  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .window_reversal = window_reversal,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spacial_dimensions = inputSpatialDimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spacial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spacial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});
  Vector<StorageT> expected_data =
      Vector<StorageT>{1, 2, 3, 1, 2, 3, 1, 2, 3, 2};

  Vector<StorageT> expected_data_quantized(shape_result.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_data_quantized.begin(), [&](StorageT val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1.0) / scale);
                 });

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));

  constexpr double kEpsilon = 0.1;
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(QuantizedIntConvolutionTest, PerAxisRaiseAnError) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape_lhs({1, 1, 10});
  const Shape shape_rhs({1, 1, 1});
  const Shape shape_padding({1, 2});
  const Shape shape_parametrs({1});
  const Shape shape_result({1, 1, 10});

  Vector<int64_t> lhs_data_int{1, 2, 3, 4, 5, 1, 2, 3, 4, -5};
  Vector<StorageT> lhs_data(lhs_data_int.begin(), lhs_data_int.end());

  Vector<StorageT> rhs_data = Vector<StorageT>{5};

  Vector<StorageT> output_data(shape_result.NumElements());
  Vector<int64_t> window_stride_values({1});
  Vector<int64_t> padding_values({0, 0});
  Vector<int64_t> lhs_dilation_values({1});
  Vector<int64_t> rhs_dilation_values({1});
  Vector<bool> window_reversal_values({false});
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 1;
  Vector<int64_t> inputSpatialDimensions_values({2});
  int64_t kernel_input_feature_dimension = 1;
  int64_t kernel_output_feature_dimension = 0;
  Vector<int64_t> kernel_spatial_dimensions_values({2});
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 1;
  Vector<int64_t> output_spatial_dimensions_values({2});
  int64_t feature_group_count = 1;
  int64_t batch_group_count = 1;

  std::initializer_list<float> zero_points = {0, 0, 0};
  std::initializer_list<float> scales = {1.7, 1.6, 1.5};

  const ExpressedT scale = static_cast<ExpressedT>(1);
  const StorageT zero_point = static_cast<StorageT>(0);

  QuantizedElementTypePerTensor tensor_type = QuantizedElementTypePerTensor(
      TypeParam::kStorage, zero_point, TypeParam::kExpressed, scale);

  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);

  QuantizedElementTypePerAxis tensor_type_axis_res(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 1);

  Tensor lhs{.type = QuantizedPerTensorTensorType{.shape = shape_lhs,
                                                  .element_type = tensor_type},
             .data = lhs_data.data()};
  Tensor rhs{
      .type = QuantizedPerAxisTensorType{.shape = shape_rhs,
                                         .element_type = tensor_type_axis},
      .data = rhs_data.data()};
  Tensor window_stride{.type = TensorType{.shape = shape_parametrs,
                                          .element_type = DataType::kSI64},
                       .data = window_stride_values.data()};
  Tensor padding{.type = TensorType{.shape = shape_padding,
                                    .element_type = DataType::kSI64},
                 .data = padding_values.data()};
  Tensor lhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = lhs_dilation_values.data()};
  Tensor rhs_dilation{.type = TensorType{.shape = shape_parametrs,
                                         .element_type = DataType::kSI64},
                      .data = rhs_dilation_values.data()};
  Tensor window_reversal{.type = TensorType{.shape = shape_parametrs,
                                            .element_type = DataType::kI1},
                         .data = window_reversal_values.data()};
  Tensor inputSpatialDimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = inputSpatialDimensions_values.data()};
  Tensor kernel_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = kernel_spatial_dimensions_values.data()};
  Tensor output_spatial_dimensions{
      .type =
          TensorType{.shape = shape_parametrs, .element_type = DataType::kSI64},
      .data = output_spatial_dimensions_values.data()};
  Tensor output_tensor{
      .type = QuantizedPerAxisTensorType{.shape = shape_result,
                                         .element_type = tensor_type_axis_res},
      .data = output_data.data()};

  absl::InlinedVector<PrecisionTypes, 2> precision_configs = {
      PrecisionTypes::DEFAULT, PrecisionTypes::DEFAULT};
  auto op = Create(ConvolutionOp::Attributes{
      .window_strides = window_stride,
      .padding = padding,
      .lhs_dilation = lhs_dilation,
      .rhs_dilation = rhs_dilation,
      .window_reversal = window_reversal,
      .input_batch_dimension = input_batch_dimension,
      .input_feature_dimension = input_feature_dimension,
      .input_spacial_dimensions = inputSpatialDimensions,
      .kernel_input_feature_dimension = kernel_input_feature_dimension,
      .kernel_output_feature_dimension = kernel_output_feature_dimension,
      .kernel_spacial_dimensions = kernel_spatial_dimensions,
      .output_batch_dimension = output_batch_dimension,
      .output_feature_dimension = output_feature_dimension,
      .output_spacial_dimensions = output_spatial_dimensions,
      .feature_group_count = feature_group_count,
      .batch_group_count = batch_group_count,
      .precision_configs = precision_configs});

  Vector<StorageT> expected_data =
      Vector<StorageT>{5, 10, 15, 20, 25, 5, 10, 15, 20, -25};
  ;
  if (std::is_same<StorageT, I4>::value) {
    StorageT min_value = static_cast<int>(Storage<DataType::kSI4>::kMinValue);
    StorageT max_value = static_cast<int>(Storage<DataType::kSI4>::kMaxValue);
    for (int i = 0; i < expected_data.size(); ++i) {
      if (expected_data[i] < min_value) expected_data[i] = min_value;

      if (expected_data[i] > max_value) expected_data[i] = max_value;
    }
  }

  Vector<StorageT> expected_data_quantized(shape_result.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_data_quantized.begin(), [&](StorageT val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1) / scale);
                 });

  ASSERT_OK(Prepare(op, lhs, rhs, output_tensor));
  ASSERT_OK(Evaluate(op, lhs, rhs, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

}  // namespace
}  // namespace shlo_ref
