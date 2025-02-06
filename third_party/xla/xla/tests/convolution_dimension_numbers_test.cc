/* Copyright 2017 The OpenXLA Authors.

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

#include <algorithm>
#include <array>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/array4d.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/reference_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

absl::StatusOr<ConvolutionDimensionNumbers> CreateConvDimensionNumbers(
    int64_t input_batch, int64_t input_feature, int64_t input_first_spatial,
    int64_t input_second_spatial, int64_t output_batch, int64_t output_feature,
    int64_t output_first_spatial, int64_t output_second_spatial,
    int64_t kernel_output_feature, int64_t kernel_input_feature,
    int64_t kernel_first_spatial, int64_t kernel_second_spatial) {
  ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_input_batch_dimension(input_batch);
  dimension_numbers.set_input_feature_dimension(input_feature);
  dimension_numbers.add_input_spatial_dimensions(input_first_spatial);
  dimension_numbers.add_input_spatial_dimensions(input_second_spatial);
  dimension_numbers.set_kernel_output_feature_dimension(kernel_output_feature);
  dimension_numbers.set_kernel_input_feature_dimension(kernel_input_feature);
  dimension_numbers.add_kernel_spatial_dimensions(kernel_first_spatial);
  dimension_numbers.add_kernel_spatial_dimensions(kernel_second_spatial);
  dimension_numbers.set_output_batch_dimension(output_batch);
  dimension_numbers.set_output_feature_dimension(output_feature);
  dimension_numbers.add_output_spatial_dimensions(output_first_spatial);
  dimension_numbers.add_output_spatial_dimensions(output_second_spatial);
  TF_RETURN_IF_ERROR(XlaBuilder::Validate(dimension_numbers));
  return dimension_numbers;
}

class ConvolutionDimensionNumbersTest : public ClientLibraryTestBase {};

// Tests the convolution operation with invalid input dimension numbers.
TEST_F(ConvolutionDimensionNumbersTest, InvalidInputDimensionNumbers) {
  auto dimension_numbers_status =
      CreateConvDimensionNumbers(0, 2, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
  ASSERT_FALSE(dimension_numbers_status.ok());
  ASSERT_THAT(dimension_numbers_status.status().message(),
              ::testing::HasSubstr("input are not unique"));
}

// Tests the convolution operation with invalid weight dimension numbers.
TEST_F(ConvolutionDimensionNumbersTest, InvalidWeightDimensionNumbers) {
  auto dimension_numbers_status =
      CreateConvDimensionNumbers(0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 2, 3);
  ASSERT_FALSE(dimension_numbers_status.ok());
  ASSERT_THAT(dimension_numbers_status.status().message(),
              ::testing::HasSubstr("weight are not unique"));
}

// Tests the convolution operation with invalid output dimension numbers.
TEST_F(ConvolutionDimensionNumbersTest, InvalidOutputDimensionNumbers) {
  auto dimension_numbers_status =
      CreateConvDimensionNumbers(0, 1, 2, 3, 0, 2, 2, 3, 0, 1, 2, 3);
  ASSERT_FALSE(dimension_numbers_status.ok());
  ASSERT_THAT(dimension_numbers_status.status().message(),
              ::testing::HasSubstr("output are not unique"));
}

XLA_TEST_F(ConvolutionDimensionNumbersTest,
           TwoConvsWithDifferentDimensionNumbers) {
  auto input_array = std::make_unique<Array4D<float>>(2, 3, 5, 5);
  input_array->FillWithMultiples(0.1);
  auto weight_array = std::make_unique<Array4D<float>>(4, 3, 1, 1);
  weight_array->FillWithMultiples(0.2);
  auto weight_data =
      client_->TransferToServer(LiteralUtil::CreateR4FromArray4D(*weight_array))
          .value();

  XlaBuilder builder(TestName());
  auto input = ConstantR4FromArray4D<float>(&builder, *input_array);
  auto weight =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {4, 3, 1, 1}), "weight");
  auto conv1 = Conv(input, weight, {1, 1}, Padding::kValid);

  ConvolutionDimensionNumbers dim_nums =
      XlaBuilder::CreateDefaultConvDimensionNumbers();
  // Swap batch_dimension and feature_dimension.
  int64_t old_input_batch_dim = dim_nums.input_batch_dimension();
  int64_t old_output_batch_dim = dim_nums.output_batch_dimension();
  dim_nums.set_input_batch_dimension(dim_nums.input_feature_dimension());
  dim_nums.set_output_batch_dimension(dim_nums.output_feature_dimension());
  dim_nums.set_input_feature_dimension(old_input_batch_dim);
  dim_nums.set_output_feature_dimension(old_output_batch_dim);
  // Swap kernel_input_feature_dimension and kernel_output_feature_dimension.
  int64_t old_kernel_input_feature_dim =
      dim_nums.kernel_input_feature_dimension();
  dim_nums.set_kernel_input_feature_dimension(
      dim_nums.kernel_output_feature_dimension());
  dim_nums.set_kernel_output_feature_dimension(old_kernel_input_feature_dim);
  ConvWithGeneralDimensions(input, conv1, {1, 1}, Padding::kValid, dim_nums);

  auto expected_conv1 = ReferenceUtil::ConvArray4D(*input_array, *weight_array,
                                                   {1, 1}, Padding::kValid);
  auto expected_conv2 = ReferenceUtil::ConvArray4DGeneralDimensions(
      *input_array, *expected_conv1, {1, 1}, Padding::kValid, dim_nums);

  ComputeAndCompareR4<float>(&builder, *expected_conv2, {weight_data.get()},
                             ErrorSpec(0.001, 0.01));
}

}  // namespace
}  // namespace xla
