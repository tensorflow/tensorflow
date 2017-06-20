/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ConvolutionDimensionNumbersTest : public ClientLibraryTestBase {};

// Tests the convolution operation with invalid input dimension numbers.
TEST_F(ConvolutionDimensionNumbersTest, InvalidInputDimensionNumbers) {
  auto dimension_numbers_status =
      ComputationBuilder::CreateConvDimensionNumbers(0, 2, 2, 3, 0, 1, 2, 3);
  ASSERT_FALSE(dimension_numbers_status.ok());
  ASSERT_THAT(dimension_numbers_status.status().error_message(),
              ::testing::HasSubstr("input are not unique"));
}

// Tests the convolution operation with invalid weight dimension numbers.
TEST_F(ConvolutionDimensionNumbersTest, InvalidWeightDimensionNumbers) {
  auto dimension_numbers_status =
      ComputationBuilder::CreateConvDimensionNumbers(0, 1, 2, 3, 2, 3, 2, 3);
  ASSERT_FALSE(dimension_numbers_status.ok());
  ASSERT_THAT(dimension_numbers_status.status().error_message(),
              ::testing::HasSubstr("weight are not unique"));
}

XLA_TEST_F(ConvolutionDimensionNumbersTest,
           TwoConvsWithDifferentDimensionNumbers) {
  auto input_array = MakeUnique<Array4D<float>>(2, 3, 5, 5);
  input_array->FillWithMultiples(0.1);
  auto weight_array = MakeUnique<Array4D<float>>(4, 3, 1, 1);
  weight_array->FillWithMultiples(0.2);
  auto weight_data =
      client_
          ->TransferToServer(*LiteralUtil::CreateR4FromArray4D(*weight_array))
          .ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  auto input = builder.ConstantR4FromArray4D<float>(*input_array);
  auto weight =
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {4, 3, 1, 1}), "weight");
  auto conv1 = builder.Conv(input, weight, {1, 1}, Padding::kValid);

  ConvolutionDimensionNumbers dim_nums =
      ComputationBuilder::CreateDefaultConvDimensionNumbers();
  // Swap batch_dimension and feature_dimension.
  int64 tmp = dim_nums.batch_dimension();
  dim_nums.set_batch_dimension(dim_nums.feature_dimension());
  dim_nums.set_feature_dimension(tmp);
  // Swap kernel_input_feature_dimension and kernel_output_feature_dimension.
  tmp = dim_nums.kernel_input_feature_dimension();
  dim_nums.set_kernel_input_feature_dimension(
      dim_nums.kernel_output_feature_dimension());
  dim_nums.set_kernel_output_feature_dimension(tmp);
  builder.ConvWithGeneralDimensions(input, conv1, {1, 1}, Padding::kValid,
                                    dim_nums);

  auto expected_conv1 = ReferenceUtil::ConvArray4D(*input_array, *weight_array,
                                                   {1, 1}, Padding::kValid);
  auto expected_conv2 = ReferenceUtil::ConvArray4DGeneralDimensions(
      *input_array, *expected_conv1, {1, 1}, Padding::kValid, dim_nums);

  ComputeAndCompareR4<float>(&builder, *expected_conv2, {weight_data.get()},
                             ErrorSpec(0.001, 0.01));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
