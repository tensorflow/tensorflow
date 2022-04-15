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

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

std::string GetFloatDataType(bool use_bfloat16) {
  return use_bfloat16 ? "bf16" : "f32";
}

struct BatchGroupedConvolution2DSpec {
  int64_t output_batch, window, window_dilation;
  std::vector<int64_t> activation_dims;
  std::vector<int64_t> kernel_dims;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> activation_and_kernel_layout;
  std::vector<int64_t> output_layout;
};

class BatchGroupedConvolution2DTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ::testing::tuple<BatchGroupedConvolution2DSpec, bool>> {};

class BatchGroupedConvolution2DDepthTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ::testing::tuple<BatchGroupedConvolution2DSpec, bool>> {};

static std::vector<BatchGroupedConvolution2DSpec> GetConv2DTestCases(
    bool use_depth_multiplier) {
  std::vector<BatchGroupedConvolution2DSpec> config_set;
  std::vector<std::vector<int64_t>> config_options = {
      {129, 10, 3, 2}, {4, 3, 3, 258}, {8, 4, 2, 128},
      {8, 3, 2, 256},  {256, 7, 5, 4}, {128, 6, 6, 4},
      {32, 5, 2, 129}, {16, 4, 3, 2},  {16, 3, 2, 64}};

  int64_t counter = 2;
  for (auto option : config_options) {
    int64_t feature = option[3];
    int64_t activation_size = option[1];
    int64_t kernel_size = option[2];
    int64_t batch = option[0];

    BatchGroupedConvolution2DSpec config;
    config.window_dilation = 1;
    config.output_batch = feature;
    config.window = kernel_size;

    config.activation_dims = {batch, activation_size, activation_size, feature};

    const int64_t depthwise_multiplier = use_depth_multiplier ? counter++ : 1;
    config.kernel_dims = {batch, kernel_size, kernel_size,
                          feature * depthwise_multiplier};
    // Don't let the counter grow too much, else the compute demand will grow.
    if (counter == 4) {
      counter = 2;
    }
    int64_t output_space_size = 3 + activation_size - kernel_size;
    config.output_dims = {output_space_size, output_space_size,
                          feature * depthwise_multiplier, 1};

    config.activation_and_kernel_layout = {0, 3, 1, 2};
    config.output_layout = {2, 3, 0, 1};
    config_set.push_back(config);

    BatchGroupedConvolution2DSpec different_layout_config = config;
    different_layout_config.activation_and_kernel_layout = {3, 0, 1, 2};
    config_set.push_back(different_layout_config);

    // Add configurations for window dilation cases.
    if (activation_size % 2 == 0 && activation_size == kernel_size) {
      BatchGroupedConvolution2DSpec config;
      config.window_dilation = 2;
      config.output_batch = feature;
      config.window = kernel_size / 2;
      config.activation_dims = {batch, activation_size, activation_size,
                                feature};
      config.kernel_dims = {batch, kernel_size / 2, kernel_size / 2, feature};
      config.activation_and_kernel_layout = {0, 3, 1, 2};
      config.output_layout = {2, 3, 0, 1};

      int64_t output_space_size = 5;
      config.output_dims = {output_space_size, output_space_size, feature, 1};

      config_set.push_back(config);

      BatchGroupedConvolution2DSpec different_layout_config = config;
      different_layout_config.activation_and_kernel_layout = {3, 0, 1, 2};
      config_set.push_back(different_layout_config);
    }
  }

  return config_set;
}

std::string BatchGroupedConvolution2DTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<BatchGroupedConvolution2DSpec, bool>>& data) {
  const auto& spec = ::testing::get<0>(data.param);
  const std::string data_type = GetFloatDataType(::testing::get<1>(data.param));
  std::string str = absl::StrCat(
      "activation_dims_", absl::StrJoin(spec.activation_dims, "x"),
      "_kernel_dims_", absl::StrJoin(spec.kernel_dims, "x"),
      "_activation_layout_",
      absl::StrJoin(spec.activation_and_kernel_layout, "_"), "_output_dims_",
      absl::StrJoin(spec.output_dims, "x"), data_type, "_output_layout_",
      absl::StrJoin(spec.output_layout, "_"));

  // Test names are not allowed to contain the '-' character.
  absl::c_replace(str, '-', 'n');
  return str;
}

std::string BuildHloTextBatchGroupedConvolution2D(
    const BatchGroupedConvolution2DSpec& spec, bool use_bfloat16,
    bool scheduled = false) {
  const std::string data_type = GetFloatDataType(use_bfloat16);
  const std::string scheduled_tag = scheduled ? ",is_scheduled=true" : "";
  return absl::StrFormat(
      R"(
    HloModule TensorFlowDepthwiseConv %s

    ENTRY main {
      activation = %s[%s]{%s} parameter(0)
      kernel = %s[%s]{%s} parameter(1)
      ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
          window={size=%dx%d pad=1_%dx1_%d rhs_dilate=%dx%d}, dim_labels=f01b_i01o->01fb,
          batch_group_count=%d
    }
    )",
      scheduled_tag, data_type, absl::StrJoin(spec.activation_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), data_type,
      absl::StrJoin(spec.kernel_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), data_type,
      absl::StrJoin(spec.output_dims, ","),
      absl::StrJoin(spec.output_layout, ","), data_type,
      absl::StrJoin(spec.activation_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), data_type,
      absl::StrJoin(spec.kernel_dims, ","),
      absl::StrJoin(spec.activation_and_kernel_layout, ","), spec.window,
      spec.window, spec.window_dilation, spec.window_dilation,
      spec.window_dilation, spec.window_dilation, spec.output_batch);
}

XLA_TEST_P(BatchGroupedConvolution2DTest, DoIt) {
  const BatchGroupedConvolution2DSpec& spec = ::testing::get<0>(GetParam());
  bool use_bfloat16 = ::testing::get<1>(GetParam());

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_BFLOAT16
  if (use_bfloat16) {
    return;
  }
#endif

  const std::string hlo_text = BuildHloTextBatchGroupedConvolution2D(
      spec, use_bfloat16, /*scheduled=*/false);

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0.01, 0.01}));
}

INSTANTIATE_TEST_CASE_P(
    BatchGroupedConvolution2DTestWithRandomIndices,
    BatchGroupedConvolution2DTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetConv2DTestCases(/*use_depth_multiplier=*/false)),
        ::testing::Bool()),
    BatchGroupedConvolution2DTestDataToString);

INSTANTIATE_TEST_CASE_P(
    BatchGroupedConvolution2DDepthMultiplierTestWithRandomIndices,
    BatchGroupedConvolution2DTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetConv2DTestCases(/*use_depth_multiplier=*/true)),
        ::testing::Bool()),
    BatchGroupedConvolution2DTestDataToString);

}  // namespace
}  // namespace xla
