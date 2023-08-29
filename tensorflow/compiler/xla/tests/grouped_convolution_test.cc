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

#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/service/float_normalization.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {
namespace {

std::string GetFloatDataType(bool use_bfloat16) {
  return use_bfloat16 ? "bf16" : "f32";
}

struct GroupedConvolution2DSpec {
  int64_t input_feature, output_feature, window, stride, pad, lhs_dilate;
  int64_t group_size, group_count;
  std::vector<int64_t> activation_dims;
  std::vector<int64_t> activation_layout;
  std::vector<int64_t> kernel_dims;
  std::vector<int64_t> kernel_layout;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> output_layout;
};

class GroupedConvolution2DTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ::testing::tuple<GroupedConvolution2DSpec, bool>> {};

static std::vector<GroupedConvolution2DSpec> GetConv2DTestCases() {
  std::vector<GroupedConvolution2DSpec> config_set;
  // Add to this set if you want a new test configuration.
  // Rule : the penultimate number must be divisible by the last number.
  std::vector<std::vector<int64_t>> config_options = {
      {8, 2, 2, 1, 1024, 128},
      {512, 3, 3, 144, 1024, 16},
      {256, 3, 3, 129, 512, 64},
      {64, 1, 2, 127, 32, 8},
      {256, 3, 3, 256, 1024, 4}};

  for (auto option : config_options) {
    int64_t output_feature = option[0];
    int64_t activation_size = option[1];
    int64_t kernel_size = option[2];
    int64_t batch = option[3];
    int64_t input_feature = option[4];
    int64_t group_size = option[5];

    std::vector<int64_t> kernel_layout = {3, 2, 1, 0};
    GroupedConvolution2DSpec config;
    config.group_size = group_size;
    config.group_count = input_feature / group_size;
    config.output_feature = output_feature;
    config.window = kernel_size;

    config.activation_dims = {batch, activation_size, activation_size,
                              input_feature};
    config.activation_layout = {3, 0, 2, 1};

    config.kernel_dims = {kernel_size, kernel_size, group_size, output_feature};
    config.kernel_layout = {3, 2, 1, 0};

    if (activation_size == 1 && kernel_size == 2) {
      config.stride = config.pad = config.lhs_dilate = -1;
      // Test for outer dim.
      config.output_dims = {batch, activation_size + kernel_size - 1,
                            activation_size + kernel_size, output_feature};
    } else if (output_feature == 256) {
      // Restrict dilation-based tests only to one feature configuration.
      config.stride = activation_size - 1;
      config.pad = 0;
      config.lhs_dilate = output_feature / 32;
      config.output_dims = {batch, output_feature / 32,
                            activation_size - kernel_size + 1, output_feature};
    } else {
      config.stride = config.pad = config.lhs_dilate = -1;
      config.output_dims = {batch, activation_size - kernel_size + 1,
                            activation_size - kernel_size + 1, output_feature};
    }

    // Try this layout for all kernel shapes.
    config.output_layout = {3, 0, 2, 1};
    config_set.push_back(config);

    // Try other layouts only for certain kernel shapes.
    if (kernel_size % 2 == 0) {
      config.activation_layout = {0, 3, 2, 1};
      config_set.push_back(config);

      config.output_layout = {0, 3, 2, 1};
      config_set.push_back(config);

      config.activation_layout = {3, 0, 2, 1};
      config_set.push_back(config);
    }
  }

  return config_set;
}

std::string GroupedConvolution2DTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<GroupedConvolution2DSpec, bool>>& data) {
  const auto& spec = ::testing::get<0>(data.param);
  const std::string data_type = GetFloatDataType(::testing::get<1>(data.param));
  std::string str = absl::StrCat(
      "activation_dims_", absl::StrJoin(spec.activation_dims, "x"),
      "_activation_layout_", absl::StrJoin(spec.activation_layout, "_"),
      "_kernel_dims_", absl::StrJoin(spec.kernel_dims, "x"), "_kernel_layout_",
      absl::StrJoin(spec.kernel_layout, "_"), "_output_dims_",
      absl::StrJoin(spec.output_dims, "x"), "_output_layout_",
      absl::StrJoin(spec.output_layout, "_"), data_type);
  // -1 indicates non-existence.
  if (spec.stride != -1) {
    absl::StrAppend(&str, "_lhs_dilation_", spec.lhs_dilate, "x1");
  }

  // Test names are not allowed to contain the '-' character.
  absl::c_replace(str, '-', 'n');
  return str;
}

std::string BuildHloTextGroupedConvolution2D(
    const GroupedConvolution2DSpec& spec, bool use_bfloat16) {
  const std::string data_type = GetFloatDataType(use_bfloat16);
  if (spec.activation_dims[1] == 1 && spec.kernel_dims[1] == 2) {
    // Check for outer dim.
    return absl::StrFormat(
        R"(
    HloModule TensorFlowDepthwiseConv

    ENTRY main {
      activation = %s[%s]{%s} parameter(0)
      kernel = %s[%s]{%s} parameter(1)
      ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
          window={size=%dx%d  pad=1_1x%d_%d rhs_dilate=1x%d}, dim_labels=b01f_01io->b01f,
          feature_group_count=%d
    }
    )",
        data_type, absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), data_type,
        absl::StrJoin(spec.output_dims, ","),
        absl::StrJoin(spec.output_layout, ","), data_type,
        absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), spec.window, spec.window,
        spec.window, spec.window, spec.window, spec.group_count);

  } else if (spec.stride == -1) {
    // Check for basic, non-dilated cases.
    return absl::StrFormat(
        R"(
      HloModule TensorFlowDepthwiseConv

      ENTRY main {
        activation = %s[%s]{%s} parameter(0)
        kernel = %s[%s]{%s} parameter(1)
        ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
            window={size=%dx%d}, dim_labels=b01f_01io->b01f,
            feature_group_count=%d
      }
      )",
        data_type, absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), data_type,
        absl::StrJoin(spec.output_dims, ","),
        absl::StrJoin(spec.output_layout, ","), data_type,
        absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), spec.window, spec.window,
        spec.group_count);
  } else {
    // Check for base dilations.
    return absl::StrFormat(
        R"(
    HloModule TensorFlowDepthwiseConv

    ENTRY main {
      activation = %s[%s]{%s} parameter(0)
      kernel = %s[%s]{%s} parameter(1)
      ROOT conv = %s[%s]{%s} convolution(%s[%s]{%s} activation, %s[%s]{%s} kernel),
          window={size=%dx%d stride=%dx1 pad=%d_%dx0_0 lhs_dilate=%dx1}, 
          dim_labels=b01f_01io->b01f, feature_group_count=%d
    }
    )",
        data_type, absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), data_type,
        absl::StrJoin(spec.output_dims, ","),
        absl::StrJoin(spec.output_layout, ","), data_type,
        absl::StrJoin(spec.activation_dims, ","),
        absl::StrJoin(spec.activation_layout, ","), data_type,
        absl::StrJoin(spec.kernel_dims, ","),
        absl::StrJoin(spec.kernel_layout, ","), spec.window, spec.window,
        spec.stride, 0, 0, spec.lhs_dilate, spec.group_count);
  }
}

XLA_TEST_P(GroupedConvolution2DTest, DoIt) {
  const GroupedConvolution2DSpec& spec = ::testing::get<0>(GetParam());
  bool use_bfloat16 = ::testing::get<1>(GetParam());

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_BFLOAT16
  if (use_bfloat16) {
    return;
  }
#endif

  const std::string hlo_text =
      BuildHloTextGroupedConvolution2D(spec, use_bfloat16);

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0.01, 0.01},
                            [](HloModule* module) -> Status {
                              BFloat16MixedPrecisionRemoval remover;
                              TF_RETURN_IF_ERROR(remover.Run(module).status());
                              Despecializer despecializer;
                              return despecializer.Run(module).status();
                            }));
}

INSTANTIATE_TEST_CASE_P(
    GroupedConvolution2DTestWithRandomIndices, GroupedConvolution2DTest,
    ::testing::Combine(::testing::ValuesIn(GetConv2DTestCases()),
                       ::testing::Bool()),
    GroupedConvolution2DTestDataToString);

using GroupedConvolutionTest = HloTestBase;

XLA_TEST_F(GroupedConvolutionTest, BackwardInputConvolution) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule convolution_module

ENTRY convolution {
  p1 = f32[2,1,1,1]{3,2,1,0} parameter(0)
  p2 = f32[2,4,4,1]{3,2,1,0} parameter(1)
  reverse = f32[2,4,4,1]{3,2,1,0} reverse(p2), dimensions={1,2}
  ROOT convolution = f32[2,4,4,1]{3,2,1,0} convolution(p1, reverse), window={size=4x4 pad=3_3x3_3}, dim_labels=fb01_o01i->f01b, feature_group_count=2
}
)")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(auto fake_arguments, MakeFakeArguments(module.get()));
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return &const_cast<Literal&>(literal); });
  EXPECT_TRUE(RunAndCompare(std::move(module), fake_argument_ptrs,
                            ErrorSpec{0.01, 0.01}));
}

}  // namespace
}  // namespace xla
