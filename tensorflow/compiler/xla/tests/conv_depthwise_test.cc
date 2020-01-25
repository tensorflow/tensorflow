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
#include "tensorflow/compiler/xla/tests/conv_depthwise_common.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class DepthwiseConvolution2DTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ::testing::tuple<DepthwiseConvolution2DSpec, bool>> {};

static std::vector<DepthwiseConvolution2DSpec> GetConv2DTestCases() {
  std::vector<DepthwiseConvolution2DSpec> config_set;
  std::vector<std::vector<int64>> config_options = {
      {128, 6, 3, 64},  {256, 5, 3, 256}, {256, 5, 2, 144}, {144, 5, 3, 64},
      {144, 5, 2, 256}, {8, 48, 17, 8},   {128, 20, 6, 64}, {64, 14, 12, 172},
      {16, 9, 4, 16},   {128, 1, 2, 144}, {256, 1, 2, 64},  {256, 1, 2, 2},
      {144, 5, 3, 3},   {8, 48, 17, 1},   {16, 9, 5, 4}};

  for (auto option : config_options) {
    int64 feature = option[0];
    int64 activation_size = option[1];
    int64 kernel_size = option[2];
    int64 batch = option[3];

    std::vector<int64> kernel_layout = {3, 2, 1, 0};
    DepthwiseConvolution2DSpec config;
    config.output_feature = feature;
    config.window = kernel_size;

    config.activation_dims = {batch, activation_size, activation_size, feature};
    config.activation_layout = {3, 0, 2, 1};

    config.kernel_dims = {kernel_size, kernel_size, 1, feature};
    config.kernel_layout = {3, 2, 1, 0};
    config.output_layout = {3, 0, 2, 1};

    if (activation_size == 1 && kernel_size == 2) {
      config.stride = config.pad = config.lhs_dilate = -1;
      // Test for outer dim.
      config.output_dims = {batch, activation_size + kernel_size - 1,
                            activation_size + kernel_size, feature};
    } else if (feature == 256) {
      // Restrict dilation-based tests only to one feature configuration.
      config.stride = activation_size - 1;
      config.pad = 0;
      config.lhs_dilate = feature / 32;
      config.output_dims = {batch, feature / 32,
                            activation_size - kernel_size + 1, feature};
    } else {
      config.stride = config.pad = config.lhs_dilate = -1;
      config.output_dims = {batch, activation_size - kernel_size + 1,
                            activation_size - kernel_size + 1, feature};
    }
    config_set.push_back(config);
  }

  return config_set;
}


XLA_TEST_P(DepthwiseConvolution2DTest, DoIt) {
  const DepthwiseConvolution2DSpec& spec = ::testing::get<0>(GetParam());
  bool use_bfloat16 = ::testing::get<1>(GetParam());

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_BFLOAT16
  if (use_bfloat16) {
    return;
  }
#endif

  const string hlo_text =
      BuildHloTextDepthwiseConvolution2D(spec, use_bfloat16);

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0.01, 0.01},
                            [](HloModule* module) -> Status {
                              BFloat16MixedPrecisionRemoval remover;
                              TF_RETURN_IF_ERROR(remover.Run(module).status());
                              Despecializer despecializer;
                              return despecializer.Run(module).status();
                            }));
}

INSTANTIATE_TEST_CASE_P(
    DepthwiseConvolution2DTestWithRandomIndices, DepthwiseConvolution2DTest,
    ::testing::Combine(::testing::ValuesIn(GetConv2DTestCases()),
                       ::testing::Bool()),
    DepthwiseConvolution2DTestDataToString);

}  // namespace
}  // namespace xla
