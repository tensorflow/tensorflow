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

#ifndef XLA_TESTS_CONV_DEPTHWISE_COMMON_H_
#define XLA_TESTS_CONV_DEPTHWISE_COMMON_H_

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/test.h"

namespace xla {
std::string GetFloatDataType(bool use_bfloat16);

struct DepthwiseConvolution2DSpec {
  int64_t output_feature = -1, window = -1, stride = -1, pad = -1,
          lhs_dilate = -1;
  std::vector<int64_t> activation_dims;
  std::vector<int64_t> activation_layout;
  std::vector<int64_t> kernel_dims;
  std::vector<int64_t> kernel_layout;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> output_layout;
};

std::string DepthwiseConvolution2DTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<DepthwiseConvolution2DSpec, bool>>& data);

std::string BuildHloTextDepthwiseConvolution2D(
    const DepthwiseConvolution2DSpec& spec, bool use_bfloat16,
    bool is_scheduled = false);

}  // namespace xla
#endif  // XLA_TESTS_CONV_DEPTHWISE_COMMON_H_
