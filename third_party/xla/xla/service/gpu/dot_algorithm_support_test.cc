/* Copyright 2024 The OpenXLA Authors.

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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct TestParams {
  PrecisionConfig::Algorithm algorithm;
  PrimitiveType input_storage_type;
  PrimitiveType output_storage_type;
  se::CudaComputeCapability min_cuda_capability;
};

std::string TestParamsToString(
    const ::testing::TestParamInfo<TestParams>& info) {
  const TestParams& params = info.param;
  return absl::StrFormat(
      "%s_with_input_%s_output_%s_from_cc_%d_%d",
      AlgorithmToString(params.algorithm),
      primitive_util::LowercasePrimitiveTypeName(params.input_storage_type),
      primitive_util::LowercasePrimitiveTypeName(params.output_storage_type),
      params.min_cuda_capability.major, params.min_cuda_capability.minor);
}

// These are integration tests.
// TODO(tdanyluk): Consider checking somehow directly if the correct algorithms
// are called / emitted. Currently the emitters should decline unsupported
// algorithms, but maybe we could check this directly.

class DotAlgorithmSupportTest
    : public HloTestBase,
      public ::testing::WithParamInterface<TestParams> {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

// A parametrized test that checks if an algorithm is supported, with the given
// input and output storage types, from a given cuda capability.
TEST_P(DotAlgorithmSupportTest, AlgorithmIsSupportedFromCudaCapability) {
  const TestParams& params = GetParam();
  const std::string hlo_text = absl::Substitute(
      R"(
    HloModule test

    ENTRY test {
      x = $1[32,32] parameter(0)
      y = $1[32,32] parameter(1)

      ROOT out = $2[32,32] dot(x, y),
                 lhs_contracting_dims={1},
                 rhs_contracting_dims={0},
                 algorithm=$0
    }
  )",
      AlgorithmToString(params.algorithm),
      primitive_util::LowercasePrimitiveTypeName(params.input_storage_type),
      primitive_util::LowercasePrimitiveTypeName(params.output_storage_type));

  if (GetCudaComputeCapability().IsAtLeast(params.min_cuda_capability.major,
                                           params.min_cuda_capability.minor)) {
    EXPECT_TRUE(Run(hlo_text));
  } else {
    EXPECT_THAT(Run(hlo_text).message(),
                ::testing::HasSubstr("Unsupported algorithm"));
  }
}

using PC = PrecisionConfig;
using CC = se::CudaComputeCapability;
INSTANTIATE_TEST_SUITE_P(
    All, DotAlgorithmSupportTest,
    ::testing::ValuesIn(std::vector<TestParams>{
        // Other combinations with input_storage_type=F8E5M2 should also work,
        // but we don't want to generate too many tests here.
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32, F8E5M2, F8E5M2, CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32, F8E4M3FN, F8E4M3FN, CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32, F8E4M3FN, F16, CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32, F8E4M3FN, BF16, CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32, F8E4M3FN, F32, CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM, F8E4M3FN, F8E4M3FN,
         CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM, F8E4M3FN, F16, CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM, F8E4M3FN, BF16, CC(8, 9)},
        {PC::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM, F8E4M3FN, F32, CC(8, 9)},
        {PC::ALG_DOT_F16_F16_F32, F16, F16, CC(0, 0)},
        {PC::ALG_DOT_F16_F16_F32, F16, F32, CC(0, 0)},
        {PC::ALG_DOT_BF16_BF16_F32, BF16, BF16, CC(8, 0)},
        {PC::ALG_DOT_BF16_BF16_F32, BF16, F32, CC(8, 0)},
        {PC::ALG_DOT_BF16_BF16_F32_X6, F32, F32, CC(8, 0)},
        {PC::ALG_DOT_BF16_BF16_F32_X3, F32, F32, CC(8, 0)},
        {PC::ALG_DOT_TF32_TF32_F32, F32, F32, CC(8, 0)},
        {PC::ALG_DOT_F32_F32_F32, F32, F32, CC(0, 0)},
        {PC::ALG_DOT_F64_F64_F64, F64, F64, CC(0, 0)},
    }),
    TestParamsToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
