/* Copyright 2025 The OpenXLA Authors.

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

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/transforms/block_scaling_rewriter.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using BlockScalingRewriterExecutionTest = HloPjRtTestBase;

TEST_F(BlockScalingRewriterExecutionTest, QuantizeDequantizeCompare) {
  constexpr absl::string_view hlo_test = R"(
HloModule test
ENTRY main {
  %input = f32[256,256] parameter(0)
  %quantized = (f8e4m3fn[256,256], f8e5m2[256,16]) custom-call(%input),
      custom_call_target="__op$quantize"
  %values = f8e4m3fn[256,256] get-tuple-element(%quantized), index=0
  %scales = f8e5m2[256,16] get-tuple-element(%quantized), index=1
  ROOT %dequantized = f32[256,256] custom-call(%values, %scales),
      custom_call_target="__op$dequantize"
})";
  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnUnverifiedModule(hlo_test));

  BlockScalingRewriter pass(se::dnn::VersionInfo{});
  TF_ASSERT_OK_AND_ASSIGN(
      auto changed, pass.Run(test_module.get(), /*execution_threads=*/{}));
  EXPECT_TRUE(changed);

  constexpr absl::string_view hlo_reference = R"(
HloModule reference
ENTRY main {
  ROOT %input = f32[256,256] parameter(0)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto reference_module,
                          ParseAndReturnUnverifiedModule(hlo_reference));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(test_module),
                                      std::move(reference_module),
                                      ErrorSpec(/*aabs=*/0.01, /*arel=*/0.07),
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace xla::gpu
