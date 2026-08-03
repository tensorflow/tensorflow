/* Copyright 2026 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla::cpu {
namespace {

class CorpusCompileTest : public HloTestBase,
                          public ::testing::WithParamInterface<const char*> {};

TEST_P(CorpusCompileTest, CompilesCleanly) {
  const char* filename = GetParam();
  std::string file_path = tsl::io::JoinPath(
      tsl::testing::XlaSrcRoot(), "backends/cpu/benchmarks/hlo", filename);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       LoadModuleFromFile(file_path));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  EXPECT_NE(executable, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    CorpusCompileTestSuite, CorpusCompileTest,
    ::testing::Values(
        "argsort_axis_1024x512_bf16.hlo",
        "depthwise_conv_3x3_1x256x56x56_bf16.hlo",
        "dynamic_slice_loop_1x2048x768_bf16.hlo",
        "gemma3_1b_flax_sample_loop.hlo",
        "jax.issue.33666.linx.frag_0100.module_0005.hlo",
        "jax.issue.33666.linx.frag_0100.module_0009.hlo",
        "jax.issue.33666.linx.frag_0100.module_0019.hlo",
        "jax.issue.33666.linx.frag_0428.module_0002.hlo",
        "jax.issue.33666.linx.frag_0428.module_0004.hlo",
        "jax.issue.33666.linx.frag_0428.module_0009.hlo",
        "jax.issue.33666.linx.slow_0100.module_0007.hlo",
        "jax.issue.33666.linx.slow_0100.module_0013.hlo",
        "jax.issue.33666.linx.slow_0428.module_0003.hlo",
        "jax.issue.33666.linx.slow_0428.module_0006.hlo",
        "layer_norm_1x4096x768_bf16.hlo", "mean_axis_1x4096x1024_bf16.hlo",
        "mha_block_1x12x128x64_bf16.hlo", "sort_full_1024x4096_bf16.hlo",
        "sum_axis_1x4096x1024_bf16.hlo", "topk_logits_k10_1x50000_bf16.hlo",
        "gemma3_1b_flax_call.hlo", "gemma2_2b_keras_jax.hlo"));

}  // namespace
}  // namespace xla::cpu
