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

#include "xla/backends/gpu/transforms/dot_algorithm_rewriter.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class DotAlgorithmRewriterTest : public HloHardwareIndependentTestBase {};

TEST_F(DotAlgorithmRewriterTest, DefaultToBF16) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(auto pass_result,
                       RunHloPass(DotAlgorithmRewriter(), module.get()));
  EXPECT_TRUE(pass_result);

  const char* expected = R"(
      CHECK: %[[P0:.*]] = f32[32,32]{{.*}} parameter(0)
      CHECK: %[[P0_BF16:.*]] = bf16[32,32]{{.*}} convert(%[[P0]])
      CHECK: %[[P1:.*]] = f32[32,32]{{.*}} parameter(1)
      CHECK: %[[P1_BF16:.*]] = bf16[32,32]{{.*}} convert(%[[P1]])
      CHECK: ROOT %dot.1 = f32[32,32]{{.*}} dot(%[[P0_BF16]], %[[P1_BF16]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_f32
  )";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), expected));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotAlgorithmRewriterTest, NoDefaultToBF16) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(false);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));

  ASSERT_OK_AND_ASSIGN(bool pass_result,
                       RunHloPass(DotAlgorithmRewriter(), module.get()));
  EXPECT_FALSE(pass_result);
}

}  // namespace
}  // namespace xla::gpu
