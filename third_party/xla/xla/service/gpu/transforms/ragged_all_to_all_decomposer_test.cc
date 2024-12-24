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

#include "xla/service/gpu/transforms/ragged_all_to_all_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class RaggedAllToAllDecomposerTest : public HloRunnerAgnosticTestBase {
 public:
  RaggedAllToAllDecomposerTest()
      : HloRunnerAgnosticTestBase(
            std::make_unique<HloRunner>(
                PlatformUtil::GetDefaultPlatform().value()),
            std::make_unique<HloRunner>(
                PlatformUtil::GetDefaultPlatform().value())) {}
};

TEST_F(RaggedAllToAllDecomposerTest, SimpleRaggedAllToAllIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  input = bf16[16] parameter(0)
  output = bf16[16] parameter(1)
  input_offsets = s32[2] parameter(2)
  send_sizes = s32[2] parameter(3)
  output_offsets = s32[2] parameter(4)
  recv_sizes = s32[2] parameter(5)
  ROOT ra2a = bf16[16] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
}
)"));

  RaggedAllToAllDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_TRUE(changed);
  TF_EXPECT_OK(VerifyHloModule(module.get(), true, true));
  TF_EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK-COUNT-2: bf16[16]{0} dynamic-slice
    // CHECK: (bf16[16]{0}, bf16[16]{0}) all-to-all
    // CHECK-COUNT-2: bf16[32]{0} dynamic-update-slice
    // CHECK: bf16[16]{0} slice({{.*}}), slice={[0:16]}
    // CHECK: ROOT {{.*}} bf16[16]{0} select
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest,
       RaggedAllToAllWithMultiDimInputIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  input = bf16[16,8,32] parameter(0)
  output = bf16[16,8,32] parameter(1)
  input_offsets = s32[4] parameter(2)
  send_sizes = s32[4] parameter(3)
  output_offsets = s32[4] parameter(4)
  recv_sizes = s32[4] parameter(5)
  ROOT ra2a = bf16[16,8,32] ragged-all-to-all(input, output, input_offsets,
  send_sizes, output_offsets, recv_sizes), replica_groups={{0,1,2,3}}
}
)"));

  RaggedAllToAllDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_TRUE(changed);
  TF_EXPECT_OK(VerifyHloModule(module.get(), true, true));
  TF_EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(
      *RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
    // CHECK-COUNT-4: bf16[16,8,32]{2,1,0} dynamic-slice
    // CHECK: (bf16[16,8,32]{2,1,0}, bf16[16,8,32]{2,1,0}, bf16[16,8,32]{2,1,0},
    // CHECK-SAME: bf16[16,8,32]{2,1,0}) all-to-all
    // CHECK-COUND-4: bf16[32]{0} dynamic-update-slice
    // CHECK: bf16[16,8,32]{2,1,0} slice({{.*}}), slice={[0:16], [0:8], [0:32]}
    // CHECK: ROOT {{.*}} bf16[16,8,32]{2,1,0} select
  )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
