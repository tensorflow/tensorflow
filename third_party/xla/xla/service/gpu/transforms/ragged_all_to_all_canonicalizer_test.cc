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

#include "xla/service/gpu/transforms/ragged_all_to_all_canonicalizer.h"

#include <memory>

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class RaggedAllToAllCanonicalizerTest : public HloRunnerAgnosticTestBase {
 public:
  RaggedAllToAllCanonicalizerTest()
      : HloRunnerAgnosticTestBase(std::make_unique<HloRunner>(
            PlatformUtil::GetDefaultPlatform().value())) {}
};

TEST_F(RaggedAllToAllCanonicalizerTest, SimpleRaggedAllToAllIsCanonicalized) {
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

  RaggedAllToAllCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(module.get(), {}));
  EXPECT_TRUE(changed);
  TF_EXPECT_OK(VerifyHloModule(module.get(), true, true));

  auto* ragged_all_to_all =
      FindInstruction(module.get(), HloOpcode::kRaggedAllToAll);
  EXPECT_NE(ragged_all_to_all, nullptr);
  EXPECT_EQ(ragged_all_to_all->operand(2)->shape().element_type(), S64);
  EXPECT_EQ(ragged_all_to_all->operand(3)->shape().element_type(), S64);
  EXPECT_EQ(ragged_all_to_all->operand(4)->shape().element_type(), S64);
  EXPECT_EQ(ragged_all_to_all->operand(5)->shape().element_type(), S64);
}

TEST_F(RaggedAllToAllCanonicalizerTest, CanonicalRaggedAllToAllIsNotChanged) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  input = bf16[16] parameter(0)
  output = bf16[16] parameter(1)
  input_offsets = s64[2] parameter(2)
  send_sizes = s64[2] parameter(3)
  output_offsets = s64[2] parameter(4)
  recv_sizes = s64[2] parameter(5)
  ROOT ra2a = bf16[16] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
}
)"));

  RaggedAllToAllCanonicalizer canonicalizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, canonicalizer.Run(module.get(), {}));
  EXPECT_FALSE(changed);
  TF_EXPECT_OK(VerifyHloModule(module.get(), true, true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
