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

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/hlo_cse.h"
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

using ::testing::HasSubstr;

class RaggedAllToAllDecomposerTest : public HloRunnerAgnosticTestBase {
 public:
  RaggedAllToAllDecomposerTest()
      : HloRunnerAgnosticTestBase(std::make_unique<HloRunner>(
            PlatformUtil::GetDefaultPlatform().value())) {}
};

TEST_F(RaggedAllToAllDecomposerTest, SimpleRaggedAllToAllIsSupported) {
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

  RaggedAllToAllDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_TRUE(changed);
  TF_EXPECT_OK(VerifyHloModule(module.get(), true, true));
  TF_EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: s64[2,1]{1,0} all-to-all
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: (bf16[1,16]{1,0}, bf16[1,16]{1,0}) all-to-all
    // CHECK: dynamic-update-slice
    // CHECK: iota
    // CHECK: compare
    // CHECK: select
    // CHECK: select
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest,
       RaggedAllToAllWithoutReplicaGroupsIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module, replica_count=2

ENTRY main {
  input = bf16[16] parameter(0)
  output = bf16[16] parameter(1)
  input_offsets = s64[2] parameter(2)
  send_sizes = s64[2] parameter(3)
  output_offsets = s64[2] parameter(4)
  recv_sizes = s64[2] parameter(5)
  ROOT ra2a = bf16[16] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={}
}
)"));

  RaggedAllToAllDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_TRUE(changed);
  TF_EXPECT_OK(VerifyHloModule(module.get(), true, true));
  TF_EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: s64[2,1]{1,0} all-to-all
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: (bf16[1,16]{1,0}, bf16[1,16]{1,0}) all-to-all
    // CHECK: dynamic-update-slice
    // CHECK: iota
    // CHECK: compare
    // CHECK: select
    // CHECK: select
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest,
       RaggedAllToAllWithMultipleUpdatesPerReplicaIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  input = bf16[16] parameter(0)
  output = bf16[16] parameter(1)
  input_offsets = s64[4] parameter(2)
  send_sizes = s64[4] parameter(3)
  output_offsets = s64[4] parameter(4)
  recv_sizes = s64[4] parameter(5)
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
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: (bf16[2,16]{1,0}, bf16[2,16]{1,0}) all-to-all
    // CHECK: dynamic-update-slice
    // CHECK: iota
    // CHECK: compare
    // CHECK: select
    // CHECK: select
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest,
       RaggedAllToAllWithMultiDimInputIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  input = bf16[16,8,32] parameter(0)
  output = bf16[16,8,32] parameter(1)
  input_offsets = s64[4] parameter(2)
  send_sizes = s64[4] parameter(3)
  output_offsets = s64[4] parameter(4)
  recv_sizes = s64[4] parameter(5)
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
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: dynamic-slice
    // CHECK: reshape
    // CHECK: concatenate
    // CHECK: (bf16[1,16,8,32]{3,2,1,0}, bf16[1,16,8,32]{3,2,1,0}, bf16[1,16,8,32]{3,2,1,0}, bf16[1,16,8,32]{3,2,1,0}) all-to-all
    // CHECK: dynamic-update-slice
    // CHECK: iota
    // CHECK: compare
    // CHECK: select
    // CHECK: select
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest, OffsetsAndSizesNotS64AreRejected) {
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
  absl::StatusOr<bool> status = decomposer.Run(module.get(), {});
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.status().message(),
      HasSubstr("RaggedAllToAllDecomposer only supports S64 offsets. Was "
                "`ragged-all-to-all-canonicalizer` pass executed?"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
