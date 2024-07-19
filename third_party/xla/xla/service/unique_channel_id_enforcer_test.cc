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

#include "xla/service/unique_channel_id_enforcer.h"

#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using UniqueChannelIdEnforcerTest = HloTestBase;

TEST_F(UniqueChannelIdEnforcerTest, EnsureUniqueChannelIdsAllGather) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  allgather0 = f32[32] all-gather(param0), channel_id=1, replica_groups={}, dimensions={0}
  allgather1 = f32[32] all-gather(param1), channel_id=1, replica_groups={}, dimensions={0}
  ROOT tuple = (f32[32], f32[32]) tuple(allgather0, allgather1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  UniqueChannelIdEnforcer enforcer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, enforcer.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify that channel IDs are unique for all-gather ops
  std::optional<int64_t> all_gather1_channel_id;
  std::optional<int64_t> all_gather2_channel_id;

  for (HloInstruction* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kAllGather) {
      if (!all_gather1_channel_id.has_value()) {
        all_gather1_channel_id = inst->channel_id();
      } else {
        all_gather2_channel_id = inst->channel_id();
      }
    }
  }

  ASSERT_TRUE(all_gather1_channel_id.has_value());
  ASSERT_TRUE(all_gather2_channel_id.has_value());
  EXPECT_NE(all_gather1_channel_id.value(), all_gather2_channel_id.value());
}

TEST_F(UniqueChannelIdEnforcerTest, ChannelIdsAlreadyUnique) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  allgather0 = f32[32] all-gather(param0), channel_id=1, replica_groups={}, dimensions={0}
  allgather1 = f32[32] all-gather(param1), channel_id=2, replica_groups={}, dimensions={0}
  ROOT tuple = (f32[32], f32[32]) tuple(allgather0, allgather1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  UniqueChannelIdEnforcer enforcer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, enforcer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(UniqueChannelIdEnforcerTest, DuplicateChannelIdsAssertTrue) {
  const char* const hlo_string = R"(
    HloModule Module

    ENTRY entry {
      param0 = f32[8] parameter(0)
      param1 = f32[8] parameter(1)
      allgather0 = f32[32] all-gather(param0), channel_id=1, replica_groups={}, dimensions={0}
      allgather1 = f32[32] all-gather(param1), channel_id=1, replica_groups={}, dimensions={0}
      ROOT tuple = (f32[32], f32[32]) tuple(allgather0, allgather1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  UniqueChannelIdEnforcer enforcer(/*assert_unique_channel_ids=*/true);
  auto status_or_changed = enforcer.Run(module.get());

  EXPECT_FALSE(status_or_changed.ok());
}

}  // namespace
}  // namespace xla
