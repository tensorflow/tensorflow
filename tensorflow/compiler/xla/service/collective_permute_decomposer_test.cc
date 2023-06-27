/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/collective_permute_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
namespace op = xla::testing::opcode_matchers;
using CollectivePermuteDecomposerTest = HloTestBase;

TEST_F(CollectivePermuteDecomposerTest, SyncNotTransformed) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] replica-id()
        start = (u32[], u32[]) collective-permute-start(p),
          source_target_pairs={{0,1}, {1,2}},
          backend_config="{\"is_sync\":true}"
        ROOT done = u32[] collective-permute-done(start)
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, WithCycleNotTransformed) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = (u32[], u32[]) replica-id()
        start = u32[] collective-permute-start(p),
          source_target_pairs={{0,1}, {1,0}}
        ROOT done = u32[] collective-permute-done(start)
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, WithContextDataNotTransformed) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    start = (u32[], u32[], u32[], u32[]) collective-permute-start(p),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
    ROOT done = u32[] collective-permute-done(start)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteDecomposerTest, TransformedDefaultChannelId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    start = (u32[], u32[]) collective-permute-start(p),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
    ROOT done = u32[] collective-permute-done(start)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* after_all = FindInstruction(module.get(), "after-all");
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->operand(0), after_all);
  EXPECT_EQ(recv->channel_id().value(), 0);
  EXPECT_THAT(
      recv->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3},{3,4}}\""));
  HloInstruction* recv_done = FindInstruction(module.get(), "recv-done");
  EXPECT_EQ(recv_done->operand(0), recv);

  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_EQ(send->operand(1), after_all);
  EXPECT_EQ(send->control_predecessors()[0], recv);
  EXPECT_EQ(send->channel_id().value(), 0);
  EXPECT_THAT(
      send->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs=\"{{0,1},{1,2},{2,3},{3,4}}\""));
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  EXPECT_EQ(send_done->operand(0), send);
  EXPECT_EQ(send_done->control_predecessors()[0], recv_done);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(recv_done, 0));
}

TEST_F(CollectivePermuteDecomposerTest, TransformedExplicitChannelId) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[] replica-id()
    start = (u32[], u32[]) collective-permute-start(p), channel_id=2,
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}
    ROOT done = u32[] collective-permute-done(start)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* recv = FindInstruction(module.get(), "recv");
  EXPECT_EQ(recv->channel_id().value(), 2);
  HloInstruction* send = FindInstruction(module.get(), "send");
  EXPECT_EQ(send->channel_id().value(), 2);
}

}  // namespace
}  // namespace xla
