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

#include "xla/service/gpu/collective_permute_cycle_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using CollectivePermuteCycleDecomposerTest = HloTestBase;

using ::testing::HasSubstr;
using CollectivePermuteDecomposerTest = HloTestBase;

TEST_F(CollectivePermuteDecomposerTest, DefaultChannelNotTransformed) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] replica-id()
        ROOT start = u32[] collective-permute(p),
          source_target_pairs={{0,1},{1,0}}
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteCycleDecomposerTest, TrivialNotTransformed) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] partition-id()
        ROOT start = u32[] collective-permute(p), channel_id=1,
          source_target_pairs={{0,0}}
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteCycleDecomposerTest, BelowThresholdNotTransformed) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] partition-id()
        ROOT start = u32[] collective-permute(p), channel_id=1,
          source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/33);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteCycleDecomposerTest, ForwardCycle) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] partition-id()
        ROOT start = u32[3,2] collective-permute(p), channel_id=1,
          source_target_pairs={{0,1},{1,2},{2,3},{3,0}},
          frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"},
          metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);

  auto check_metadata = [](const HloInstruction* inst) {
    EXPECT_EQ(inst->metadata().op_name(), "op1/op2/add");
    EXPECT_EQ(inst->metadata().source_file(), "foo/bar/mysource.py");
    EXPECT_EQ(inst->metadata().source_line(), 35);
  };

  HloCollectivePermuteInstruction* cp1 =
      DynCast<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), "collective-permute"));
  HloCollectivePermuteInstruction* cp2 =
      DynCast<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), "collective-permute.1"));
  EXPECT_NE(cp1, nullptr);
  EXPECT_NE(cp2, nullptr);
  EXPECT_EQ(cp1->operand(0), cp2->operand(0));
  EXPECT_GT(cp2->channel_id().value(), cp1->channel_id().value());
  EXPECT_THAT(cp1->ToString(), HasSubstr("source_target_pairs={{3,0}}"));
  EXPECT_THAT(cp1->ToString(),
              HasSubstr("_xla_send_recv_validation=\"{{3,10}}\""));
  EXPECT_THAT(cp2->ToString(),
              HasSubstr("source_target_pairs={{0,1},{1,2},{2,3}}"));
  EXPECT_THAT(cp2->ToString(),
              HasSubstr("_xla_send_recv_validation=\"{{0,7},{1,8},{2,9}}\""));
  check_metadata(cp1);
  check_metadata(cp2);
}

TEST_F(CollectivePermuteCycleDecomposerTest, BackwardCycle) {
  const absl::string_view kModuleStr = R"(
      HloModule test
      ENTRY test_computation {
        p = u32[] partition-id()
        ROOT start = u32[] collective-permute(p), channel_id=1,
          source_target_pairs={{0,3},{1,0},{2,1},{3,2}},
          frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"},
          metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py" source_line=35}
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kModuleStr)));
  CollectivePermuteCycleDecomposer decomposer(/*threshold_in_bytes=*/0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  auto check_metadata = [](const HloInstruction* inst) {
    EXPECT_EQ(inst->metadata().op_name(), "op1/op2/add");
    EXPECT_EQ(inst->metadata().source_file(), "foo/bar/mysource.py");
    EXPECT_EQ(inst->metadata().source_line(), 35);
  };

  HloCollectivePermuteInstruction* cp1 =
      DynCast<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), "collective-permute"));
  HloCollectivePermuteInstruction* cp2 =
      DynCast<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), "collective-permute.1"));
  EXPECT_NE(cp1, nullptr);
  EXPECT_NE(cp2, nullptr);
  EXPECT_EQ(cp1->operand(0), cp2->operand(0));
  EXPECT_GT(cp2->channel_id().value(), cp1->channel_id().value());
  EXPECT_THAT(cp1->ToString(), HasSubstr("source_target_pairs={{0,3}}"));
  EXPECT_THAT(cp1->ToString(),
              HasSubstr("_xla_send_recv_validation=\"{{0,7}}\""));
  EXPECT_THAT(cp2->ToString(),
              HasSubstr("source_target_pairs={{1,0},{2,1},{3,2}}"));
  EXPECT_THAT(cp2->ToString(),
              HasSubstr("_xla_send_recv_validation=\"{{1,8},{2,9},{3,10}}\""));
  check_metadata(cp1);
  check_metadata(cp2);
}

}  // namespace
}  // namespace xla
