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

#include "xla/service/gpu/transforms/collective_select_folder.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using CollectiveSelectFolderTest = HloTestBase;

const char* kHLOTemplate = R"(
  HloModule test
  ENTRY circular_exchange {
    in_tpl = (f32[16], f32[16]) parameter(0)
    fwd_data = f32[16]{0} get-tuple-element(in_tpl), index=0
    bwd_data = f32[16]{0} get-tuple-element(in_tpl), index=1

    c_first_id = u32[] constant($first_id_constant)
    c_last_id = u32[] constant($last_id_constant)
    repl_id = u32[] replica-id()

    pred_first_id = pred[] compare(repl_id, c_first_id), direction=EQ
    is_first = pred[] broadcast(pred_first_id), dimensions={}

    pred_last_id = pred[] compare(repl_id, c_last_id), direction=EQ
    is_last = pred[] broadcast(pred_last_id), dimensions={}

    // select data to send (redundant!)
    data_snd = f32[16] select(is_last, bwd_data, fwd_data)

    bwd_data_rcv = f32[16] collective-permute(data_snd), channel_id=1, source_target_pairs=$backward_pairs
    fwd_data_rcv = f32[16] collective-permute(data_snd), channel_id=2, source_target_pairs=$forward_pairs
    ROOT data_rcv = f32[16] select(is_first, bwd_data_rcv, fwd_data_rcv)
  }
)";

void VerifyDirectDataFeed(HloModule* module) {
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kCollectivePermute);
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kCollectivePermute);
  EXPECT_THAT(root->operand(1)->operand(0)->name(), HasSubstr("bwd_data"))
      << "first collective permute should directly operate on bwd_data";
  EXPECT_THAT(root->operand(2)->operand(0)->name(), HasSubstr("fwd_data"))
      << "second collective permute should directly operate on fwd_data";
}

TEST_F(CollectiveSelectFolderTest, SimpleForwardCycle) {
  std::string hlo_string = absl::StrReplaceAll(
      kHLOTemplate, {{"$first_id_constant", "0"},
                     {"$last_id_constant", "3"},
                     {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                     {"$backward_pairs", "{{3,0}}"}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(CollectiveSelectFolder(), module.get()));
  EXPECT_TRUE(changed);
  VerifyDirectDataFeed(module.get());
}

TEST_F(CollectiveSelectFolderTest, SimpleBackwardCycle) {
  std::string hlo_string = absl::StrReplaceAll(
      kHLOTemplate, {{"$first_id_constant", "3"},
                     {"$last_id_constant", "0"},
                     {"$forward_pairs", "{{3,2},{2,1},{1,0}}"},
                     {"$backward_pairs", "{{0,3}}"}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(CollectiveSelectFolder(), module.get()));
  EXPECT_TRUE(changed);
  VerifyDirectDataFeed(module.get());
}

TEST_F(CollectiveSelectFolderTest, ForwardWithPartitionId) {
  std::string hlo_string = absl::StrReplaceAll(
      kHLOTemplate, {{"$first_id_constant", "0"},
                     {"$last_id_constant", "3"},
                     {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                     {"$backward_pairs", "{{3,0}}"},
                     {"replica-id()", "partition-id()"}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(CollectiveSelectFolder(), module.get()));
  EXPECT_TRUE(changed);
  VerifyDirectDataFeed(module.get());
}

TEST_F(CollectiveSelectFolderTest, CompareNotEqual_NotTransformed) {
  std::string hlo_string = absl::StrReplaceAll(
      kHLOTemplate, {{"$first_id_constant", "0"},
                     {"$last_id_constant", "3"},
                     {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                     {"$backward_pairs", "{{3,0}}"},
                     {"direction=EQ", "direction=NE"}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(CollectiveSelectFolder(), module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveSelectFolderTest, LastDeviceIdMismatch_NotTransformed) {
  std::string hlo_string = absl::StrReplaceAll(
      kHLOTemplate, {{"$first_id_constant", "0"},
                     {"$last_id_constant", "2"},  // mismatch
                     {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                     {"$backward_pairs", "{{3,0}}"}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(CollectiveSelectFolder(), module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CollectiveSelectFolderTest, NotAReplicaOrPartition_NotTransformed) {
  std::string hlo_string = absl::StrReplaceAll(
      kHLOTemplate, {{"$first_id_constant", "0"},
                     {"$last_id_constant", "3"},
                     {"$forward_pairs", "{{0,1},{1,2},{2,3}}"},
                     {"$backward_pairs", "{{3,0}}"},
                     {"replica-id()", "constant(7)"}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(CollectiveSelectFolder(), module.get()));
  EXPECT_FALSE(changed);
}

// TODO (b/359348622) add a test case with just one
// collective-permute(select(arg))

}  // namespace
}  // namespace xla
