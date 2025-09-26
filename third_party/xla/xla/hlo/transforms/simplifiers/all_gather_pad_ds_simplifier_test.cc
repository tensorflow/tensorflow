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

#include "xla/hlo/transforms/simplifiers/all_gather_pad_ds_simplifier.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::ElementsAre;
using ::testing::Pair;

class AllGatherPadDsSimplifierTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, int64_t num_replicas,
      int64_t num_partitions, bool expect_change) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/num_replicas,
        /*num_partitions=*/num_partitions);
    config.set_use_spmd_partitioning(num_partitions > 1);
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));
    auto changed = AllGatherPadDsSimplifier().Run(module.get(), {});
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    LOG(INFO) << "new module: " << module->ToString();
    return module;
  }
};

TEST_F(AllGatherPadDsSimplifierTest, MultiReplicaGenericCaseLowPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Broadcast(op::Constant())),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root               // select
                                         ->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(5, 4), Pair(1, 0)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(6, 4), Pair(2, 0)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(7, 4), Pair(3, 0)));
}

TEST_F(AllGatherPadDsSimplifierTest,
       MultiReplicaGenericCaseLowPadMultiplyIndex) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_0

      // Get the ds offset on large pad.
      // const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      const.list = s32[8]{0} constant({0, 1, 2, 3, 0, 1, 2, 3})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      const.24 = s32[1]{0} constant({24})
      multiply = s32[1]{0} multiply(dynamic-slice, const.24)
      reshape.15 = s32[] reshape(multiply)

      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Broadcast(op::Constant())),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root               // select
                                         ->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(5, 4), Pair(1, 0)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(6, 4), Pair(2, 0)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(7, 4), Pair(3, 0)));
}

TEST_F(AllGatherPadDsSimplifierTest, 4ReplicaGenericCaseLowPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,4,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[4,2]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,24,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_20x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 6, 0, 6, 0, 6, 0, 6})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,6,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,6,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Broadcast(op::Constant())),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root               // select
                                         ->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 1, 0, 1, 0, 1, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(7, 6), Pair(5, 4), Pair(3, 2), Pair(1, 0)));
}

TEST_F(AllGatherPadDsSimplifierTest, MultiReplicaGenericCaseHighPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
            zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x88_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Select(op::Compare(op::Reshape(op::DynamicSlice(op::Constant(),
                                                          op::PartitionId())),
                             op::Constant()),
                 op::Concatenate(
                     op::Broadcast(op::Constant()),
                     op::CollectivePermute(op::Parameter(0)),  // {{4,7},{0,3}}
                     op::CollectivePermute(op::Parameter(0)),  // {{5,7},{1,3}}
                     op::CollectivePermute(op::Parameter(0)),  // {{6,7},{2,3}}
                     op::Parameter(0)),
                 op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({0, 0, 0, 1, 0, 0, 0, 1}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(4, 7), Pair(0, 3)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(5, 7), Pair(1, 3)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(6, 7), Pair(2, 3)));
}

TEST_F(AllGatherPadDsSimplifierTest, MultiReplicaGenericCaseLowPadNoGap) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,32,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_24x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 8, 16, 24, 0, 8, 16, 24})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,8,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,8,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0))),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(5, 4), Pair(1, 0)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(6, 4), Pair(2, 0)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(7, 4), Pair(3, 0)));
}

TEST_F(AllGatherPadDsSimplifierTest, SingleReplicaGenericCaseLowPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,128,40]{2,1,0} pad(all-gather, zero), padding=0_0x112_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 16, 32, 48, 64, 80 , 96, 112})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,16,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,16,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Parameter(0)),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({0, 0, 0, 0, 0, 0, 0, 1}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(0)->source_target_pairs(),
              ElementsAre(Pair(0, 7)));
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(1, 7)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(2, 7)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(3, 7)));
  EXPECT_THAT(concate->operand(4)->source_target_pairs(),
              ElementsAre(Pair(4, 7)));
  EXPECT_THAT(concate->operand(5)->source_target_pairs(),
              ElementsAre(Pair(5, 7)));
  EXPECT_THAT(concate->operand(6)->source_target_pairs(),
              ElementsAre(Pair(6, 7)));
}

TEST_F(AllGatherPadDsSimplifierTest, SingleReplicaGenericCaseLowPadUnordered) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,128,40]{2,1,0} pad(all-gather, zero), padding=0_0x112_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({16, 0, 32, 48, 64, 112 , 96, 80})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,16,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,16,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

TEST_F(AllGatherPadDsSimplifierTest, NotMatchInValidPadSmallPad) {
  absl::string_view hlo_string_small_pad = R"(
   HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,24,40]{2,1,0} pad(all-gather, zero), padding=0_0x8_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 3, 6, 9, 12, 15 , 18, 21})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,3,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,3,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string_small_pad,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

TEST_F(AllGatherPadDsSimplifierTest, NotMatchInValidPadInteriorPad) {
  absl::string_view hlo_string_interior_padding = R"(
   HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,76,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_0_4x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 16, 32, 48, 64, 80 , 96, 112})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,16,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,16,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string_interior_padding,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

TEST_F(AllGatherPadDsSimplifierTest, NotMatchShortSlice) {
  absl::string_view hlo_string = R"(
   HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,128,40]{2,1,0} pad(all-gather, zero), padding=0_0x112_0x0_0

      // Slice on two all-gather result.
      const.list = s32[8]{0} constant({0, 8, 16, 24, 32, 40 , 48, 56})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,8,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,8,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

class ExtractValidPadSpecTest : public HloHardwareIndependentTestBase {};

// Test case for ExtractValidPadSpec where the padding is valid.
// The padding is on the high side of the split dimension.
TEST_F(ExtractValidPadSpecTest, SuccessHighPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result on the high side of dimension 1.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_0

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  ASSERT_TRUE(pad_spec.has_value());
  EXPECT_EQ(pad_spec->split_dim, 1);
  EXPECT_EQ(pad_spec->start_offset, 8);
  EXPECT_EQ(pad_spec->end_offset, 96);
}

// Test case for ExtractValidPadSpec where the padding is on multiple
// dimensions. This is not a valid padding spec.
TEST_F(ExtractValidPadSpecTest, PadOnMultipleDims) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result on dims 1 and 2.
      pad = f64[1,96,41]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_1

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  EXPECT_FALSE(pad_spec.has_value());
}

// Test case for ExtractValidPadSpec where the padding is on both sides of the
// split dimension. This is not a valid padding spec.
TEST_F(ExtractValidPadSpecTest, PadOnBothSides) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result on both sides of dim 1.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x1_87x0_0

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  EXPECT_FALSE(pad_spec.has_value());
}

// Test case for ExtractValidPadSpec where the padding has interior padding.
// This is not a valid padding spec.
TEST_F(ExtractValidPadSpecTest, InteriorPadding) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result with interior padding on dim 1.
      pad = f64[1,103,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88_1x0_0

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  EXPECT_FALSE(pad_spec.has_value());
}
class GetPartitionIdForOffsetTest : public HloHardwareIndependentTestBase {};

TEST_F(GetPartitionIdForOffsetTest, InRangeFunctionality) {
  OffsetToIdMap offset_to_partition_map = {{0, 0}, {24, 1}, {48, 2}, {72, 3}};

  auto result = GetPartitionIdForOffset(offset_to_partition_map, 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 0);
  EXPECT_EQ(result.value()->second, 0);

  result = GetPartitionIdForOffset(offset_to_partition_map, -1);
  ASSERT_FALSE(result.has_value());

  result = GetPartitionIdForOffset(offset_to_partition_map, 23);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 0);
  EXPECT_EQ(result.value()->second, 0);

  result = GetPartitionIdForOffset(offset_to_partition_map, 24);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 24);
  EXPECT_EQ(result.value()->second, 1);

  result = GetPartitionIdForOffset(offset_to_partition_map, 47);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 24);
  EXPECT_EQ(result.value()->second, 1);

  result = GetPartitionIdForOffset(offset_to_partition_map, 48);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 48);
  EXPECT_EQ(result.value()->second, 2);

  result = GetPartitionIdForOffset(offset_to_partition_map, 72);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 72);
  EXPECT_EQ(result.value()->second, 3);

  result = GetPartitionIdForOffset(offset_to_partition_map, 75);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 72);
  EXPECT_EQ(result.value()->second, 3);
}

TEST_F(GetPartitionIdForOffsetTest, EmptyMap) {
  OffsetToIdMap offset_to_partition_map = {};
  EXPECT_EQ(GetPartitionIdForOffset(offset_to_partition_map, 24), std::nullopt);
}

TEST_F(GetPartitionIdForOffsetTest, OutOfRange) {
  OffsetToIdMap offset_to_partition_map = {{0, 0}, {24, 1}, {48, 2}, {72, 3}};
  EXPECT_EQ(GetPartitionIdForOffset(offset_to_partition_map, -1), std::nullopt);

  auto result = GetPartitionIdForOffset(offset_to_partition_map, 100);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 72);
  EXPECT_EQ(result.value()->second, 3);
}

class AddPredInstrBasedOnPartitionIdAndListTest
    : public HloHardwareIndependentTestBase {};

TEST_F(AddPredInstrBasedOnPartitionIdAndListTest, Basic) {
  HloModuleConfig config = GetModuleConfigForTest();
  auto module = std::make_unique<HloModule>(TestName(), config);
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {}), "dumb"));
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  std::vector<int64_t> select_list = {0, 0, 0, 1, 0, 0, 0, 1};
  HloInstruction* pred =
      AddPredInstrBasedOnPartitionIdAndList(computation, select_list);

  EXPECT_THAT(pred, op::Compare(op::Reshape(op::DynamicSlice(
                                    op::Constant(), op::PartitionId())),
                                op::Constant()));
  EXPECT_EQ(pred->shape().element_type(), PRED);
}

}  // namespace
}  // namespace xla
