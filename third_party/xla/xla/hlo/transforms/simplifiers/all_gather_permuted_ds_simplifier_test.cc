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

#include "xla/hlo/transforms/simplifiers/all_gather_permuted_ds_simplifier.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using ::testing::UnorderedElementsAreArray;

class AllGatherPermutedDsSimplifierTest
    : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, int64_t num_replicas,
      int64_t num_partitions, bool expect_change) const {
    HloModuleConfig config =
        GetModuleConfigForTest(num_replicas, num_partitions);
    config.set_use_spmd_partitioning(num_partitions > 1);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo_module, config));
    absl::StatusOr<bool> changed =
        AllGatherDynamicSlicePermutedOffsetSimplifier().Run(module.get(), {});
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(*changed, expect_change);
    return module;
  }
};

TEST_F(AllGatherPermutedDsSimplifierTest,
       AllPartitionsPermutedSingleReplicaGroup) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      p = f32[32,8,128] parameter(0)
      ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
        dimensions={0}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuteed_idx_list = s32[8]{0} constant({224,192,160,128,96,64,32,0})
      offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, zero, zero),
        dynamic_slice_sizes={32,8,128}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/1,
                                  /*num_partitions=*/8,
                                  /*expect_change=*/true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(
      root->source_target_pairs(),
      (UnorderedElementsAreArray<std::pair<int64_t, int64_t>>(
          {{0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0}})));
}

TEST_F(AllGatherPermutedDsSimplifierTest,
       AllPartitionsPermutedSingleReplicaGroupAlmostMatch) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
    p = f32[32,8,128] parameter(0)
    ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
      dimensions={0}, channel_id=1, use_global_device_ids=true
    pid = u32[] partition-id()
    permuteed_idx_list = s32[8]{0} constant({0,64,32,96,128,160,192,224})
    offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
      dynamic_slice_sizes={1}
    offset_reshape = s32[] reshape(offset)
    zero = s32[] constant(0)
    ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, zero, zero),
      dynamic_slice_sizes={32,8,128}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/1,
                                  /*num_partitions=*/8,
                                  /*expect_change=*/true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(root->source_target_pairs(),
              (UnorderedElementsAreArray<std::pair<int64_t, int64_t>>(
                  {{1, 2}, {2, 1}})));
}

TEST_F(AllGatherPermutedDsSimplifierTest,
       AllPartitionsPermutedMultipleReplicaGroups) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      p = f32[1,4,32] parameter(0)
      ag = f32[1,16,32] all-gather(p), replica_groups={{0,1,2,3},{4,5,6,7}},
        dimensions={1}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuteed_idx_list = s32[8]{0} constant({12,8,4,0,8,4,12,0})
      offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[1,4,32] dynamic-slice(ag, zero, offset_reshape, zero),
        dynamic_slice_sizes={1,4,32}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/1,
                                  /*num_partitions=*/8,
                                  /*expect_change=*/true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(root->source_target_pairs(),
              (UnorderedElementsAreArray<std::pair<int64_t, int64_t>>(
                  {{0, 3}, {1, 2}, {2, 1}, {3, 0}, {4, 7}, {6, 4}, {7, 6}})));
}

TEST_F(AllGatherPermutedDsSimplifierTest,
       AllPartitionsPermutedMultipleRgsWithMultiply) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      p = f32[1,4,32] parameter(0)
      ag = f32[1,16,32] all-gather(p), replica_groups={{0,1,2,3},{4,5,6,7}},
        dimensions={1}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuteed_idx_list = s32[8]{0} constant({3,2,1,0,3,2,1,0})
      offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
        dynamic_slice_sizes={1}
      multiplier = s32[1]{0} constant(4)
      offset_mul = s32[1]{0} multiply(offset, multiplier)
      offset_reshape = s32[] reshape(offset_mul)
      zero = s32[] constant(0)
      ROOT ds = f32[1,4,32] dynamic-slice(ag, zero, offset_reshape, zero),
        dynamic_slice_sizes={1,4,32}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/1,
                                  /*num_partitions=*/8,
                                  /*expect_change=*/true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CollectivePermute(op::Parameter(0)));
  EXPECT_THAT(
      root->source_target_pairs(),
      (UnorderedElementsAreArray<std::pair<int64_t, int64_t>>(
          {{0, 3}, {1, 2}, {2, 1}, {3, 0}, {4, 7}, {5, 6}, {6, 5}, {7, 4}})));
}

TEST_F(AllGatherPermutedDsSimplifierTest,
       NoChangeWhenUseGlobalDeviceIdsIsFalse) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      p = f32[32,8,128] parameter(0)
      ag = f32[256,8,128] all-gather(p), replica_groups={}, dimensions={0},
        channel_id=1
      pid = u32[] partition-id()
      permuteed_idx_list = s32[8]{0} constant({224,192,160,128,96, 64,32,0})
      offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, zero, zero),
        dynamic_slice_sizes={32,8,128}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/1,
                                  /*num_partitions=*/8,
                                  /*expect_change=*/false));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::AllGather(op::Parameter(0)), _, _, _));
}

TEST_F(AllGatherPermutedDsSimplifierTest, NoChangeWithMultipleReplicas) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      p = f32[32,8,128] parameter(0)
      ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
        dimensions={0}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuteed_idx_list = s32[8]{0} constant({224,192,160,128,96, 64,32,0})
      offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, zero, zero),
        dynamic_slice_sizes={32,8,128}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/8,
                                  /*num_partitions=*/1,
                                  /*expect_change=*/false));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::AllGather(op::Parameter(0)), _, _, _));
}

TEST_F(AllGatherPermutedDsSimplifierTest, NoChangeWhenSliceDimMismatch) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      p = f32[32,8,128] parameter(0)
      ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
        dimensions={0}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuteed_idx_list = s32[8]{0} constant({0,0,0,0,0,0,0,0})
      offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
       dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      static_offset = s32[] constant(224)
      ROOT ds = f32[32,8,128] dynamic-slice(ag, static_offset,
        offset_reshape, zero),
        dynamic_slice_sizes={32,8,128}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/1,
                                  /*num_partitions=*/8,
                                  /*expect_change=*/false));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::AllGather(op::Parameter(0)), _, _, _));
}

TEST_F(AllGatherPermutedDsSimplifierTest, NoChangeWhenShapeMismatch) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      p = f32[32,8,128] parameter(0)
      ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
        dimensions={0}, channel_id=1, use_global_device_ids=true
      pid = u32[] partition-id()
      permuteed_idx_list = s32[8]{0} constant({224,192,160,128,96, 64,32,0})
      offset = s32[1] dynamic-slice(permuteed_idx_list, pid),
        dynamic_slice_sizes={1}
      offset_reshape = s32[] reshape(offset)
      zero = s32[] constant(0)
      ROOT ds = f32[16,8,128] dynamic-slice(ag, offset_reshape, zero, zero),
        dynamic_slice_sizes={16,8,128}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          RunPass(hlo_string,
                                  /*num_replicas=*/1,
                                  /*num_partitions=*/8,
                                  /*expect_change=*/false));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::AllGather(op::Parameter(0)), _, _, _));
}

}  // namespace
}  // namespace xla
