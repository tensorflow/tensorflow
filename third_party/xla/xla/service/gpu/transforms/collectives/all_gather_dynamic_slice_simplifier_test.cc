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

#include "xla/service/gpu/transforms/collectives/all_gather_dynamic_slice_simplifier.h"

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
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class AllGatherDynamicSliceSimplifierTest
    : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, int64_t num_replicas,
      int64_t num_partitions, bool expect_change,
      bool allow_multiple_users = false) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/num_replicas,
        /*num_partitions=*/num_partitions);
    config.set_use_spmd_partitioning(num_partitions > 1);
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));
    AllGatherDynamicSliceSimplifier::Config pass_config;
    pass_config.allow_multiple_users = allow_multiple_users;
    auto changed =
        AllGatherDynamicSliceSimplifier(pass_config).Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return module;
  }
};

// Test cancellation of all-gather followed by dynamic-slice across all
// partitions.
TEST_F(AllGatherDynamicSliceSimplifierTest, AllPartitions) {
  absl::string_view hlo_string = R"(
  HloModule AllGather

  ENTRY %AllGather {
    %param = f32[32,8,128]{2,1,0} parameter(0)
    %ag = f32[256,8,128]{2,1,0} all-gather(%param), replica_groups={{0,1,2,3,4,5,6,7}},
      dimensions={0}, channel_id=1, use_global_device_ids=true
    %pid = u32[] partition-id()
    %pid_s32 = s32[] convert(%pid)
    %slice_size = s32[] constant(32)
    %offset = s32[] multiply(%pid_s32, %slice_size)
    %zero = s32[] constant(0)
    ROOT %ds = f32[32,8,128]{2,1,0} dynamic-slice(%ag, %offset, %zero, %zero),
      dynamic_slice_sizes={32,8,128}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Parameter(0));
}

// Test cancellation of all-gather followed by dynamic-slice across all replicas
// with reshape.
TEST_F(AllGatherDynamicSliceSimplifierTest, AllReplicasWithReshape) {
  absl::string_view hlo_string = R"(
   HloModule AllGather

   ENTRY %AllGather {
    %param = f32[32,8,128]{2,1,0} parameter(0)
    %ag = f32[256,8,128]{2,1,0} all-gather(%param), replica_groups={{0,1,2,3,4,5,6,7}},
      dimensions={0}, channel_id=1, use_global_device_ids=true
    %reshape = f32[256,8,64,2]{3,2,1,0} reshape(%ag)
    %pid = u32[] partition-id()
    %pid_s32 = s32[] convert(%pid)
    %slice_size = s32[] constant(32)
    %offset = s32[] multiply(%pid_s32, %slice_size)
    %zero = s32[] constant(0)
    ROOT %ds = f32[32,8,64,2]{3,2,1,0} dynamic-slice(%reshape, %offset, %zero, %zero, %zero),
      dynamic_slice_sizes={32,8,64,2}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(op::Parameter(0)));
}

// Test no cancellation when reshape is on the slice dimension.
TEST_F(AllGatherDynamicSliceSimplifierTest,
       AllPartitionsWithReshapeOnSliceDim) {
  absl::string_view hlo_string = R"(
  HloModule AllGather

  ENTRY %AllGather {
    %param = f32[32,8,128]{2,1,0} parameter(0)
    %ag = f32[256,8,128]{2,1,0} all-gather(%param), replica_groups={{0,1,2,3,4,5,6,7}},
      dimensions={0}, channel_id=1, use_global_device_ids=true
    %reshape = f32[2048,128]{1,0} reshape(%ag)
    %pid = u32[] partition-id()
    %pid_s32 = s32[] convert(%pid)
    %slice_size = s32[] constant(256)
    %offset = s32[] multiply(%pid_s32, %slice_size)
    %zero = s32[] constant(0)
    ROOT %ds = f32[256,128]{1,0} dynamic-slice(%reshape, %offset, %zero),
      dynamic_slice_sizes={256,128}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(
                  op::Reshape(op::AllGather(op::Parameter(0))),
                  op::Multiply(op::Convert(op::PartitionId()), op::Constant()),
                  op::Constant()));
}

// Test no cancellation when there is no all-gather.
TEST_F(AllGatherDynamicSliceSimplifierTest, NoAllGather) {
  absl::string_view hlo_string = R"(
  HloModule NoAllGather

  ENTRY %NoAllGather {
    %param = f32[32,8,128]{2,1,0} parameter(0)
    %pid = u32[] partition-id()
    %pid_s32 = s32[] convert(%pid)
    %slice_size = s32[] constant(32)
    %offset = s32[] multiply(%pid_s32, %slice_size)
    %zero = s32[] constant(0)
    ROOT %ds = f32[32,8,128]{2,1,0} dynamic-slice(%param, %offset, %zero, %zero),
      dynamic_slice_sizes={32,8,128}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(
                  op::Parameter(0),
                  op::Multiply(op::Convert(op::PartitionId()), op::Constant()),
                  op::Constant(), op::Constant()));
}

// Test no cancellation when the all-gather dimension is incorrect.
TEST_F(AllGatherDynamicSliceSimplifierTest, IncorrectAllGatherDimension) {
  absl::string_view hlo_string = R"(
  HloModule IncorrectAllGatherDimension

  ENTRY %IncorrectAllGatherDimension {
    %param = f32[32,8,128]{2,1,0} parameter(0)
    %ag = f32[32,64,128]{2,1,0} all-gather(%param), replica_groups={},
      dimensions={1}, channel_id=1
    %pid = u32[] partition-id()
    %pid_s32 = s32[] convert(%pid)
    %slice_size = s32[] constant(8)
    %offset = s32[] multiply(%pid_s32, %slice_size)
    %zero = s32[] constant(0)
    ROOT %ds = f32[32,8,128]{2,1,0} dynamic-slice(%ag, %zero, %offset, %zero),
      dynamic_slice_sizes={32,8,128}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(
                  op::AllGather(op::Parameter(0)), op::Constant(),
                  op::Multiply(op::Convert(op::PartitionId()), op::Constant()),
                  op::Constant()));
}

TEST_F(AllGatherDynamicSliceSimplifierTest,
       AllGatherDimDoesNotMatchDynamicSlice) {
  absl::string_view hlo_string = R"(
  HloModule m

  ENTRY root {
    param = f32[2,16] parameter(0)
    ag = f32[16,16] all-gather(%param), dimensions={0}
    pid = u32[] partition-id()
    pid_s32 = s32[] convert(%pid)
    slice_size = s32[] constant(2)
    offset = s32[] multiply(%pid_s32, %slice_size)
    zero = s32[] constant(0)
    ROOT _ = f32[16,2] dynamic-slice(ag, zero, offset),
      dynamic_slice_sizes={16,2}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

// Test cancellation of all-gather followed by dynamic-slice across all replicas
// with reshape and multiple users of the all-gather.
TEST_F(AllGatherDynamicSliceSimplifierTest,
       AllReplicasWithReshapeMultipleUsers) {
  absl::string_view hlo_string = R"(
  HloModule AllGather

  ENTRY %AllGather {
    %param = f32[32,8,128]{2,1,0} parameter(0)
    %ag = f32[256,8,128]{2,1,0} all-gather(%param), replica_groups={{0,1,2,3,4,5,6,7}},
      dimensions={0}, channel_id=1, use_global_device_ids=true
    %reshape = f32[256,8,64,2]{3,2,1,0} reshape(%ag)
    %pid = u32[] partition-id()
    %pid_s32 = s32[] convert(%pid)
    %slice_size = s32[] constant(32)
    %offset = s32[] multiply(%pid_s32, %slice_size)
    %zero = s32[] constant(0)
    %ds = f32[32,8,64,2]{3,2,1,0} dynamic-slice(%reshape, %offset, %zero, %zero, %zero),
      dynamic_slice_sizes={32,8,64,2}
    ROOT %tuple = (f32[32,8,64,2]{3,2,1,0}, f32[256,8,128]{2,1,0}) tuple(%ds, %ag)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true,
                                               /*allow_multiple_users=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Reshape(op::Parameter(0)),
                        op::AllGather(op::Parameter(0))));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
