/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/reduce_scatter_creator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/algebraic_simplifier.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class GpuReduceScatterCreatorTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, int64_t num_replicas,
      int64_t num_partitions, bool use_spmd_partitioning, bool expect_change) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/num_replicas, /*num_partitions=*/num_partitions);
    config.set_use_spmd_partitioning(use_spmd_partitioning);
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));
    auto changed = ReduceScatterCreator().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t AllReduceCount(std::unique_ptr<HloModule>& module) {
    return CollectiveCount(module, HloOpcode::kAllReduce);
  }

  size_t ReduceScatterCount(std::unique_ptr<HloModule>& module) {
    return CollectiveCount(module, HloOpcode::kReduceScatter);
  }

  template <typename T>
  size_t AllReduceCount(std::unique_ptr<T>& module) {
    return CollectiveCount(module.get(), HloOpcode::kAllReduce);
  }

  template <typename T>
  size_t ReduceScatterCount(std::unique_ptr<T>& module) {
    return CollectiveCount(module.get(), HloOpcode::kReduceScatter);
  }

 private:
  size_t CollectiveCount(std::unique_ptr<HloModule>& module, HloOpcode opcode) {
    return absl::c_count_if(
        module->entry_computation()->instructions(),
        [&opcode](HloInstruction* instr) { return instr->opcode() == opcode; });
  }

  size_t CollectiveCount(HloModule* module, HloOpcode opcode) {
    return absl::c_count_if(
        module->entry_computation()->instructions(),
        [&opcode](HloInstruction* instr) { return instr->opcode() == opcode; });
  }
};

TEST_F(GpuReduceScatterCreatorTest, AllReplicas) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={}, to_apply=%sum
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
  %rid = u32[] replica-id()
  %id = s32[1] dynamic-slice(%table, %rid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(4)
  %offset = s32[] multiply(%reshape, %slice_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce, %offset, %zero, %zero),
    dynamic_slice_sizes={4,8,128}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_EQ(rs->scatter_dimension(), 0) << rs->ToString();
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, AllReplicasWithOffsetReshape) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={}, to_apply=%sum
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
  %rid = u32[] replica-id()
  %id = s32[1] dynamic-slice(%table, %rid), dynamic_slice_sizes={1}
  %slice_size = s32[1] constant({4})
  %offset = s32[1] multiply(%id, %slice_size)
  %reshape = s32[] reshape(%offset)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce, %reshape, %zero, %zero),
    dynamic_slice_sizes={4,8,128}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_EQ(rs->scatter_dimension(), 0) << rs->ToString();
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, AllReplicasWithReshape) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={}, to_apply=%sum
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
  %rid = u32[] replica-id()
  %id = s32[1] dynamic-slice(%table, %rid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(4)
  %offset = s32[] multiply(%reshape, %slice_size)
  %zero = s32[] constant(0)
  %reshape.1 = f32[32,16,64] reshape(%all-reduce)
  ROOT %dynamic-slice = f32[4,16,64] dynamic-slice(%reshape.1, %offset, %zero, %zero),
    dynamic_slice_sizes={4,16,64}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::ReduceScatter(m::Parameter(0)))));
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, AllReplicasWithReshapeSplitDimModified) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[336,1024] parameter(0)
  %all-reduce = f32[336,1024] all-reduce(%param), replica_groups={}, to_apply=%sum
  %rid = u32[] replica-id()
  %id = s32[] convert(%rid)
  %slice_size = s32[] constant(128)
  %offset = s32[] multiply(%id, %slice_size)
  %zero = s32[] constant(0)
  %reshape.1 = f32[4,84,1024] reshape(%all-reduce)
  ROOT %dynamic-slice = f32[4,84,128] dynamic-slice(%reshape.1, %zero, %zero, %offset),
    dynamic_slice_sizes={4,84,128}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::ReduceScatter(m::Parameter(0)))));
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, AllReplicasDim2) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={}, to_apply=%sum
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
  %rid = u32[] replica-id()
  %rid_s32 = s32[] convert(%rid)
  %slice_size = s32[] constant(16)
  %offset = s32[] multiply(%rid_s32, %slice_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[32,8,16] dynamic-slice(%all-reduce, %zero, %zero, %offset),
    dynamic_slice_sizes={32,8,16}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_EQ(rs->scatter_dimension(), 2) << rs->ToString();
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, AllReplicasWrongOffsets) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={}, to_apply=%sum
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,8})
  %rid = u32[] replica-id()
  %id = s32[1] dynamic-slice(%table, %rid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(4)
  %offset = s32[] multiply(%reshape, %slice_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce, %offset, %zero, %zero),
    dynamic_slice_sizes={4,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/false));
}

TEST_F(GpuReduceScatterCreatorTest, AllReplicasIotaTable) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={}, to_apply=%sum
  %table = s32[8]{0} iota(), iota_dimension=0
  %rid = u32[] replica-id()
  %id = s32[1] dynamic-slice(%table, %rid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(4)
  %offset = s32[] multiply(%reshape, %slice_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce, %offset, %zero, %zero),
    dynamic_slice_sizes={4,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/2,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, SubgroupedReplicas) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={{1,3,2,0},{4,5,6,7}}, to_apply=%sum
  %gtable = s32[8]{0} constant({3,0,2,1,0,1,2,3})
  %rid = u32[] replica-id()
  %id = s32[1] dynamic-slice(%gtable, %rid), dynamic_slice_sizes={1}
  %reshape.0 = s32[] reshape(%id)
  %table = s32[4]{0} constant({0,8,16,24})
  %offset = s32[1] dynamic-slice(%table, %reshape.0), dynamic_slice_sizes={1}
  %reshape.1 = s32[] reshape(%offset)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[8,8,128] dynamic-slice(%all-reduce, %reshape.1, %zero, %zero),
    dynamic_slice_sizes={8,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/2,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, AllPartitions) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={{0},{1}}, to_apply=%sum, channel_id=1
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
  %pid = u32[] partition-id()
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(4)
  %offset = s32[] multiply(%reshape, %slice_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce, %offset, %zero, %zero),
    dynamic_slice_sizes={4,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/2,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, AllReduceFollowedByAllReduce) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce.scattered = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}}, to_apply=%sum, use_global_device_ids=true, channel_id=1
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
  %pid = u32[] partition-id()
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(4)
  %offset = s32[] multiply(%reshape, %slice_size)
  %zero = s32[] constant(0)
  %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce.scattered, %offset, %zero, %zero),
    dynamic_slice_sizes={4,8,128}
  ROOT %all-reduce.sync = f32[4,8,128]{2,1,0} all-reduce(%dynamic-slice),
    replica_groups={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}, to_apply=%sum, use_global_device_ids=true, channel_id=2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/2,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  EXPECT_EQ(AllReduceCount(module), 1);
  EXPECT_EQ(ReduceScatterCount(module), 1);
}

TEST_F(GpuReduceScatterCreatorTest, SubgroupsGlobals) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={{1,3,2,0},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %rid = u32[] replica-id()
  %pcount = u32[] constant(4)
  %ridxp = u32[] multiply(%rid, %pcount)
  %gid = u32[] add(%ridxp, %pid)
  %gtable = s32[8]{0} constant({3,0,2,1,0,1,2,3})
  %id = s32[1] dynamic-slice(%gtable, %gid), dynamic_slice_sizes={1}
  %reshape.0 = s32[] reshape(%id)
  %table = s32[4]{0} constant({0,8,16,24})
  %offset = s32[1] dynamic-slice(%table, %reshape.0), dynamic_slice_sizes={1}
  %reshape.1 = s32[] reshape(%offset)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[8,8,128] dynamic-slice(%all-reduce, %reshape.1, %zero, %zero),
    dynamic_slice_sizes={8,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/2,
                                               /*num_partitions=*/4,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, SubgroupsGlobalsOrthogonalReplicas) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={{1,3,2,0},{5,7,6,4}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %pid_table = s32[4]{0} constant({3,0,2,1})
  %offset = s32[1] dynamic-slice(%pid_table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%offset)
  %shard_size = s32[] constant(8)
  %mul = s32[] multiply(%reshape, %shard_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[8,8,128] dynamic-slice(%all-reduce, %mul, %zero, %zero),
    dynamic_slice_sizes={8,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/2,
                                               /*num_partitions=*/4,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  EXPECT_EQ(AllReduceCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, SubgroupsGlobalsNonOrthogonalReplicas) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={{1,3,2,0},{7,5,6,4}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %pid_table = s32[4]{0} constant({3,0,2,1})
  %offset = s32[1] dynamic-slice(%pid_table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%offset)
  %shard_size = s32[] constant(8)
  %mul = s32[] multiply(%reshape, %shard_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[8,8,128] dynamic-slice(%all-reduce, %mul, %zero, %zero),
    dynamic_slice_sizes={8,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/2,
                                               /*num_partitions=*/4,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/false));
}

TEST_F(GpuReduceScatterCreatorTest, NonUniformSplit) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[1,7]{1,0} parameter(0)
  %all-reduce = f32[1,7]{1,0} all-reduce(%param),
    replica_groups={{0,1},{2,3},{4,5},{6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %pid_table = s32[8]{0} constant({0, 1, 0, 1, 0, 1, 0, 1})
  %offset = s32[1] dynamic-slice(%pid_table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%offset)
  %shard_size = s32[] constant(3)
  %mul = s32[] multiply(%reshape, %shard_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[1,3] dynamic-slice(%all-reduce, %zero, %mul),
    dynamic_slice_sizes={1,3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Slice(m::Parameter(0)))));
}

TEST_F(GpuReduceScatterCreatorTest,
       ReduceScatterCreatorPreservesBackendConfig) {
  absl::string_view hlo_string = R"(
HloModule AllReduce

%sum {
%a = f32[] parameter(0)
%b = f32[] parameter(1)
ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
%param = f32[32,8,128]{2,1,0} parameter(0)
%all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
replica_groups={}, to_apply=%sum, backend_config={"collective_backend_config":{"is_pipelined":true}}
%table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
%rid = u32[] replica-id()
%id = s32[1] dynamic-slice(%table, %rid), dynamic_slice_sizes={1}
%reshape = s32[] reshape(%id)
%slice_size = s32[] constant(4)
%offset = s32[] multiply(%reshape, %slice_size)
%zero = s32[] constant(0)
ROOT %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce, %offset, %zero, %zero),
dynamic_slice_sizes={4,8,128}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(rs->backend_config<GpuBackendConfig>()
                  ->collective_backend_config()
                  .is_pipelined());
}

TEST_F(GpuReduceScatterCreatorTest,
       ReduceScatterCreatorWithSPMDPartitioningDisabled) {
  absl::string_view hlo_string = R"(
HloModule test

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
      replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=1,
      use_global_device_ids=true, to_apply=%sum
  %table = s32[8]{0} constant({0,1,2,3,4,5,6,7})
  %pid = u32[] partition-id()
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(4)
  %offset = s32[] multiply(%reshape, %slice_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,8,128] dynamic-slice(%all-reduce, %offset, %zero,
      %zero), dynamic_slice_sizes={4,8,128}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/false,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
}

TEST_F(GpuReduceScatterCreatorTest, SubtractionPatternWithTableLookup) {
  absl::string_view hlo_string = R"(
HloModule SubtractionPattern

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %SubtractionPattern {
  %param = f32[4,64,1024]{2,1,0} parameter(0)
  %all-reduce = f32[4,64,1024]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %table = s32[8]{0} constant({0, 0, 0, 0, 64, 64, 64, 64})
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(16)
  %pid_s32 = s32[] convert(%pid)
  %multiply = s32[] multiply(%pid_s32, %slice_size)
  %offset = s32[] subtract(%multiply, %reshape)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,16,1024] dynamic-slice(%all-reduce, %zero, %offset, %zero),
    dynamic_slice_sizes={4,16,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_EQ(rs->scatter_dimension(), 1) << rs->ToString();
  EXPECT_EQ(AllReduceCount(module), 0);
}

// Multiplier is clamped but table index is not, and we can't prove the clamp is
// no-op.
TEST_F(GpuReduceScatterCreatorTest,
       SubtractionPatternWithTableLookupMultiplierUnprovableNoopClamp) {
  absl::string_view hlo_string = R"(
HloModule SubtractionPatternClampNoop

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %SubtractionPatternClampNoop {
  %param = f32[4,64,1024]{2,1,0} parameter(0)
  %all-reduce = f32[4,64,1024]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %table = s32[8]{0} constant({0, 0, 0, 0, 64, 64, 64, 64})
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(16)
  %zero_u32 = u32[] constant(0)
  %max_u32 = u32[] constant(7)
  %pid_clamp = u32[] clamp(%zero_u32, %pid, %max_u32)
  %pid_s32 = s32[] convert(%pid_clamp)
  %multiply = s32[] multiply(%pid_s32, %slice_size)
  %offset = s32[] subtract(%multiply, %reshape)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,16,1024] dynamic-slice(%all-reduce, %zero, %offset, %zero),
    dynamic_slice_sizes={4,16,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/false));
  EXPECT_EQ(AllReduceCount(module), 1);
  EXPECT_EQ(ReduceScatterCount(module), 0);
}

// Table index is clamped but multiplier is not, and we can't prove the clamp is
// no-op.
TEST_F(GpuReduceScatterCreatorTest,
       SubtractionPatternWithTableLookupIndexUnprovableNoopClamp) {
  absl::string_view hlo_string = R"(
HloModule SubtractionPatternClampIndex

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %SubtractionPatternClampIndex {
  %param = f32[4,64,1024]{2,1,0} parameter(0)
  %all-reduce = f32[4,64,1024]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %pid_min = u32[] constant(0)
  %pid_max = u32[] constant(7)
  %pid_clamp = u32[] clamp(%pid_min, %pid, %pid_max)
  %table = s32[8]{0} constant({0, 0, 0, 0, 64, 64, 64, 64})
  %id = s32[1] dynamic-slice(%table, %pid_clamp), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(16)
  %pid_s32 = s32[] convert(%pid)
  %multiply = s32[] multiply(%pid_s32, %slice_size)
  %offset = s32[] subtract(%multiply, %reshape)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,16,1024] dynamic-slice(%all-reduce, %zero, %offset, %zero),
    dynamic_slice_sizes={4,16,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/false));
  EXPECT_EQ(AllReduceCount(module), 1);
  EXPECT_EQ(ReduceScatterCount(module), 0);
}

// Multiplier is clamped to [0,3], which is NOT a no-op for 8 partitions.
// Table index uses raw pid. Pattern should not match.
TEST_F(GpuReduceScatterCreatorTest, SubtractionPatternWithMultiplierClamped) {
  absl::string_view hlo_string = R"(
HloModule SubtractionPatternClampNonNoop

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %SubtractionPatternClampNonNoop {
  %param = f32[4,64,1024]{2,1,0} parameter(0)
  %all-reduce = f32[4,64,1024]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %pid_min = u32[] constant(0)
  %pid_max = u32[] constant(3)
  %pid_clamp = u32[] clamp(%pid_min, %pid, %pid_max)
  %table = s32[8]{0} constant({0, 0, 0, 0, 64, 64, 64, 64})
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(16)
  %pid_s32 = s32[] convert(%pid_clamp)
  %multiply = s32[] multiply(%pid_s32, %slice_size)
  %offset = s32[] subtract(%multiply, %reshape)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,16,1024] dynamic-slice(%all-reduce, %zero, %offset, %zero),
    dynamic_slice_sizes={4,16,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/false));
  EXPECT_EQ(AllReduceCount(module), 1);
  EXPECT_EQ(ReduceScatterCount(module), 0);
}

// Table index and multiplier have (different) noop chains, but ultimately are
// derived from the same value.
TEST_F(GpuReduceScatterCreatorTest,
       SubtractionPatternWithTableLookupNoopChains) {
  absl::string_view hlo_string = R"(
HloModule SubtractionPatternNoopChains

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %SubtractionPatternNoopChains {
  %param = f32[4,64,1024]{2,1,0} parameter(0)
  %all-reduce = f32[4,64,1024]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %table = s32[8]{0} constant({0, 0, 0, 0, 64, 64, 64, 64})
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %id_bitcast = s32[1] bitcast(%id)
  %reshape = s32[] reshape(%id_bitcast)
  %slice_size = s32[] constant(16)
  %pid_copy = u32[] copy(%pid)
  %pid_s32 = s32[] convert(%pid_copy)
  %multiply = s32[] multiply(%pid_s32, %slice_size)
  %offset = s32[] subtract(%multiply, %reshape)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,16,1024] dynamic-slice(%all-reduce, %zero, %offset, %zero),
    dynamic_slice_sizes={4,16,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_EQ(rs->scatter_dimension(), 1) << rs->ToString();
  EXPECT_EQ(AllReduceCount(module), 0);
}

// Global IDs wrap around the table for ND meshes.
TEST_F(GpuReduceScatterCreatorTest,
       SubtractionPatternWithTableLookupGlobalIdsModulo) {
  absl::string_view hlo_string = R"(
HloModule SubtractionPatternGlobalIdsModulo

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %SubtractionPatternGlobalIdsModulo {
  %param = f32[4,64,1024]{2,1,0} parameter(0)
  %all-reduce = f32[4,64,1024]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %table = s32[4]{0} constant({0, 0, 0, 0})
  %id = s32[1] dynamic-slice(%table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(16)
  %pid_s32 = s32[] convert(%pid)
  %multiply = s32[] multiply(%pid_s32, %slice_size)
  %offset = s32[] subtract(%multiply, %reshape)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,16,1024] dynamic-slice(%all-reduce, %zero, %offset, %zero),
    dynamic_slice_sizes={4,16,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/2,
                                               /*num_partitions=*/4,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_EQ(rs->scatter_dimension(), 1) << rs->ToString();
  EXPECT_EQ(AllReduceCount(module), 0);
}

// Manually computing global ID and using it directly produces wrong offsets
// for groups beyond the first one.
TEST_F(GpuReduceScatterCreatorTest,
       SubtractionPatternWithManualGlobalIdFalsePositive) {
  absl::string_view hlo_string = R"(
HloModule SubtractionPatternGlobalIdsModuloFalsePositive

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %SubtractionPatternGlobalIdsModuloFalsePositive {
  %param = f32[4,64,1024]{2,1,0} parameter(0)
  %all-reduce = f32[4,64,1024]{2,1,0} all-reduce(%param),
    replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %rid = u32[] replica-id()
  %pcount = u32[] constant(4)
  %ridxp = u32[] multiply(%rid, %pcount)
  %gid = u32[] add(%ridxp, %pid)
  %table = s32[8]{0} constant({0, 0, 0, 0, 0, 0, 0, 0})
  %id = s32[1] dynamic-slice(%table, %gid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%id)
  %slice_size = s32[] constant(16)
  %gid_s32 = s32[] convert(%gid)
  %multiply = s32[] multiply(%gid_s32, %slice_size)
  %offset = s32[] subtract(%multiply, %reshape)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[4,16,1024] dynamic-slice(%all-reduce, %zero, %offset, %zero),
    dynamic_slice_sizes={4,16,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/2,
                                               /*num_partitions=*/4,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/false));
  EXPECT_EQ(AllReduceCount(module), 1);
  EXPECT_EQ(ReduceScatterCount(module), 0);
}

TEST_F(GpuReduceScatterCreatorTest, ReproClampOnGroupLocalIndex) {
  absl::string_view hlo_string = R"(
HloModule reduce_scatter_repro, num_partitions=128

%add.17.clone (x.35: bf16[], y.35: bf16[]) -> bf16[] {
  %x.35 = bf16[] parameter(0)
  %y.35 = bf16[] parameter(1)
  ROOT %add.1629 = bf16[] add(%x.35, %y.35)
}

ENTRY main {
  input = bf16[1,8192,2560] parameter(0)

  %constant.1513 = s32[] constant(0)
  %constant.1710 = s32[8]{0} constant({0, 0, 0, 0, 8192, 8192, 8192, 8192})
  %partition-id.14 = u32[] partition-id()

  %all-reduce.365 = bf16[1,8192,2560]{2,1,0} all-reduce(input),
    channel_id=72, replica_groups=[32,4]<=[128], use_global_device_ids=true, to_apply=%add.17.clone

  %constant.1691 = u32[128]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7})
  %dynamic-slice.444 = u32[1]{0} dynamic-slice(%constant.1691, %partition-id.14), dynamic_slice_sizes={1}
  %reshape.2062 = u32[] reshape(%dynamic-slice.444)
  %dynamic-slice.448 = s32[1]{0} dynamic-slice(%constant.1710, %reshape.2062), dynamic_slice_sizes={1}
  %reshape.2090 = s32[] reshape(%dynamic-slice.448)

  %constant.1690 = u32[] constant(0)
  %constant.1692 = u32[] constant(7)
  %clamp.8 = u32[] clamp(%constant.1690, %reshape.2062, %constant.1692)
  %convert.7 = s32[] convert(%clamp.8)
  %constant.1693 = s32[] constant(2048)
  %multiply.260 = s32[] multiply(%convert.7, %constant.1693)

  %subtract.55 = s32[] subtract(%multiply.260, %reshape.2090)

  ROOT %dynamic-slice.456 = bf16[1,2048,2560]{2,1,0} dynamic-slice(%all-reduce.365, %constant.1513, %subtract.55, %constant.1513), dynamic_slice_sizes={1,2048,2560}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/128,
                                               /*use_spmd_partitioning=*/true,
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  EXPECT_EQ(AllReduceCount(module), 0);
  EXPECT_EQ(ReduceScatterCount(module), 1);
}

TEST_F(GpuReduceScatterCreatorTest, AllReduceThroughTuple) {
  absl::string_view hlo_string = R"(
HloModule AllReduceThroughTuple

%sum {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(%a, %b)
}

ENTRY %AllReduce {
  %param0 = f32[4096,4096]{1,0} parameter(0)
  %param1 = f32[1024,4096]{1,0} parameter(1)
  %all-reduce = f32[4096,4096]{1,0} all-reduce(%param0),
    replica_groups={{0,1,2,3,4,5,6,7}}, channel_id=5, use_global_device_ids=true, to_apply=%sum
  %tuple = (f32[4096,4096]{1,0}, f32[1024,4096]{1,0}) tuple(%all-reduce, %param1)
  %get-tuple-element = f32[4096,4096]{1,0} get-tuple-element(%tuple), index=0
  %pid = u32[] partition-id()
  %pid_s32 = s32[] convert(%pid)
  %slice_size = s32[] constant(512)
  %offset = s32[] multiply(%pid_s32, %slice_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[512,4096]{1,0} dynamic-slice(%get-tuple-element, %offset, %zero),
    dynamic_slice_sizes={512,4096}
}
)";

  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/1, /*num_partitions=*/8);
  config.set_use_spmd_partitioning(true);

  TF_ASSERT_OK_AND_ASSIGN(auto module_without_algsimp,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed_without,
      ReduceScatterCreator().Run(module_without_algsimp.get()));
  EXPECT_FALSE(changed_without) << "ReduceScatterCreator should not transform "
                                   "without AlgebraicSimplifier";
  EXPECT_EQ(AllReduceCount(module_without_algsimp), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto module_with_algsimp,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  AlgebraicSimplifierOptions options;
  se::GpuComputeCapability compute_capability{se::CudaComputeCapability{8, 0}};
  GpuAlgebraicSimplifier algsimp(options, compute_capability);
  TF_ASSERT_OK_AND_ASSIGN(bool algsimp_changed,
                          algsimp.Run(module_with_algsimp.get(), {}));
  EXPECT_TRUE(algsimp_changed);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed_with, ReduceScatterCreator().Run(module_with_algsimp.get()));
  EXPECT_TRUE(changed_with)
      << "ReduceScatterCreator should transform after AlgebraicSimplifier";
  EXPECT_GE(ReduceScatterCount(module_with_algsimp), 1)
      << "Expected at least one ReduceScatter after transformation";
}

}  // namespace
}  // namespace gpu
}  // namespace xla
