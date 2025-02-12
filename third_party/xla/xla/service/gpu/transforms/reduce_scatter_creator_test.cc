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

#include "xla/service/gpu/transforms/reduce_scatter_creator.h"

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
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class GpuReduceScatterCreatorTest : public HloTestBase {
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
    auto changed = ReduceScatterCreator().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t AllReduceCount(std::unique_ptr<HloModule> &module) {
    return CollectiveCount(module, HloOpcode::kAllReduce);
  }

  size_t ReduceScatterCount(std::unique_ptr<HloModule> &module) {
    return CollectiveCount(module, HloOpcode::kAllReduce);
  }

 private:
  size_t CollectiveCount(std::unique_ptr<HloModule> &module, HloOpcode opcode) {
    return absl::c_count_if(
        module->entry_computation()->instructions(),
        [&opcode](HloInstruction *instr) { return instr->opcode() == opcode; });
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
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto *rs = Cast<HloReduceScatterInstruction>(
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
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto *rs = Cast<HloReduceScatterInstruction>(
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
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto *rs = Cast<HloReduceScatterInstruction>(
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
                                               /*expect_change=*/true));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceScatter(m::Parameter(0))));
  const auto *rs = Cast<HloReduceScatterInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(rs->backend_config<GpuBackendConfig>()
                  ->collective_backend_config()
                  .is_pipelined());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
