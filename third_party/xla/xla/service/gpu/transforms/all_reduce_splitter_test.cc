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

#include "xla/service/gpu/transforms/all_reduce_splitter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/transforms/reduce_scatter_creator.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class AllReduceSplitterTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> PrepareModule(
      absl::string_view hlo_module, int64_t num_replicas,
      int64_t num_partitions) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/num_replicas,
        /*num_partitions=*/num_partitions);
    config.set_use_spmd_partitioning(num_partitions > 1);
    return ParseAndReturnVerifiedModule(hlo_module, config);
  }

  size_t AllReduceCount(const HloModule &module) {
    return CollectiveCount(module, HloOpcode::kAllReduce);
  }

 private:
  size_t CollectiveCount(const HloModule &module, HloOpcode opcode) {
    return absl::c_count_if(
        module.entry_computation()->instructions(),
        [&opcode](HloInstruction *instr) { return instr->opcode() == opcode; });
  }
};

class AllReduceSplitterFilecheckTest : public AllReduceSplitterTest {
 public:
  absl::Status FileCheck(const std::string &hlo_text,
                         absl::string_view pattern) {
    TF_ASSIGN_OR_RETURN(bool matched, RunFileCheck(hlo_text, pattern));
    if (!matched) {
      return absl::InternalError("Filecheck failed.");
    }
    return absl::OkStatus();
  }
};

TEST_F(
    AllReduceSplitterFilecheckTest,
    MatchBasicPatternIfDynamicSliceIsRootAndThereExistsAllReduceWithSameReplicaGroups) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(first.ar, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  ROOT _ = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(true));
  TF_EXPECT_OK(FileCheck(module->ToString(), R"(
    CHECK-DAG:    %[[P0:.*]] = bf16[2,4096,4096]{2,1,0} parameter(0)
    CHECK:        %[[AR0:.*]] = bf16[2,4096,4096]{2,1,0} all-reduce(%[[P0]])
    CHECK-SAME:   replica_groups={[[DESIRED_RGS:.*]]}
    CHECK-DAG:    %[[ZERO:.*]] = bf16[] constant(0)
    CHECK-DAG:    %[[LOCAL_REDUCE:.*]] = bf16[4096]{0} reduce(%[[AR0]], %[[ZERO]])
    CHECK:        %[[AR1:.*]] = bf16[4096]{0} all-reduce(%[[LOCAL_REDUCE]])
    CHECK-SAME:   replica_groups={[[DESIRED_RGS]]}
    CHECK:        %[[DS:.*]] = bf16[1024]{0} dynamic-slice(%[[AR1]], %[[_:.*]])
    CHECK-SAME:   dynamic_slice_sizes={1024}
    CHECK-NEXT:   ROOT %[[AR2:.*]] = bf16[1024]{0} all-reduce(%[[DS]])
    CHECK-SAME:   replica_groups={{[{]}}{0,4},{1,5},{2,6},{3,7}{{[}]}}
    )"));
}

TEST_F(
    AllReduceSplitterTest,
    DoesNotMatchMatchBasicPatternIfDynamicSliceIsRootAndThereIsNoAllReduceWithSameReplicaGroups) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(p, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  ROOT _ = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(false));

  EXPECT_EQ(AllReduceCount(*module), 1);
}

TEST_F(
    AllReduceSplitterFilecheckTest,
    MatchBasicPatternIfDynamicSliceIsNotRootAndThereExistsAllReduceWithSameReplicaGroups) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  zero = bf16[] constant(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  reduce = bf16[4096] reduce(p, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  dynamic_slice = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
  broadcast = bf16[1024,1024] broadcast(dynamic_slice), dimensions={0}
  ROOT _ = tuple(broadcast, first.ar)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(true));
  TF_EXPECT_OK(FileCheck(module->ToString(), R"(
    CHECK-DAG:    %[[P0:.*]] = bf16[2,4096,4096]{2,1,0} parameter(0)
    CHECK-DAG:    %[[ZERO:.*]] = bf16[] constant(0)
    CHECK-DAG:    %[[LOCAL_REDUCE:.*]] = bf16[4096]{0} reduce(%[[P0]], %[[ZERO]])
    CHECK:        %[[AR0:.*]] = bf16[4096]{0} all-reduce(%[[LOCAL_REDUCE]])
    CHECK-SAME:   replica_groups={[[DESIRED_RGS:.*]]}
    CHECK:        %[[DS:.*]] = bf16[1024]{0} dynamic-slice(%[[AR0]], %[[_:.*]])
    CHECK-SAME:   dynamic_slice_sizes={1024}
    CHECK-NEXT:   %[[AR1:.*]] = bf16[1024]{0} all-reduce(%[[DS]])
    CHECK-SAME:   replica_groups={{[{]}}{0,4},{1,5},{2,6},{3,7}{{[}]}}
    CHECK:        %[[EXISTING_AR:.*]] = bf16[2,4096,4096]{2,1,0} all-reduce(%[[P0]])
    CHECK-SAME:   replica_groups={[[DESIRED_RGS]]}
    CHECK:        ROOT
    CHECK-NOT:    %[[AR1]]
    CHECK-SAME:   %[[EXISTING_AR]]
    )"));
}

TEST_F(
    AllReduceSplitterTest,
    DoesNotMatchBasicPatternIfDynamicSliceIsNotRootAndThereIsNoAllReduceWithSameReplicaGroups) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  p.1 = bf16[2,4096,4096] parameter(1)
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(p, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  dynamic_slice = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
  broadcast = bf16[1024,1024] broadcast(dynamic_slice), dimensions={0}
  add = bf16[2,4096,4096] add(p,p.1)
  ROOT _ = tuple(broadcast, add)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(AllReduceCount(*module), 1);
}

TEST_F(AllReduceSplitterTest,
       DoesNotMatchBasicPatternIfDynamicSliceIsFullySharded) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(first.ar, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(512)
  offset = s32[] multiply(reshape, slice_size)
  ROOT _ = bf16[512] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={512}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(AllReduceCount(*module), 2);
}

TEST_F(AllReduceSplitterTest,
       DoesNotMatchBasicPatternIfItIsNotCompiledWithSPMDPartitioning) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(first.ar, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  ROOT _ = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
}
)";
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/1, /*num_partitions=*/8);
  config.set_use_spmd_partitioning(false);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(false));
  EXPECT_THAT(AllReduceCount(*module), 2);
}

TEST_F(AllReduceSplitterTest,
       DoesNotMatchBasicPatternIfUseGlobalDeviceIdsIsFalse) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, channel_id=1
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(first.ar, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  ROOT _ = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(false));

  EXPECT_EQ(AllReduceCount(*module), 2);
}

TEST_F(AllReduceSplitterTest,
       DoesNotMatchBasicPatternIfIsNotCrossAllPartitionsAllReduce) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(first.ar, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  ROOT _ = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  EXPECT_THAT(AllReduceSplitter().Run(module.get()), IsOkAndHolds(false));

  EXPECT_EQ(AllReduceCount(*module), 2);
}

TEST_F(
    AllReduceSplitterFilecheckTest,
    PipelineMatchesBasicPatternWithDynamicSliceAsRootAndRewritesToReduceScatter) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  zero = bf16[] constant(0)
  reduce = bf16[4096] reduce(first.ar, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  ROOT _ = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  HloPassPipeline pipeline("all-reduce-splitter-rewrite");
  pipeline.AddPass<AllReduceSplitter>();
  pipeline.AddPass<ReduceScatterCreator>();
  EXPECT_THAT(pipeline.Run(module.get()), IsOkAndHolds(true));
  TF_EXPECT_OK(FileCheck(module->ToString(), R"(
    CHECK-DAG:    %[[P0:.*]] = bf16[2,4096,4096]{2,1,0} parameter(0)
    CHECK:        %[[AR0:.*]] = bf16[2,4096,4096]{2,1,0} all-reduce(%[[P0]])
    CHECK-SAME:   replica_groups={[[DESIRED_RGS:.*]]}
    CHECK-DAG:    %[[ZERO:.*]] = bf16[] constant(0)
    CHECK-DAG:    %[[LOCAL_REDUCE:.*]] = bf16[4096]{0} reduce(%[[AR0]], %[[ZERO]])
    CHECK:        %[[REDUCE_SCATTER:.*]] = bf16[1024]{0} reduce-scatter(%[[LOCAL_REDUCE]])
    CHECK-SAME:   replica_groups={[[DESIRED_RGS]]}
    CHECK-NEXT:   ROOT %[[AR2:.*]] = bf16[1024]{0} all-reduce(%[[REDUCE_SCATTER]])
    CHECK-SAME:   replica_groups={{[{]}}{0,4},{1,5},{2,6},{3,7}{{[}]}}
    )"));
}

TEST_F(
    AllReduceSplitterFilecheckTest,
    PipelineMatchesBasicPatternWithDynamicSliceNotAsRootAndRewritesToReduceScatter) {  // NOLINT
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a,b)
}

ENTRY main {
  p = bf16[2,4096,4096] parameter(0)
  zero = bf16[] constant(0)
  first.ar = bf16[2,4096,4096] all-reduce(p), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  reduce = bf16[4096] reduce(p, zero), dimensions={0,1}, to_apply=sum
  all-reduce = bf16[4096] all-reduce(reduce), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum, use_global_device_ids=true, channel_id=1
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  dynamic_slice = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
  broadcast = bf16[1024,1024] broadcast(dynamic_slice), dimensions={0}
  ROOT _ = tuple(broadcast, first.ar)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      PrepareModule(hlo_string, /*num_replicas=*/1, /*num_partitions=*/8));

  HloPassPipeline pipeline("all-reduce-splitter-rewrite");
  pipeline.AddPass<AllReduceSplitter>();
  pipeline.AddPass<ReduceScatterCreator>();
  EXPECT_THAT(pipeline.Run(module.get()), IsOkAndHolds(true));
  TF_EXPECT_OK(FileCheck(module->ToString(), R"(
    CHECK-DAG:    %[[P0:.*]] = bf16[2,4096,4096]{2,1,0} parameter(0)
    CHECK-DAG:    %[[ZERO:.*]] = bf16[] constant(0)
    CHECK-DAG:    %[[LOCAL_REDUCE:.*]] = bf16[4096]{0} reduce(%[[P0]], %[[ZERO]])
    CHECK:        %[[REDUCE_SCATTER:.*]] = bf16[1024]{0} reduce-scatter(%[[LOCAL_REDUCE]])
    CHECK-NEXT:   %[[AR1:.*]] = bf16[1024]{0} all-reduce(%[[REDUCE_SCATTER]])
    CHECK-SAME:   replica_groups={{[{]}}{0,4},{1,5},{2,6},{3,7}{{[}]}}
    CHECK:        %[[EXISTING_AR:.*]] = bf16[2,4096,4096]{2,1,0} all-reduce(%[[P0]])
    CHECK:        ROOT
    CHECK-NOT:    %[[AR1]]
    CHECK-SAME:   %[[EXISTING_AR]]
    )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
