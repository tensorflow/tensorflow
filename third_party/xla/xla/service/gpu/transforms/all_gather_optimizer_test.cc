/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/all_gather_optimizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuAllGatherOptimizerTest : public HloTestBase {
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

    auto changed = AllGatherOptimizer().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  template <HloOpcode oc>
  size_t CollectiveCount(std::unique_ptr<HloModule> &module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            HloPredicateIsOp<oc>);
  }
};

TEST_F(GpuAllGatherOptimizerTest, BranchesOptimized) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024]{2,1,0} parameter(0)
param.2 = bf16[8,128,1024]{2,1,0} parameter(1)
reduce-scatter.1 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[8,128,1024]{2,1,0} all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
reduce-scatter.2 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.2), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.2 = bf16[8,128,1024]{2,1,0} all-gather(reduce-scatter.2), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
add.1 = bf16[8,128,1024]{2,1,0} add(all-gather.1, all-gather.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/true));
  // graph should contain 1 all-gather but since the node removal piece
  // is diferred, they still exist at this stage
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllGather>(module), 3);
  EXPECT_EQ(CollectiveCount<HloOpcode::kReduceScatter>(module), 2);
}

TEST_F(GpuAllGatherOptimizerTest, DisbledSPMDPartitioningJAXBug) {
  absl::string_view hlo_string = R"(
HloModule pjit_f, entry_computation_layout={(f32[4,8]{1,0}, f32[4,8]{1,0})->f32[8,8]{1,0}}

ENTRY %main.6_spmd (param: f32[4,8], param.1: f32[4,8]) -> f32[8,8] {
  %param = f32[4,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  %all-gather = f32[8,8]{1,0} all-gather(f32[4,8]{1,0} %param), channel_id=1, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(f)/jit(main)/add" source_file="third_party/py/jax/tests/pjit_test.py" source_line=207}
  %param.1 = f32[4,8]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}
  %all-gather.1 = f32[8,8]{1,0} all-gather(f32[4,8]{1,0} %param.1), channel_id=2, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(f)/jit(main)/add" source_file="third_party/py/jax/tests/pjit_test.py" source_line=207}
  ROOT %add.0 = f32[8,8]{1,0} add(f32[8,8]{1,0} %all-gather, f32[8,8]{1,0} %all-gather.1), metadata={op_name="pjit(f)/jit(main)/add" source_file="third_party/py/jax/tests/pjit_test.py" source_line=207}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/2,
                                               /*expect_change=*/true));
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllGather>(module), 1);
}

TEST_F(GpuAllGatherOptimizerTest, MoreThanSingleUserForAllGather) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024]{2,1,0} parameter(0)
param.2 = bf16[8,128,1024]{2,1,0} parameter(1)
param.3 = bf16[8,128,1024]{2,1,0} parameter(2)
reduce-scatter.1 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[8,128,1024]{2,1,0} all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
reduce-scatter.2 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.2), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.2 = bf16[8,128,1024]{2,1,0} all-gather(reduce-scatter.2), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
reduce-scatter.3 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.3), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.3 = bf16[8,128,1024]{2,1,0} all-gather(reduce-scatter.3), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
add.1 = bf16[8,128,1024]{2,1,0} add(all-gather.1, all-gather.3)
add.2 = bf16[8,128,1024]{2,1,0} add(all-gather.1, all-gather.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
  // see the comment for BranchesOptimized test
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllGather>(module), 3);
  EXPECT_EQ(CollectiveCount<HloOpcode::kReduceScatter>(module), 3);
}

TEST_F(GpuAllGatherOptimizerTest, AllGatherWithOpInBetweenOnRightBranch) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024]{2,1,0} parameter(0)
param.2 = bf16[8,128,1024]{2,1,0} parameter(1)
param.3 = bf16[8,128,1024]{2,1,0} parameter(2)
reduce-scatter.1 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
reduce-scatter.2 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.2), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
add.1 = bf16[8,64,1024]{2,1,0} add(reduce-scatter.1, reduce-scatter.2)
all-gather.1 = bf16[8,128,1024]{2,1,0} all-gather(add.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
reduce-scatter.3 = bf16[8,64,1024]{2,1,0} reduce-scatter(param.3), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.3 = bf16[8,128,1024]{2,1,0} all-gather(reduce-scatter.3), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
add.2 = bf16[8,128,1024]{2,1,0} add(all-gather.1, all-gather.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/true));
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllGather>(module), 3);
  EXPECT_EQ(CollectiveCount<HloOpcode::kReduceScatter>(module), 3);
}

TEST_F(GpuAllGatherOptimizerTest, AllGatherOneSided) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024]{2,1,0} parameter(0)
param.2 = bf16[8,128,1024]{2,1,0} parameter(1)
param.3 = bf16[8,128,1024]{2,1,0} parameter(2)

add.1 = bf16[8,128,1024]{2,1,0} add(param.1, param.2)
reduce-scatter = bf16[8,64,1024]{2,1,0} reduce-scatter(param.3), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather = bf16[8,128,1024]{2,1,0} all-gather(reduce-scatter), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
add.2 = bf16[8,128,1024]{2,1,0} add(all-gather, add.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllGather>(module), 1);
  EXPECT_EQ(CollectiveCount<HloOpcode::kReduceScatter>(module), 1);
}

TEST_F(GpuAllGatherOptimizerTest, DifferentOperandShapes) {
  absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
param.1 = bf16[8,64,128]{2,1,0} parameter(0)
param.2 = bf16[8,128,64]{2,1,0} parameter(1)
all-gather.1 = bf16[8,128,128]{2,1,0} all-gather(param.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
all-gather.2 = bf16[8,128,128]{2,1,0} all-gather(param.2), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, use_global_device_ids=true
add.1 = bf16[8,128,128]{2,1,0} add(all-gather.1, all-gather.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
