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
#include "xla/service/gpu/transforms/collectives/collective_backend_assigner.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class CollectiveBackendAssignerTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunCollectiveBackendAssigner(HloModule* module) {
    se::GpuComputeCapability gpu_version = se::CudaComputeCapability(8, 0);
    return RunHloPass(
        CollectiveBackendAssigner(gpu_version, /*num_devices_per_host=*/1),
        module);
  }

  absl::StatusOr<CollectiveBackendConfig_CollectiveBackend>
  GetCollectiveBackendConfig(const HloInstruction* instr) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        instr->backend_config<GpuBackendConfig>());
    return gpu_config.collective_backend_config().backend();
  }
};

TEST_F(CollectiveBackendAssignerTest, SmallAllReduceUsesNvshmem) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[1024,1024] parameter(0)
      ROOT result = f32[1024,1024] all-reduce(p0), to_apply=add, replica_groups={{0,1}}, channel_id=1
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()), IsOkAndHolds(true));

  const HloInstruction* all_reduce =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(GetCollectiveBackendConfig(all_reduce),
              IsOkAndHolds(CollectiveBackendConfig::NVSHMEM));
}

TEST_F(CollectiveBackendAssignerTest, LargeAllReduceUsesDefault) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[8192,8192] parameter(0)
      ROOT result = f32[8192,8192] all-reduce(p0), to_apply=add, replica_groups={{0,1}}, channel_id=2
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()), IsOkAndHolds(false));

  const HloInstruction* all_reduce =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(GetCollectiveBackendConfig(all_reduce),
              IsOkAndHolds(CollectiveBackendConfig::DEFAULT));
}

TEST_F(CollectiveBackendAssignerTest, SmallCollectivePermuteUsesNvshmem) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[1024,1024] parameter(0)
      ROOT result = u32[1024,1024] collective-permute(p0), channel_id=3,
        source_target_pairs={{0,1},{1,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()), IsOkAndHolds(true));

  const HloInstruction* permute =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(GetCollectiveBackendConfig(permute),
              IsOkAndHolds(CollectiveBackendConfig::NVSHMEM));
}

TEST_F(CollectiveBackendAssignerTest, LargeCollectivePermuteUsesDefault) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[8192,8192] parameter(0)
      ROOT result = u32[8192,8192] collective-permute(p0), channel_id=4,
        source_target_pairs={{0,1},{1,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()), IsOkAndHolds(false));

  const HloInstruction* permute =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(GetCollectiveBackendConfig(permute),
              IsOkAndHolds(CollectiveBackendConfig::DEFAULT));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
