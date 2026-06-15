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
#include "xla/backends/gpu/transforms/collectives/collective_backend_assigner.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class CollectiveBackendAssignerTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunCollectiveBackendAssigner(HloModule* module,
                                                    int num_devices_per_host,
                                                    int64_t slice_size = 0) {
    se::GpuComputeCapability gpu_version = se::CudaComputeCapability(8, 0);
    return RunHloPass(CollectiveBackendAssigner(
                          gpu_version, num_devices_per_host, slice_size),
                      module);
  }

  absl::StatusOr<CollectiveBackendConfig_CollectiveBackend>
  GetCollectiveBackendConfig(const HloInstruction* instr) {
    ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr->backend_config<GpuBackendConfig>());
    return gpu_config.collective_backend_config().backend();
  }

  absl::StatusOr<DebugOptions::CollectivesMode> GetCollectivesMode(
      const HloInstruction* instr) {
    ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr->backend_config<GpuBackendConfig>());
    return gpu_config.collective_backend_config().collectives_mode();
  }
};

TEST_F(CollectiveBackendAssignerTest,
       CollectivePermuteSymmetricMemorySetsMode) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[8,8] parameter(0)
      ROOT result = u32[8,8] collective-permute(p0), channel_id=10,
        source_target_pairs={{0,1},{1,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_collective_permute_mode(
          DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);

  EXPECT_THAT(RunCollectiveBackendAssigner(
                  module.get(), /*num_devices_per_host=*/1, /*slice_size=*/0),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* permute =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(permute),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY));
}

TEST_F(CollectiveBackendAssignerTest,
       CollectivePermutePrivateMemoryLeavesDefault) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[8,8] parameter(0)
      ROOT result = u32[8,8] collective-permute(p0), channel_id=12,
        source_target_pairs={{0,1},{1,0}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  // Default is COLLECTIVES_MODE_INVALID — collectives_mode should not be set.

  ASSERT_THAT(RunCollectiveBackendAssigner(module.get(),
                                           /*num_devices_per_host=*/1,
                                           /*slice_size=*/0),
              absl_testing::IsOk());

  const HloInstruction* permute =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(permute),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_MODE_INVALID));
}

TEST_F(CollectiveBackendAssignerTest,
       AllReduceUnaffectedByCollectivePermuteMode) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      ROOT result = f32[8,8] all-reduce(p0), to_apply=add,
        replica_groups={{0,1}}, channel_id=13
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_collective_permute_mode(
          DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);

  EXPECT_THAT(RunCollectiveBackendAssigner(
                  module.get(), /*num_devices_per_host=*/1, /*slice_size=*/0),
              absl_testing::IsOkAndHolds(false));

  const HloInstruction* all_reduce =
      module->entry_computation()->root_instruction();
  // collectives_mode should remain COLLECTIVES_MODE_INVALID (unaffected by
  // permute flag).
  EXPECT_THAT(
      GetCollectivesMode(all_reduce),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_MODE_INVALID));
}

TEST_F(CollectiveBackendAssignerTest, AllGatherSymmetricMemorySetsMode) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[4,8] parameter(0)
      ROOT result = u32[8,8] all-gather(p0), dimensions={0},
        replica_groups={{0,1}}, channel_id=20
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  module->mutable_config().mutable_debug_options().set_xla_gpu_all_gather_mode(
      DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);

  EXPECT_THAT(RunCollectiveBackendAssigner(
                  module.get(), /*num_devices_per_host=*/1, /*slice_size=*/0),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(all_gather),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY));
}

TEST_F(CollectiveBackendAssignerTest, AllGatherPrivateMemoryLeavesDefault) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[4,8] parameter(0)
      ROOT result = u32[8,8] all-gather(p0), dimensions={0},
        replica_groups={{0,1}}, channel_id=21
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ASSERT_THAT(RunCollectiveBackendAssigner(module.get(),
                                           /*num_devices_per_host=*/1,
                                           /*slice_size=*/0),
              absl_testing::IsOk());

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(all_gather),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_MODE_INVALID));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
