/* Copyright 2022 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/debug_options_flags.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

class CpuSpmdCompileTest : public HloTestBase {};

TEST_F(CpuSpmdCompileTest, SinglePartition) {
  // Module with "Sharding" custom call and use_spmd_partitioning enabled.
  const char *const hlo_string = R"(
HloModule module

ENTRY entry {
 %parameter.3379 = f32[1,1]{1,0} parameter(0)
 %custom-call.3380 = f32[1,1]{1,0} custom-call(f32[1,1]{1,0} %parameter.3379),
   custom_call_target="Sharding", sharding={replicated}
 ROOT %reshape.6032 = f32[] reshape(f32[1,1]{1,0} %custom-call.3380)
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  auto hlo_module = ParseAndReturnVerifiedModule(hlo_string, config);
  TF_ASSERT_OK(hlo_module.status());

  // Verify that compilation succeeded.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> executable =
      CreateExecutable(std::move(hlo_module.value()),
                       /*run_hlo_passes=*/true);
  TF_EXPECT_OK(executable.status());
}

TEST_F(CpuSpmdCompileTest,
       DynamicSliceCollectiveBroadcastUsesSupportedFallback) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  %param = f32[4,8] parameter(0), sharding={devices=[4,1]<=[4]}
  %index = s32[] parameter(1), sharding={replicated}
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[1,8] dynamic-slice(%param, %index, %zero),
    dynamic_slice_sizes={1,8}, sharding={replicated}
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  config.set_num_partitions(4);

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/4);
  for (int64_t partition = 0; partition < 4; ++partition) {
    device_assignment(0, partition) = 0;
  }
  config.set_static_device_assignment(device_assignment);

  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string, config));
  ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(hlo_module), /*run_hlo_passes=*/true));
  EXPECT_NE(executable, nullptr);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
