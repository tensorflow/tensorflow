/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tests/aot_compatibility/gpu/collective_aot_test_base.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {

absl::StatusOr<std::vector<Literal>> CollectiveAotTestBase::RunReplicated(
    absl::string_view hlo, const HloModuleConfig& config, bool run_hlo_passes) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                   ParseAndReturnVerifiedModule(hlo, config));
  DeviceAssignment device_assignment = config.static_device_assignment();
  const int64_t num_devices =
      device_assignment.replica_count() * device_assignment.computation_count();
  return ExecuteReplicated(std::move(module), absl::Span<Literal* const>{},
                           num_devices, &device_assignment, run_hlo_passes,
                           /*use_threads=*/true);
}

absl::StatusOr<std::vector<std::vector<Literal>>>
CollectiveAotTestBase::RunReplicatedRepeatedly(absl::string_view hlo,
                                               const HloModuleConfig& config,
                                               int num_runs) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                   ParseAndReturnVerifiedModule(hlo, config));
  ABSL_RETURN_IF_ERROR(PreprocessModuleForTestRunner(module.get()));
  DeviceAssignment device_assignment = config.static_device_assignment();
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices =
      device_assignment.replica_count() * device_assignment.computation_count();
  options.run_hlo_passes = true;
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> executable,
                   test_runner().CreateExecutable(std::move(module),
                                                  options.run_hlo_passes));
  std::vector<std::vector<Literal>> runs;
  runs.reserve(num_runs);
  for (int run = 0; run < num_runs; ++run) {
    ABSL_ASSIGN_OR_RETURN(runs.emplace_back(),
                     test_runner().ExecuteReplicatedWithExecutable(
                         executable.get(), options, &device_assignment));
  }
  return runs;
}

void KeepCollectivesAsync(DebugOptions& debug_options) {
  debug_options.add_xla_disable_hlo_passes(
      "gpu-convert-async-collectives-to-sync");
}

void EnableNcclSymmetricBuffers(DebugOptions& debug_options) {
  debug_options.add_xla_enable_nccl_symmetric_buffers_for_collectives()
      ->set_collective(DebugOptions::ALLCOLLECTIVES);
}

void DisableCollectiveKernels(DebugOptions& debug_options) {
  debug_options.clear_xla_gpu_experimental_use_collective_kernels();
}

}  // namespace aot_compatibility
}  // namespace xla
