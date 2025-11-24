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

#ifndef XLA_TESTS_COLLECTIVE_OPS_E2E_TEST_BASE_H_
#define XLA_TESTS_COLLECTIVE_OPS_E2E_TEST_BASE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {

class CollectiveOpsE2ETestBase : public HloHardwareIndependentTestBase {
 public:
  CollectiveOpsE2ETestBase();

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
      absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
      absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
      int64_t num_replicas, bool run_hlo_passes,
      DeviceAssignment* device_assignment);

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments, int64_t num_replicas,
      DeviceAssignment* vice_assignment, bool run_hlo_passes, bool use_threads);

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      std::vector<std::vector<Literal*>> arguments,
      DeviceAssignment* device_assignment, int64_t num_replicas,
      bool run_hlo_passes);

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      OpaqueExecutable* executable, int64_t num_replicas);

  const se::GpuComputeCapability& Capability() {
    return hlo_runner_->backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }

  bool IsHopperAndHigher() {
    return Capability().IsCuda() &&
           Capability().cuda_compute_capability()->IsAtLeastHopper();
  }

  // Makes a DeviceAssignment device#i to replica_id #i.
  DeviceAssignment MakeDeviceAssn(int64_t num_replicas) {
    DeviceAssignment assn(/*replica_count=*/num_replicas,
                          /*computation_count=*/1);
    for (int64_t i = 0; i < num_replicas; ++i) {
      assn(i, 0) = i;
    }
    return assn;
  }

 protected:
  std::unique_ptr<HloRunner> hlo_runner_;
  std::unique_ptr<HloRunner> reference_hlo_runner_;
};

// E2E tests for collective ops. These will generally verify some HLO transform
// for collectives (for example, sync -> async conversion) and correct
// execution of the transformed HLO.

// E2E test for collectives with flags set. Has constructor arguments specifying
// whether to enable/disable async collectives, and to set the memcpy_local_p2p
// flag. Subclasses pass in constructor arguments based on GetParam().
class CollectiveOpsWithFlagsBase : public CollectiveOpsE2ETestBase {
 public:
  CollectiveOpsWithFlagsBase(bool enable_async, bool enable_p2p_memcpy)
      : enable_async_(enable_async), enable_p2p_memcpy_(enable_p2p_memcpy) {
    VLOG(1) << "Running with " << num_devices_ << " devices";
    num_devices_ = hlo_runner_->backend().device_count();
  }

 protected:
  DebugOptions GetDebugOptionsForTest() const override;

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      absl::string_view hlo_string, int64_t num_replicas);

  const bool enable_async_;
  const bool enable_p2p_memcpy_;
  int64_t num_devices_;
};

}  // namespace xla

#endif  // XLA_TESTS_COLLECTIVE_OPS_E2E_TEST_BASE_H_
