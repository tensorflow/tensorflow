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

#ifndef XLA_BACKENDS_GPU_TESTS_COLLECTIVE_OPS_E2E_TEST_BASE_H_
#define XLA_BACKENDS_GPU_TESTS_COLLECTIVE_OPS_E2E_TEST_BASE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {

inline constexpr size_t kMB = 1024LL * 1024LL;
inline constexpr size_t kGB = 1024LL * kMB;

class CollectiveOpsE2ETestBase : public HloHardwareIndependentTestBase {
 public:
  CollectiveOpsE2ETestBase(size_t memory_size, size_t collectives_memory_size) {
    SetupHloRunner(memory_size, collectives_memory_size);
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();

    // Disable autotuning which is unnecessary.
    debug_options.set_xla_gpu_autotune_level(0);

    return debug_options;
  }

  struct ExecutionResult {
    std::unique_ptr<OpaqueExecutable> executable;
    std::vector<Literal> results;
    const HloModule* optimized_module;
  };

  absl::StatusOr<ExecutionResult> ExecuteReplicated(
      std::unique_ptr<HloModule> module);

  absl::StatusOr<ExecutionResult> ExecuteReplicated(
      std::unique_ptr<HloModule> module, const std::vector<Literal*>& arguments,
      bool run_hlo_passes = true);

  absl::StatusOr<ExecutionResult> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const std::vector<std::vector<Literal*>>& arguments,
      bool run_hlo_passes = true);

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      OpaqueExecutable* executable,
      const std::vector<std::vector<Literal*>>& arguments,
      bool run_hlo_passes = true);

  const se::GpuComputeCapability& Capability() {
    return gpu_compute_capability_;
  }

  bool IsHopperAndHigher() {
    return Capability().IsCuda() &&
           Capability().cuda_compute_capability()->IsAtLeastHopper();
  }

  bool IsAmpereAndHigher() {
    return Capability().IsCuda() &&
           Capability().cuda_compute_capability()->IsAtLeastAmpere();
  }

 protected:
  std::unique_ptr<HloRunnerPjRt> hlo_runner_;
  se::GpuComputeCapability gpu_compute_capability_;

 private:
  void SetupHloRunner(size_t memory_size, size_t collectives_memory_size);
};

// E2E tests for collective ops. These will generally verify some HLO transform
// for collectives (for example, sync -> async conversion) and correct
// execution of the transformed HLO.

// E2E test for collectives with flags set. Has constructor arguments specifying
// whether to enable/disable async collectives, and to set the memcpy_local_p2p
// flag. Subclasses pass in constructor arguments based on GetParam().
class CollectiveOpsWithFlagsBase : public CollectiveOpsE2ETestBase {
 public:
  CollectiveOpsWithFlagsBase(bool enable_async, bool enable_p2p_memcpy,
                             size_t memory_size, size_t collectives_memory_size)
      : CollectiveOpsE2ETestBase(memory_size, collectives_memory_size),
        enable_async_(enable_async),
        enable_p2p_memcpy_(enable_p2p_memcpy) {}

 protected:
  DebugOptions GetDebugOptionsForTest() const override;

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      absl::string_view hlo_string, int64_t num_replicas);

  const bool enable_async_;
  const bool enable_p2p_memcpy_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TESTS_COLLECTIVE_OPS_E2E_TEST_BASE_H_
