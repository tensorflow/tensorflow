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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_KERNEL_STRATEGY_ANNOTATOR_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_KERNEL_STRATEGY_ANNOTATOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
class GpuTopology;
namespace gpu {

// Annotates collective instructions with the kernel strategy that will be used
// at runtime, writing the result into CollectiveBackendConfig::kernel_strategy.
//
// Currently annotated opcodes:
//   - AllReduce / AllReduceStart : Triton one-shot, Triton two-shot, or NCCL
//     default depending on the shape and device topology.
//   - AllGather                  : Triton one-shot when eligible, NCCL
//     default otherwise.
//
// This pass must run BEFORE the latency-hiding scheduler so that
// SolLatencyEstimator can apply the correct cost model for each collective.
//
// The pass is a no-op for a collective type unless the corresponding entry is
// present in xla_gpu_experimental_use_collective_kernels (e.g.
// COLLECTIVE_KERNEL_ALL_REDUCE for AllReduce,
// COLLECTIVE_KERNEL_ALL_GATHER for AllGather).
class CollectiveKernelStrategyAnnotator : public HloModulePass {
 public:
  // `gpu_topology`        : GpuTopology instance for which the compilation is
  //                         being done; holds target GPU description (used for
  //                         capability checks)
  // `is_multimem_enabled` : mirrors the runtime flag for multimem strategy
  //                         selection (currently unused in cost model, reserved
  //                         for future kMultimem support).
  CollectiveKernelStrategyAnnotator(const GpuTopology& gpu_topology,
                                    bool is_multimem_enabled);

  absl::string_view name() const override {
    return "collective-kernel-strategy-annotator";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const GpuTopology& gpu_topology_;
  const bool is_multimem_enabled_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_KERNEL_STRATEGY_ANNOTATOR_H_
