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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_MULTI_PHASE_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_MULTI_PHASE_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Fuses small fusions with producer-consumer dependencies into nested fusions.
// Nested fusions are lowered as a sequence of phases of a phased block-cluster
// kernel by `ThreadBlockSpecializedFusion`. Phases are separated by cluster
// barriers and exchange data via DRAM and caches. Execution as sequential
// phases of a single kernel pays that launch overhead only once.
//
// This pass identifies connected DAGs of eligible small fusions that do not
// introduce dataflow cycles, orders them into phases, and wraps each subgraph
// in a `kCustom` fusion with `kind == "__phased_block_cluster"`. The fusion's
// output tuple contains both the intermediate and output buffers of all phases.
// This way, XLA's standard bufferization passes handle intermediate phases
// correctly without changes. Outputs consumed outside the multi-phase fusion
// are projected via `get-tuple-element`.
//
// Current limitations (to be lifted in follow-up CLs):
//   * Greedily forms straight-line producer-consumer chains rather than general
//     multi-branch DAGs, requiring each phase to directly consume the
//     immediately preceding phase without skip-phase edges.
//   * Only `kLoop` stages fitting within one thread-block cluster.
class MultiPhaseFusion : public HloModulePass {
 public:
  explicit MultiPhaseFusion(const se::DeviceDescription& device_info)
      : device_info_(device_info) {}

  absl::string_view name() const override { return "multi-phase-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::DeviceDescription device_info_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_MULTI_PHASE_FUSION_H_
