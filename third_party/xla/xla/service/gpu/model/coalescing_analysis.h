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

#ifndef XLA_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Computes read coalescing for operands of an instruction or a
// producer-consumer fusion.
// Note, that later, after we migrate away from using the heuristic, we might
// want to use HloFusionAdaptor instead of having two different constructors.
class CoalescingAnalysis {
 public:
  // Computes read coalescing for operands of `instr`.
  CoalescingAnalysis(const HloInstruction* instr,
                     absl::Span<const HloInstruction* const> operands,
                     const HloFusionAnalysis& fusion_analysis,
                     KernelFusionInterface* fusion_interface = nullptr,
                     mlir::MLIRContext* mlir_context = nullptr,
                     bool use_heuristic = true);

  // Computes read coalescing for operands of fused `producer` and `consumer`.
  CoalescingAnalysis(const HloInstruction* producer,
                     const HloInstruction* consumer,
                     absl::Span<const HloInstruction* const> operands,
                     const HloFusionAnalysis& fusion_analysis,
                     KernelFusionInterface* fusion_interface = nullptr,
                     mlir::MLIRContext* mlir_context = nullptr,
                     bool use_heuristic = true);

  // Returns true if the operand is read coalesced.
  bool IsReadCoalesced(const HloInstruction* operand) const;

 private:
  bool ComputeCoalescingForAllOperands(
      const HloFusionAdaptor& fusion_adaptor,
      absl::Span<const HloInstruction* const> operands,
      const HloFusionAnalysis& fusion_analysis,
      KernelFusionInterface* fusion_interface, mlir::MLIRContext* mlir_context);

  absl::flat_hash_map<const HloInstruction*, bool> coalescing_per_operand_;
  bool is_coalesced_computed_by_heuristic_ = false;
};

// Returns true if all input reads are coalesced. If consumer is not nullptr,
// producer and consumer are considered as one fusion, otherwise it's only the
// producer.
bool IsReadCoalescedHeuristic(HloFusionAnalysis::EmitterFusionKind fusion_kind,
                              const HloInstruction* producer,
                              const HloInstruction* consumer = nullptr);

// Returns the bandwidth utilization rate of the memory access for the given
// tiled HLO instruction. Naturally, values are between 0 and 1, where a
// perfectly coalesced read has a utilization rate of 1.
//
// Note: the assumption is that the tile sizes do not include padding beyond
// the end of the shape.
double BandwidthUtilizationRateHeuristicForTiledMemoryAccess(
    const TiledHloInstruction& hbm_access_instr,
    const se::DeviceDescription& device_info);

// Returns true if read of this tiled hlo operand is coalesced.
//
// We consider a read coalesced if the operand tile consist of contiguous chunk
// of memory that saturate DRAM->L2 cache line. For post-V100 NVIDIA GPUs, that
// is 64 bytes by default.
//
// TODO(b/332714755): check whether we should bump up the granularity of
// memory transactions.
bool IsTiledReadCoalescedHeuristic(const TiledHloInstruction& operand,
                                   const se::DeviceDescription& device_info);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_
