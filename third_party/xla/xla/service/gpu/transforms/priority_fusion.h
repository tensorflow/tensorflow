/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_PRIORITY_FUSION_H_
#define XLA_SERVICE_GPU_TRANSFORMS_PRIORITY_FUSION_H_

#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/cost_model/fusion_analysis_cache.h"
#include "xla/backends/gpu/cost_model/gpu_hlo_cost_analysis.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// PriorityFusion is the main fusion pass for XLA:GPU. It is an HLO pass that
// assigns a priority to each producer instruction based on the estimated
// performance benefit of fusing it into its consumers. The benefit is
// calculated using a performance cost model:
//
//   priority = time_unfused - time_fused
//
// Note: If fusing a producer into its consumers requires duplicating the
// producer, the cost model accounts for this duplication.
//
// The algorithm can be summarized in the following steps:
// 1. For each producer, call the cost model to estimate the potential benefit
//    of fusing it with all its consumers.
// 2. Put all producers with a positive benefit into a priority queue, ordered
//    by benefit.
// 3. Pop the producer with the highest priority from the queue.
// 4. Fuse the producer with its consumers. This may result in a new fusion
//    instruction, or merging into an existing fusion.
// 5. Update the priorities of the operands of the fused instructions and
//    of instructions whose consumers have changed, and update them in the
//    priority queue.
// 6. If the queue is not empty, go to step 3.
//
// Example:
// Consider A -> B -> C, where A, B, and C are fusible operations.
// The fusible producers are A and B.
//
// Priorities are computed:
//  - P(A) = benefit of fusing A into B.
//  - P(B) = benefit of fusing B into C.
//
// Assuming P(A)=10 and P(B)=5, the queue is [(A,10), (B,5)].
//  - A is popped and fused into B, creating fusion(A+B).
//  - The graph becomes fusion(A+B) -> C.
//  - Priority of fusion(A+B) is computed, P(fusion(A+B))=8.
//  - The queue becomes [(fusion(A+B),8)].
//  - fusion(A+B) is popped and fused into C, creating fusion(A+B+C).
//  - The queue becomes empty, and fusion terminates.
//
class PriorityFusion : public HloModulePass {
 public:
  PriorityFusion(tsl::thread::ThreadPool* thread_pool,
                 const se::DeviceDescription& device,
                 const AliasInfo* alias_info,
                 GpuHloCostAnalysis::Options cost_analysis_options,
                 mlir::MLIRContext* mlir_context)
      : thread_pool_(thread_pool),
        device_info_(device),
        alias_info_(alias_info),
        cost_analysis_options_(std::move(cost_analysis_options)),
        fusion_analysis_cache_(device_info_),
        mlir_context_(mlir_context) {}

  absl::string_view name() const override { return "priority-fusion"; }

 protected:
  HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                        const HloInstruction* consumer);

  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                       bool use_multi_output_fusion = false);

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Consumes a unit of compiler fuel and returns true if we should
  // continue with the transformation.
  bool ConsumeFuel(HloInstruction* producer, HloInstruction* consumer);

  // Returns the decision if the constant can be fused into the user.
  FusionDecision CanFuseConstant(const HloInstruction* constant,
                                 const HloInstruction* user);

  tsl::thread::ThreadPool* thread_pool_;
  se::DeviceDescription device_info_;
  const AliasInfo* alias_info_;

  // Cost model options that defines priorities in the queue.
  GpuHloCostAnalysis::Options cost_analysis_options_;

  // Proto with structured logs of fusion decisions. Used only for debugging. If
  // null, logging is disabled.
  std::unique_ptr<FusionProcessDumpProto> fusion_process_dump_;

  HloFusionAnalysisCache fusion_analysis_cache_;

  mlir::MLIRContext* mlir_context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_PRIORITY_FUSION_H_
