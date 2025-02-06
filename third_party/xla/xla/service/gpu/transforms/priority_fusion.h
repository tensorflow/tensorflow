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
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

class PriorityFusion : public HloModulePass {
 public:
  PriorityFusion(tsl::thread::ThreadPool* thread_pool,
                 const se::DeviceDescription& device,
                 GpuHloCostAnalysis::Options cost_analysis_options)
      : thread_pool_(thread_pool),
        device_info_(device),
        cost_analysis_options_(std::move(cost_analysis_options)),
        fusion_analysis_cache_(device_info_) {}

  absl::string_view name() const override { return "priority-fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 protected:
  HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                        const HloInstruction* consumer);

  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer);

 private:
  // Consumes a unit of compiler fuel and returns true if we should
  // continue with the transformation.
  bool ConsumeFuel(HloInstruction* producer, HloInstruction* consumer);

  // Returns the decision if the constant can be fused into the user.
  FusionDecision CanFuseConstant(const HloInstruction* constant,
                                 const HloInstruction* user);

  tsl::thread::ThreadPool* thread_pool_;
  se::DeviceDescription device_info_;

  // Cost model options that defines priorities in the queue.
  GpuHloCostAnalysis::Options cost_analysis_options_;

  // Proto with structured logs of fusion decisions. Used only for debugging. If
  // null, logging is disabled.
  std::unique_ptr<FusionProcessDumpProto> fusion_process_dump_;

  HloFusionAnalysisCache fusion_analysis_cache_;

  mlir::MLIRContext mlir_context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_PRIORITY_FUSION_H_
