/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/emitters/concatenate.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/concatenate_kernel_emitter.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using KernelEmitter = emitters::ConcatenateFusionKernelEmitter;

ConcatenateFusion::ConcatenateFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis),
      largest_shape_(KernelEmitter::GetIndexingShape(analysis_.fusion_spec())),
      config_(ComputeLoopFusionConfig(analysis_, largest_shape_)) {
  config_.unroll_factor = KernelEmitter::GetValidUnrollFactor(
      analysis_.fusion_spec(), config_.unroll_factor);
}

LaunchDimensions ConcatenateFusion::launch_dimensions() const {
  return CalculateLaunchDimensions(largest_shape_, analysis_.device_info(),
                                   config_);
}

std::optional<IndexingMap> ConcatenateFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

std::optional<IndexingMap> ConcatenateFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  // TODO(b/331356433): Add constraints depending on the `hero_operand_index`.
  return KernelEmitter::ComputeWorkItemIdToOutputIndexing(GetWorkDimensions(),
                                                          largest_shape_, ctx);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConcatenateFusion::CreateMLIRModule(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  emitters::ConcatenateFusionKernelEmitter emitter(
      context, fusion, analysis_.fusion_spec(), buffer_assignment,
      GetDefaultBufferAlignment(), GetWorkDimensions(), entry_function_name,
      BackendKind::kGpu);

  TF_ASSIGN_OR_RETURN(auto kernel_definition, emitter.EmitKernelDefinition());
  auto [spec, source] = std::move(kernel_definition).ReleaseStorage();
  return std::move(source).ReleaseStorage().module;
}

absl::Status ConcatenateFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return absl::UnimplementedError("This shouldn never be called.");
}

WorkDimensions ConcatenateFusion::GetWorkDimensions() const {
  WorkDimensions work_dimensions = launch_dimensions().AsWorkDimensions();
  work_dimensions.work_tile_size.dimensions.push_back(config_.unroll_factor);
  return work_dimensions;
}

}  // namespace gpu
}  // namespace xla
