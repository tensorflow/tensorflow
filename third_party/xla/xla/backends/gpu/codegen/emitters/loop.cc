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
#include "xla/backends/gpu/codegen/emitters/loop.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/loop_kernel_emitter.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::MLIRContext;

const Shape& GetIndexShape(const Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes(0) : shape;
}

}  // namespace

std::optional<IndexingMap> LoopFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  return emitters::LoopFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
      GetWorkDimensions(),
      GetIndexShape(analysis_.fusion_root(root_index).shape()), mlir_context);
}

std::optional<std::vector<IndexingMap>>
LoopFusion::ComputeThreadIdToInputIndexing(int64_t root_index,
                                           MLIRContext* mlir_context) const {
  std::optional<IndexingMap> thread_id_to_output_indexing =
      ComputeThreadIdToOutputIndexing(root_index, mlir_context);
  if (!thread_id_to_output_indexing.has_value()) {
    return std::nullopt;
  }
  const HloInstruction* fusion_root =
      &analysis_.fusion_root(root_index).instruction();
  auto output_to_input_indexing =
      ComputeOutputToInputIndexing(fusion_root, /*output_id=*/0, mlir_context);
  std::vector<IndexingMap> result;
  result.reserve(fusion_root->operand_count());
  for (int64_t operand_index = 0; operand_index < fusion_root->operand_count();
       ++operand_index) {
    auto output_to_input_indexing_maps =
        output_to_input_indexing.indexing_maps[operand_index];
    // Since we are computing the indexing for a non-fusion op, there is only
    // one indexing map per operand.
    CHECK_EQ(output_to_input_indexing_maps.size(), 1);
    IndexingMap thread_id_to_input_indexing_map =
        ComposeIndexingMaps(*thread_id_to_output_indexing,
                            output_to_input_indexing_maps.begin()->map());
    thread_id_to_input_indexing_map.Simplify();
    result.push_back(thread_id_to_input_indexing_map);
  }
  return result;
}

LaunchDimensions LoopFusion::launch_dimensions() const {
  Shape indexing_shape = emitters::LoopFusionKernelEmitter::GetIndexingShape(
      analysis_.fusion_spec());
  auto dims = CalculateLaunchDimensions(indexing_shape, analysis_.device_info(),
                                        config_);
  const auto& blocks = dims.block_counts();
  auto split_x = MaybeSplitGridDimensionX(dims.thread_counts_per_block().x,
                                          blocks.x, analysis_.device_info());
  if (split_x[0] != blocks.x) {  // dim X has been split
    if (blocks.z != 1) {
      LOG(FATAL) << "Unable to split launch dimensions since block.z ("
                 << blocks.z << ") != 1";
    }
    // split blocks.x into x and y, move blocks.y -> blocks.z
    return LaunchDimensions(se::BlockDim(split_x[0], split_x[1], blocks.y),
                            dims.thread_counts_per_block());
  }
  return dims;
}

WorkDimensions LoopFusion::GetWorkDimensions() const {
  WorkDimensions work_dimensions = launch_dimensions().AsWorkDimensions();
  work_dimensions.work_tile_size.dimensions.push_back(config_.unroll_factor);
  return work_dimensions;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoopFusion::CreateMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  emitters::LoopFusionKernelEmitter emitter(
      mlir_context, fusion, analysis_.fusion_spec(), buffer_assignment,
      GetDefaultBufferAlignment(), GetWorkDimensions(), entry_function_name,
      BackendKind::kGpu);

  TF_ASSIGN_OR_RETURN(auto kernel_definition, emitter.EmitKernelDefinition());
  return std::move(kernel_definition).TakeSource().TakeModule();
}

absl::Status LoopFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return absl::UnimplementedError("Not implemented");
}

}  // namespace gpu
}  // namespace xla
