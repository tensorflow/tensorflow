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
#include "xla/backends/gpu/codegen/emitters/in_place_dynamic_update_slice.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using emitters::ApplyIndexing;
using emitters::CallTargetProvider;
using emitters::ClampIndex;
using emitters::PartitionedComputations;
using emitters::ProvideParameter;
using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::AddIOp;
using mlir::func::ReturnOp;
using mlir::tensor::InsertOp;

constexpr int kDUSUpdateIndex = 1;

}  // namespace

LaunchDimensions InPlaceDynamicUpdateSliceFusion::launch_dimensions() const {
  const auto& update_shape =
      dus_ops_.front().GetOperand(kDUSUpdateIndex).shape();
  return CalculateLaunchDimensions(update_shape, analysis_.device_info(),
                                   config_);
}

std::optional<IndexingMap>
InPlaceDynamicUpdateSliceFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* indexing_context) const {
  // TODO(b/331355203): Implement thread ID -> operand indexing.
  if (hero_operand_index != kDUSUpdateIndex) {
    return std::nullopt;
  }
  auto launch_dims = launch_dimensions();
  // It is guaranteed that all DUS ops have the same output shape at this point.
  const auto& update_shape =
      dus_ops_.front().GetOperand(kDUSUpdateIndex).shape();
  return GetDefaultThreadIdIndexingMap(launch_dims, config_.unroll_factor,
                                       update_shape, indexing_context);
}

std::vector<emitters::EpilogueSpecification>
InPlaceDynamicUpdateSliceFusion::GetEpilogues(
    const HloFusionInstruction& fusion, mlir::MLIRContext* mlir_context) const {
  // We don't actually support epilogues for DUS, but this is how we tell
  // the base class that we don't want it to generate code for the DUS.
  std::vector<emitters::EpilogueSpecification> epilogues;
  for (const auto& [dus_op, root] :
       llvm::zip(dus_ops_, analysis_.fusion_roots())) {
    epilogues.push_back(emitters::EpilogueSpecification::FromIdentityIndexing(
        &dus_op.instruction(), &root.instruction(), mlir_context));
  }
  return epilogues;
}

absl::Status InPlaceDynamicUpdateSliceFusion::EmitEntryFunction(
    const PartitionedComputations& computations,
    const CallTargetProvider& call_targets, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());
  auto thread_and_block_ids = EmitThreadAndBlockIds(b);

  mlir::MLIRContext* mlir_context = entry_function.getContext();

  auto indexing = *ComputeThreadIdToInputIndexing(
      /*root_index=*/0,
      /*hero_operand_index=*/kDUSUpdateIndex, mlir_context);
  indexing.Simplify();
  indexing.RemoveUnusedSymbols();

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  auto result_tensors = emitters::EmitXlaLoopOp(
      b, thread_and_block_ids, output_tensor_args, indexing,
      [&](ImplicitLocOpBuilder& nested_b, ValueRange symbol_values,
          ValueRange input_indices,
          ValueRange output_tensors) -> llvm::SmallVector<Value> {
        llvm::SmallVector<Value> results;
        for (auto [instr, root, output] :
             llvm::zip(dus_ops_, analysis_.fusion_roots(), output_tensors)) {
          const auto* dus_instr =
              Cast<HloDynamicUpdateSliceInstruction>(&instr.instruction());
          const auto& update_shape = dus_instr->update()->shape();
          SmallVector<Value> update_indices;
          auto start_indices = ProvideParameterRange(
              root_computation, dus_instr,
              dus_instr->first_index_operand_number(), update_shape.rank(), {},
              call_targets, entry_function, nested_b);
          for (int i = 0; i < update_shape.rank(); ++i) {
            int64_t update_size = update_shape.dimensions(i);
            auto start_index = ClampIndex(
                start_indices[i],
                primitive_util::IsUnsignedIntegralType(
                    dus_instr
                        ->operand(i + dus_instr->first_index_operand_number())
                        ->shape()
                        .element_type()),
                dus_instr->shape().dimensions(i) - update_size, nested_b);

            update_indices.push_back(
                nested_b.create<AddIOp>(input_indices[i], start_index));
          }

          auto updated_value = ProvideParameter(
              root_computation, dus_instr, kDUSUpdateIndex, input_indices,
              call_targets, entry_function, nested_b);
          // Handle bitcasts under the DUS.
          if (dus_instr->shape() != root.shape()) {
            update_indices = ApplyIndexing(
                GetBitcastMap(dus_instr->shape(), root.shape(), b.getContext()),
                update_indices, {}, nested_b);
          }
          results.push_back(nested_b.create<InsertOp>(updated_value[0], output,
                                                      update_indices));
        }
        return results;
      });

  b.create<ReturnOp>(result_tensors);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
