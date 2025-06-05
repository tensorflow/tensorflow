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
#include "xla/backends/gpu/codegen/emitters/input_slices.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;

std::optional<IndexingMap> InputSlicesFusion::ComputeThreadIdToOutputIndexing(
    int64_t output_id, mlir::MLIRContext* ctx) const {
  auto launch_dims = launch_dimensions();
  auto* slice = &analysis_.fusion_root(output_id).instruction();
  const auto& shape = slice->operand(0)->shape();
  return GetDefaultThreadIdIndexingMap(launch_dims, unroll_factor_, shape,
                                       ctx) *
         ComputeInputToOutputIndexing(slice, 0, ctx)
             .indexing_maps.front()
             .begin()
             ->map();
}

std::vector<emitters::EpilogueSpecification> InputSlicesFusion::GetEpilogues(
    const HloFusionInstruction& fusion, mlir::MLIRContext* mlir_context) const {
  std::vector<const HloInstruction*> roots;
  roots.reserve(analysis_.fusion_root_count());
  for (const auto& root : analysis_.fusion_roots()) {
    roots.push_back(&root.instruction());
  }

  // We don't actually use epilogues here, but this is how we tell the base
  // class not to emit code for the slices.
  return {GetEpilogueForOutputIndexing(analysis_, roots, roots, mlir_context)};
}

LaunchDimensions InputSlicesFusion::launch_dimensions() const {
  // Note: these launch dimensions are not optimal if the input isn't used
  // fully.
  const auto& root = analysis_.fusion_root(0).instruction();
  const auto& shape = root.operand(0)->shape();
  return CalculateLaunchDimensions(shape, analysis_.device_info(),
                                   {unroll_factor_});
}

absl::Status InputSlicesFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  auto thread_and_block_ids = EmitThreadAndBlockIds(builder);

  auto launch_dims = launch_dimensions();
  const auto& shape =
      analysis_.fusion_root(0).instruction().operand(0)->shape();
  auto input_indexing = GetDefaultThreadIdIndexingMap(
      launch_dims, unroll_factor_, shape, builder.getContext());

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  auto result_tensors = emitters::EmitXlaLoopOp(
      builder, thread_and_block_ids, output_tensor_args, input_indexing,
      [&](ImplicitLocOpBuilder nested_b, ValueRange symbol_values,
          ValueRange map_results,
          ValueRange output_tensors) -> SmallVector<Value> {
        SmallVector<Value> input_operands(
            entry_function.getArguments().take_front(num_inputs));
        absl::c_copy(map_results, std::back_inserter(input_operands));
        SmallVector<Value> result_tensors;
        result_tensors.reserve(output_tensor_args.size());

        absl::flat_hash_map<const HloInstruction*, mlir::Value> input_values;
        for (const HloInstructionAdaptor& root : analysis_.fusion_roots()) {
          const auto* arg = root.instruction().operand(0);
          if (auto& value = input_values[arg]; !value) {
            value =
                nested_b.create<PureCallOp>(call_targets(arg), input_operands)
                    .getResult(0);
          }
        }

        for (auto [output_index, output] : llvm::enumerate(output_tensors)) {
          auto output_indexing = ComputeThreadIdToOutputIndexing(
              output_index, entry_function.getContext());
          mlir::Value in_bounds = emitters::CheckConstraints(
              *output_indexing, thread_and_block_ids, symbol_values, nested_b);
          auto if_op = nested_b.create<mlir::scf::IfOp>(
              in_bounds,
              [&, output_index = output_index, output = output](
                  mlir::OpBuilder b, mlir::Location loc) {
                mlir::ImplicitLocOpBuilder then_builder(loc, b);
                auto output_indices = emitters::ApplyIndexing(
                    *output_indexing, thread_and_block_ids, symbol_values,
                    then_builder);
                const auto* arg = analysis_.fusion_root(output_index)
                                      .instruction()
                                      .operand(0);
                auto inserted = then_builder.create<mlir::tensor::InsertOp>(
                    input_values[arg], output, output_indices);
                then_builder.create<mlir::scf::YieldOp>(inserted.getResult());
              },
              [&, output = output](mlir::OpBuilder else_builder,
                                   mlir::Location loc) {
                else_builder.create<mlir::scf::YieldOp>(loc, output);
              });
          result_tensors.push_back(if_op.getResult(0));
        }
        return result_tensors;
      });
  builder.create<mlir::func::ReturnOp>(result_tensors);

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
