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
#include "xla/service/gpu/fusions/input_slices_mlir.h"

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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;

std::optional<IndexingMap>
MlirInputSlicesFusion::ComputeThreadIdToOutputIndexing(
    int64_t output_id, mlir::MLIRContext* ctx) const {
  auto launch_dims = launch_dimensions();
  auto* slice = &analysis_.fusion_root(output_id).instruction();
  const auto& shape = slice->operand(0)->shape();
  return GetDefaultThreadIdIndexingMap(launch_dims, unroll_factor_, shape,
                                       ctx) *
         *ComputeInputToOutputIndexing(slice, 0, ctx)
              .indexing_maps.front()
              .begin();
}

std::vector<mlir_converter::EpilogueSpecification>
MlirInputSlicesFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                    mlir::MLIRContext* mlir_context) const {
  std::vector<const HloInstruction*> roots;
  roots.reserve(analysis_.fusion_root_count());
  for (const auto& root : analysis_.fusion_roots()) {
    roots.push_back(&root.instruction());
  }

  // We don't actually use epilogues here, but this is how we tell the base
  // class not to emit code for the slices.
  return {mlir_converter::EpilogueSpecification::FromOutputIndexing(
      analysis_, roots, roots, *this, mlir_context)};
}

LaunchDimensions MlirInputSlicesFusion::launch_dimensions() const {
  // Note: these launch dimensions are not optimal if the input isn't used
  // fully.
  const auto& root = analysis_.fusion_root(0).instruction();
  const auto& shape = root.operand(0)->shape();
  return CalculateLaunchDimensions(shape, analysis_.device_info(),
                                   {unroll_factor_});
}

absl::Status MlirInputSlicesFusion::EmitEntryFunction(
    const mlir_converter::PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  auto launch_dims = launch_dimensions();
  const auto& shape =
      analysis_.fusion_root(0).instruction().operand(0)->shape();
  auto input_indexing = GetDefaultThreadIdIndexingMap(
      launch_dims, unroll_factor_, shape, builder.getContext());

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  auto result_tensors = EmitThreadLoopNest(
      builder, output_tensor_args, input_indexing,
      [&](ValueRange output_tensors, ValueRange dim_values,
          ValueRange symbol_values) -> SmallVector<Value> {
        auto input_indices = mlir_converter::ApplyIndexing(
            input_indexing, dim_values, symbol_values, builder);
        SmallVector<Value> input_operands(
            entry_function.getArguments().take_front(num_inputs));
        absl::c_copy(input_indices, std::back_inserter(input_operands));
        SmallVector<Value> result_tensors;
        result_tensors.reserve(output_tensor_args.size());

        absl::flat_hash_map<const HloInstruction*, mlir::Value> input_values;
        for (const HloInstructionAdaptor& root : analysis_.fusion_roots()) {
          const auto* arg = root.instruction().operand(0);
          if (auto& value = input_values[arg]; !value) {
            value =
                builder.create<PureCallOp>(call_targets(arg), input_operands)
                    .getResult(0);
          }
        }

        for (auto [output_index, output] : llvm::enumerate(output_tensors)) {
          auto output_indexing = ComputeThreadIdToOutputIndexing(
              output_index, entry_function.getContext());
          mlir::Value in_bounds = mlir_converter::CheckConstraints(
              *output_indexing, dim_values, symbol_values, builder);
          auto if_op = builder.create<mlir::scf::IfOp>(
              in_bounds,
              [&, output_index = output_index, output = output](
                  mlir::OpBuilder b, mlir::Location loc) {
                mlir::ImplicitLocOpBuilder then_builder(loc, b);
                auto output_indices = mlir_converter::ApplyIndexing(
                    *output_indexing, dim_values, symbol_values, then_builder);
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
