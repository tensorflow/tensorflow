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

#include "xla/service/gpu/fusions/concatenate_mlir.h"

#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;

const Shape& GetLargestConcatOperandShape(const HloFusionAnalysis& analysis) {
  const HloInstruction& concat = analysis.fusion_hero(0).instruction();
  int64_t dim = concat.concatenate_dimension();
  auto less = [&](const HloInstruction* lhs, const HloInstruction* rhs) {
    return lhs->shape().dimensions(dim) < rhs->shape().dimensions(dim);
  };
  HloInstruction* operand = *absl::c_max_element(concat.operands(), less);
  return operand->shape();
}

// Computes the unroll factor that divides concat dimension of all operands.
int ComputeUnrollFactor(const HloFusionAnalysis& analysis,
                        int unroll_factor_for_the_largest_shape) {
  auto& concat = analysis.fusion_hero(0).instruction();
  int unroll_factor = unroll_factor_for_the_largest_shape;
  int64_t dim = concat.concatenate_dimension();
  for (const HloInstruction* operand : concat.operands()) {
    if (unroll_factor == 1) return 1;
    unroll_factor = std::gcd(unroll_factor, operand->shape().dimensions(dim));
  }
  return unroll_factor;
}

}  // namespace

MlirConcatenateFusion::MlirConcatenateFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis),
      largest_shape_(GetLargestConcatOperandShape(analysis_)),
      config_(ComputeLoopFusionConfig(analysis_, largest_shape_)),
      unroll_factor_(ComputeUnrollFactor(analysis_, config_.unroll_factor)) {}

LaunchDimensions MlirConcatenateFusion::launch_dimensions() const {
  return CalculateLaunchDimensions(largest_shape_, analysis_.device_info(),
                                   config_);
}

std::optional<IndexingMap>
MlirConcatenateFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

std::optional<IndexingMap>
MlirConcatenateFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  // TODO(b/331356433): Add constraints depending on the `hero_operand_index`.
  return GetDefaultThreadIdIndexingMap(launch_dimensions(), unroll_factor_,
                                       largest_shape_, ctx);
}

std::vector<mlir_converter::EpilogueSpecification>
MlirConcatenateFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                    mlir::MLIRContext* mlir_context) const {
  return {mlir_converter::EpilogueSpecification::FromIdentityIndexing(
      &analysis_.fusion_hero(0).instruction(),
      &analysis_.fusion_root(0).instruction(), mlir_context)};
}

absl::Status MlirConcatenateFusion::EmitEntryFunction(
    const mlir_converter::PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  auto thread_and_block_ids = EmitThreadAndBlockIds(builder);
  auto* ctx = entry_function.getContext();

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  SmallVector<Value> input_tensors(
      entry_function.getArguments().take_front(num_inputs));
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  SmallVector<Value> result_tensors{output_tensor_args.begin(),
                                    output_tensor_args.end()};

  auto thread_id_to_input_map =
      ComputeThreadIdToInputIndexing(
          /*root_index=*/0, /*hero_operand_index=*/0, ctx)
          .value();
  auto epilogue_indexing = ComputeEpilogueInputToOutputIndexing(
      analysis_.fusion_hero(0), analysis_.fusion_root(0), ctx);

  const auto* concat = &analysis_.fusion_hero(0).instruction();
  for (auto [operand_index, operand] : llvm::enumerate(concat->operands())) {
    auto input_to_output_map =
        *ComputeInputToOutputIndexing(concat, /*input_id=*/operand_index, ctx)
             .indexing_maps.front()
             .begin();
    auto thread_id_to_output_map = ComposeIndexingMaps(
        ComposeIndexingMaps(thread_id_to_input_map, input_to_output_map),
        epilogue_indexing);
    thread_id_to_output_map.Simplify();

    auto loop_nest_body_builder =
        [&, operand_index = operand_index](
            ImplicitLocOpBuilder& nested_b, ValueRange symbol_values,
            ValueRange output_indices,
            ValueRange output_tensors) -> SmallVector<Value> {
      auto input_indices = mlir_converter::ApplyIndexing(
          thread_id_to_input_map, thread_and_block_ids, symbol_values,
          nested_b);

      auto result_scalar = mlir_converter::ProvideParameter(
          root_computation, concat, operand_index, input_indices, call_targets,
          entry_function, nested_b);
      absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>
          hero_value{{concat, result_scalar}};
      auto result_scalars = EmitEpilogue(
          /*epilogue_index=*/0, computations, entry_function, hero_value,
          output_indices, nested_b)[&analysis_.fusion_root(0).instruction()];

      SmallVector<Value> result_tensors;
      result_tensors.reserve(output_tensor_args.size());
      for (auto [tensor, value] : llvm::zip(output_tensors, result_scalars)) {
        result_tensors.push_back(
            nested_b
                .create<mlir::tensor::InsertOp>(value, tensor, output_indices)
                .getResult());
      }

      return result_tensors;
    };
    result_tensors = mlir_converter::EmitXlaLoopOp(
        builder, thread_and_block_ids, result_tensors, thread_id_to_output_map,
        loop_nest_body_builder);
  }

  builder.create<mlir::func::ReturnOp>(result_tensors);

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
