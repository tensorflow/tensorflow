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
#include "xla/service/gpu/fusions/in_place_dynamic_update_slice_mlir.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::MLIRContext;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::AddIOp;
using mlir::func::ReturnOp;
using mlir::tensor::InsertOp;
using mlir_converter::ApplyAffineMap;
using mlir_converter::CallTargetProvider;
using mlir_converter::ClampIndex;
using mlir_converter::PartitionedComputations;
using mlir_converter::ProvideParameter;

constexpr int kDUSUpdateIndex = 1;

}  // namespace

LaunchDimensions MlirInPlaceDynamicUpdateSliceFusion::launch_dimensions()
    const {
  const auto& update_shape =
      dus_ops_.front()->operand(kDUSUpdateIndex)->shape();
  return CalculateLaunchDimensions(update_shape, analysis_.device_info());
}

std::optional<IndexingMap>
MlirInPlaceDynamicUpdateSliceFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* mlir_context) const {
  // TODO(b/331355203): Implement thread ID -> operand indexing.
  if (hero_operand_index != kDUSUpdateIndex) {
    return std::nullopt;
  }
  auto launch_dims = launch_dimensions();
  // It is guaranteed that all DUS ops have the same output shape at this point.
  const auto& update_shape =
      dus_ops_.front()->operand(kDUSUpdateIndex)->shape();
  return GetDefaultThreadIdIndexingMap(launch_dims, /*unroll_factor=*/1,
                                       update_shape, mlir_context);
}

std::vector<mlir_converter::EpilogueSpecification>
MlirInPlaceDynamicUpdateSliceFusion::GetEpilogues(
    const HloFusionInstruction& fusion, mlir::MLIRContext* mlir_context) const {
  // We don't actually support epilogues for DUS, but this is how we tell
  // the base class that we don't want it to generate code for the DUS.
  return {mlir_converter::EpilogueSpecification::FromIdentityIndexing(
      dus_ops_.front(), analysis_.fusion_roots().front(), mlir_context)};
}

absl::Status MlirInPlaceDynamicUpdateSliceFusion::EmitEntryFunction(
    const PartitionedComputations& computations,
    const CallTargetProvider& call_targets, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());

  mlir::MLIRContext* mlir_context = entry_function.getContext();

  auto indexing = *ComputeThreadIdToInputIndexing(
      /*root_index=*/0,
      /*hero_operand_index=*/kDUSUpdateIndex, mlir_context);
  indexing.Simplify(GetIndexingMapForInstruction);
  indexing.RemoveUnusedSymbols();

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  auto result_tensors = EmitThreadLoopNest(
      b, output_tensor_args, indexing,
      [&](ValueRange output_tensors, ValueRange dim_values,
          ValueRange symbol_values) -> llvm::SmallVector<Value> {
        auto input_indices = ApplyAffineMap(indexing.GetAffineMap(), dim_values,
                                            symbol_values, b);
        llvm::SmallVector<Value> results;
        for (auto [instr, root, output] :
             llvm::zip(dus_ops_, analysis_.fusion_roots(), output_tensors)) {
          const auto* dus_instr = Cast<HloDynamicUpdateSliceInstruction>(instr);
          const auto& update_shape = dus_instr->update()->shape();
          SmallVector<Value> update_indices;
          auto start_indices = ProvideParameterRange(
              root_computation, dus_instr,
              dus_instr->first_index_operand_number(), update_shape.rank(), {},
              call_targets, entry_function, b);
          for (int i = 0; i < update_shape.rank(); ++i) {
            int64_t update_size = update_shape.dimensions(i);
            auto start_index = ClampIndex(
                start_indices[i],
                primitive_util::IsUnsignedIntegralType(
                    dus_instr
                        ->operand(i + dus_instr->first_index_operand_number())
                        ->shape()
                        .element_type()),
                dus_instr->shape().dimensions(i) - update_size, b);

            update_indices.push_back(
                b.create<AddIOp>(input_indices[i], start_index));
          }

          auto updated_value =
              ProvideParameter(root_computation, dus_instr, kDUSUpdateIndex,
                               input_indices, call_targets, entry_function, b);
          // Handle bitcasts under the DUS.
          if (dus_instr->shape() != root->shape()) {
            update_indices = ApplyAffineMap(
                GetBitcastMap(dus_instr->shape(), root->shape(), b.getContext())
                    .GetAffineMap(),
                update_indices, {}, b);
          }
          results.push_back(
              b.create<InsertOp>(updated_value, output, update_indices));
        }
        return results;
      });

  b.create<ReturnOp>(result_tensors);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
