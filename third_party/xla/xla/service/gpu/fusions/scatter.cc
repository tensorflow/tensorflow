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
#include "xla/service/gpu/fusions/scatter.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/fusions/loop.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_nested.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"

namespace xla {
namespace gpu {

ScatterFusion::ScatterFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis), config_(ComputeLoopFusionConfig(analysis)) {
  CHECK_EQ(analysis.fusion_roots().size(), 1);
  CHECK_EQ(analysis.fusion_roots()[0]->opcode(), HloOpcode::kScatter);
}

LaunchDimensions ScatterFusion::launch_dimensions() const {
  const auto& updates_shape =
      analysis_.fusion_roots().front()->operands().back()->shape();
  return CalculateLaunchDimensions(updates_shape, analysis_.device_info());
}

absl::Status ScatterFusion::EmitKernel(IrEmitterContext& ir_emitter_context,
                                       const HloFusionInstruction& fusion,
                                       const LaunchDimensions& launch_dims,
                                       std::vector<llvm_ir::IrArray> inputs,
                                       std::vector<llvm_ir::IrArray> outputs,
                                       llvm::IRBuilder<>* builder) const {
  GpuElementalIrEmitter elemental_emitter(ir_emitter_context, builder);
  // Spin up a new fused emitter for the scatter kernel and emit it.
  FusedIrEmitter scatter_fused_emitter(elemental_emitter);
  auto* fused_computation = fusion.fused_instructions_computation();
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    auto fused_operand = fused_computation->parameter_instruction(i);
    scatter_fused_emitter.BindGenerator(
        *fused_operand, [builder, &input = inputs[i],
                         fused_operand](llvm_ir::IrArray::Index index) {
          return input.EmitReadArrayElement(index, builder,
                                            fused_operand->name());
        });
  }

  auto* root = fused_computation->root_instruction();
  const xla::ScatterDimensionNumbers& scatter_dims =
      Cast<HloScatterInstruction>(root)->scatter_dimension_numbers();

  std::string name = llvm_ir::IrName(root);
  const Shape& operand_shape = root->operand(0)->shape();
  const Shape& scatter_indices_shape = root->operand(1)->shape();
  const Shape& updates_shape = root->operand(2)->shape();
  const HloComputation& update_computation = *root->called_computations()[0];

  TF_ASSIGN_OR_RETURN(auto scatter_indices_gen,
                      scatter_fused_emitter.GetGenerator(*root->operand(1)));
  TF_ASSIGN_OR_RETURN(auto updates_gen,
                      scatter_fused_emitter.GetGenerator(*root->operand(2)));

  auto loop_body_emitter =
      [&](const llvm_ir::IrArray::Index& index) -> absl::Status {
    std::vector<llvm::Value*> raw_window_multidim;
    std::vector<llvm::Value*> input_scatter_multidim;
    std::vector<int64_t> raw_window_bounds;

    auto get_i64_array = [](absl::Span<const int64_t> container) {
      return llvm::ArrayRef<int64_t>{container.data(),
                                     static_cast<size_t>(container.size())};
    };

    llvm::ArrayRef<int64_t> update_window_dims =
        get_i64_array(scatter_dims.update_window_dims());
    // Partition the index into window indices and scatter indices.
    for (int64_t i = 0, e = index.size(); i != e; ++i) {
      // For window indices also remember the window size, this comes in handy
      // later.
      if (llvm::is_contained(update_window_dims, i)) {
        raw_window_multidim.push_back(index[i]);
        raw_window_bounds.push_back(updates_shape.dimensions(i));
      } else {
        input_scatter_multidim.push_back(index[i]);
      }
    }
    DCHECK_EQ(raw_window_multidim.size(),
              scatter_dims.update_window_dims_size());

    // Apply inserted_window_dims to the window dimensions.
    int64_t raw_window_multidim_idx = 0;
    llvm::SmallVector<llvm::Value*> input_window_multidim;
    llvm::SmallVector<int64_t> input_window_bounds;
    const int64_t rank = operand_shape.rank();
    input_window_bounds.reserve(rank);
    input_window_multidim.reserve(rank);

    llvm::ArrayRef<int64_t> inserted_window_dims =
        get_i64_array(scatter_dims.inserted_window_dims());
    for (int64_t i = 0; i != rank; ++i) {
      if (llvm::is_contained(inserted_window_dims, i)) {
        input_window_bounds.push_back(1);  // Trivial dimension.
        input_window_multidim.push_back(index.GetConstantWithIndexType(0));
      } else {
        input_window_bounds.push_back(
            raw_window_bounds[raw_window_multidim_idx]);
        input_window_multidim.push_back(
            raw_window_multidim[raw_window_multidim_idx]);
        ++raw_window_multidim_idx;
      }
    }
    DCHECK_EQ(input_window_multidim.size(), operand_shape.rank());

    // Insert a 1 dimension at the end if index_vector_dim requests one.
    Shape scatter_indices_shape_fixed = scatter_indices_shape;
    if (scatter_dims.index_vector_dim() == scatter_indices_shape.rank()) {
      scatter_indices_shape_fixed.add_dimensions(1);
      scatter_indices_shape_fixed.mutable_layout()->add_minor_to_major(
          scatter_dims.index_vector_dim());
    }

    // Now load the indices corresponding to the current window from
    // scatter_indices.
    std::vector<llvm::Value*> raw_scatter_index_multidim =
        input_scatter_multidim;
    raw_scatter_index_multidim.insert(
        raw_scatter_index_multidim.begin() + scatter_dims.index_vector_dim(),
        nullptr);

    llvm::ArrayRef<int64_t> scatter_dims_to_operand_dims =
        get_i64_array(scatter_dims.scatter_dims_to_operand_dims());
    llvm::Value* is_in_bounds = builder->getTrue();
    for (int64_t i = 0, e = scatter_dims_to_operand_dims.size(); i != e; ++i) {
      // Our index is stored along index_vector_dim, insert that into the lookup
      // index into scatter_indices.
      raw_scatter_index_multidim[scatter_dims.index_vector_dim()] =
          index.GetConstantWithIndexType(i);
      llvm_ir::IrArray::Index raw_scatter_index_index(
          raw_scatter_index_multidim, scatter_indices_shape_fixed,
          index.GetType());

      int64_t operand_dim = scatter_dims_to_operand_dims[i];
      if (operand_dim > rank) {
        return absl::OutOfRangeError(
            "The provided scatter_dims_to_operand_dims was out of range.");
      }
      TF_ASSIGN_OR_RETURN(
          llvm::Value* const loaded_scatter_index,
          scatter_indices_gen(raw_scatter_index_index.SourceIndexOfReshape(
              scatter_indices_shape_fixed, scatter_indices_shape, builder)));
      // And add the index to our window index. This yields the output index.
      llvm::Value* casted_scatter_index = builder->CreateIntCast(
          loaded_scatter_index, index.GetType(),
          /*isSigned=*/ShapeUtil::ElementIsSigned(scatter_indices_shape));
      llvm::Value* dim_offset = builder->CreateAdd(
          input_window_multidim[operand_dim], casted_scatter_index);
      input_window_multidim[operand_dim] = dim_offset;

      // Also do the bounds check now.
      int64_t max_index = operand_shape.dimensions(operand_dim) -
                          input_window_bounds[operand_dim] + 1;
      // is_in_bounds = index >= 0 && index < dim_size-window_size+1
      //   --> index u< dim_size-window_size+1
      is_in_bounds = builder->CreateAnd(
          is_in_bounds,
          builder->CreateICmpULT(casted_scatter_index,
                                 index.GetConstantWithIndexType(max_index)));
    }

    llvm_ir::LlvmIfData if_window_in_bounds_data = llvm_ir::EmitIfThenElse(
        is_in_bounds, "scatter.in_bounds", builder, /*emit_else=*/false);
    llvm_ir::SetToFirstInsertPoint(if_window_in_bounds_data.true_block,
                                   builder);
    // All done, now just read from the calculated input from the window, and do
    // an atomic store to the calculated location in the output.
    llvm_ir::IrArray::Index input_window_index(
        input_window_multidim, outputs.back().GetShape(), index.GetType());
    llvm::Value* output_address =
        outputs.back().EmitArrayElementAddress(input_window_index, builder);
    llvm::Value* input_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(updates_shape.element_type(),
                                       ir_emitter_context.llvm_module()),
        "input_address", builder);
    TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value, updates_gen(index));
    builder->CreateStore(input_ir_value, input_address);

    if (root->unique_indices()) {
      return CallNestedComputation(
          builder, ir_emitter_context, update_computation,
          {output_address, input_address}, output_address);
    }
    return EmitAtomicOperationForNestedComputation(
        builder, ir_emitter_context, update_computation, output_address,
        input_address, outputs.back().GetElementLlvmType());
  };

  // Launch a kernel that reads every element in the updates tensor. We could
  // also do one kernel per window instead if bounds checks turn out to be a
  // bottleneck.
  auto index_type =
      GetIndexTypeForKernel(root, launch_dims.launch_bound(), builder);
  return ParallelLoopEmitter(loop_body_emitter, updates_shape, launch_dims,
                             builder)
      .EmitLoop(name, index_type);
}

std::optional<IndexingMap> ScatterFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  auto* scatter =
      DynCast<HloScatterInstruction>(analysis_.fusion_heroes().front());
  int64_t scatter_operand_count = scatter->scatter_operand_count();
  // Scatter operands a packed in the following way:
  // Operand IDs [0, scatter_operand_count - 1] for `scatter operands`.
  // Operand ID  scatter_operand_count for `scatter indices`.
  // Operand IDs [scatter_operand_count + 1, 2 * scatter_operand_count] for
  // `scatter updates`.

  // For scatter operands we do not know the thread ID indexing.
  if (hero_operand_index < scatter_operand_count) {
    return std::nullopt;
  }
  // Compute thread id mapping based on the first update operand.
  Shape scatter_update_shape = scatter->scatter_updates().front()->shape();
  IndexingMap scatter_update_map = GetDefaultThreadIdIndexingMap(
      launch_dimensions(), config_.unroll_factor, scatter_update_shape, ctx);

  // For scatter indices we project indexing for scatter updates and take the
  // first result of the affine map only, because they coincide.
  if (hero_operand_index == scatter_operand_count) {
    Shape scatter_indices_shape = scatter->scatter_indices()->shape();
    CHECK_EQ(scatter_indices_shape.rank(), 2) << scatter->ToString();
    // Create a map from scatter update to scatter indices.
    IndexingMap updates_to_indices_map{
        mlir::AffineMap::get(
            /*dimCount=*/scatter_update_shape.rank(), /*symbolCount=*/1,
            {mlir::getAffineDimExpr(0, ctx), mlir::getAffineSymbolExpr(0, ctx)},
            ctx),
        DimVarsFromTensorSizes(scatter_update_shape.dimensions()),
        RangeVarsFromTensorSizes({scatter_indices_shape.dimensions(1)}),
        /*rt_vars=*/{}};
    auto scatter_indices_map = scatter_update_map * updates_to_indices_map;
    scatter_indices_map.Simplify(GetIndexingMapForInstruction);
    return scatter_indices_map;
  }
  return scatter_update_map;
}

}  // namespace gpu
}  // namespace xla
