/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/loop_kernel_emitter.h"

#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_item.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::emitters {

LoopFusionKernelEmitter::LoopFusionKernelEmitter(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const HloFusionSpec& fusion_spec, const BufferAssignment* buffer_assignment,
    KernelArguments::BufferAlignment buffer_alignment,
    WorkDimensions work_dimensions, int32_t unroll_factor,
    absl::string_view entry_function_name, BackendKind backend_kind)
    : mlir_context_(mlir_context),
      fusion_(fusion),
      fusion_spec_(fusion_spec),
      buffer_assignment_(buffer_assignment),
      buffer_alignment_(std::move(buffer_alignment)),
      work_dimensions_(std::move(work_dimensions)),
      unroll_factor_(unroll_factor),
      entry_function_name_(entry_function_name),
      backend_kind_(backend_kind) {}

absl::StatusOr<MlirKernelDefinition>
LoopFusionKernelEmitter::EmitKernelDefinition() {
  mlir::OpBuilder builder(&mlir_context_);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion_.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(
      loc, absl::StrCat(fusion_.name(), "_kernel_module"));

  emitters::SetIndexDataLayout(*module, fusion_);

  TF_ASSIGN_OR_RETURN(
      mlir::func::FuncOp entry_func,
      emitters::EmitKernelApi(*module, fusion_, buffer_assignment_,
                              buffer_alignment_, entry_function_name_));
  SetBackendKind(&mlir_context_, entry_func, backend_kind_);

  // Loop emitters don't support epilogues.
  emitters::PartitionedComputations computations(
      fusion_.fused_instructions_computation(), module->getContext());
  TF_ASSIGN_OR_RETURN(auto call_targets, emitters::EmitPartitionedComputations(
                                             *module, computations));

  TF_RETURN_IF_ERROR(
      EmitEntryFunction(computations, call_targets, entry_func, fusion_));

  TF_ASSIGN_OR_RETURN(auto kernel_spec, GetKernelSpec());

  return MlirKernelDefinition(std::move(kernel_spec),
                              MlirKernelSource(std::move(module)));
}

IndexingMap LoopFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
    const WorkDimensions& work_dimensions, int32_t unroll_factor,
    const Shape& root_shape, mlir::MLIRContext* ctx) {
  return GetDefaultWorkItemIndexingMap(work_dimensions, unroll_factor,
                                       root_shape, ctx);
}

IndexingMap LoopFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
    mlir::MLIRContext* ctx) const {
  return ComputeWorkItemIdToOutputIndexing(work_dimensions_, unroll_factor_,
                                           GetIndexingShape(fusion_spec_), ctx);
}

Shape LoopFusionKernelEmitter::GetIndexingShape(
    const HloFusionSpec& fusion_spec) {
  // Use the first root shape as the indexing shape.
  const Shape& root_0_shape = fusion_spec.fusion_root(0).shape();
  // Use the first shape of the first root if it's a tuple.
  // TODO(willfroom): Should we add some validation to ensure all shapes are
  // bitcastable?
  return root_0_shape.IsTuple() ? root_0_shape.tuple_shapes(0) : root_0_shape;
}

absl::StatusOr<KernelSpec> LoopFusionKernelEmitter::GetKernelSpec() const {
  if (buffer_assignment_ == nullptr) {
    return KernelSpec(entry_function_name_, work_dimensions_,
                      KernelSpec::Buffers(), KernelSpec::Buffers(),
                      absl::flat_hash_set<int64_t>());
  }

  KernelSpec::Buffers result_buffers;
  for (auto& indexed : ShapeUtil::GetLeafShapes(fusion_.shape())) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        buffer_assignment_->GetUniqueSlice(&fusion_, indexed.index));
    result_buffers.push_back(std::move(slice));
  }

  KernelSpec::Buffers argument_buffers;
  absl::flat_hash_set<int64_t> invariant_arguments;
  int64_t operand_index = 0;
  for (HloInstruction* operand : fusion_.operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      TF_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          buffer_assignment_->GetUniqueSlice(operand, indexed.index));

      bool invariant = absl::c_none_of(
          result_buffers,
          [&slice](const BufferAllocation::Slice& result_slice) {
            return result_slice.OverlapsWith(slice);
          });
      if (invariant) {
        invariant_arguments.insert(operand_index);
      }

      argument_buffers.push_back(std::move(slice));
      ++operand_index;
    }
  }

  return KernelSpec(entry_function_name_, work_dimensions_,
                    std::move(argument_buffers), std::move(result_buffers),
                    std::move(invariant_arguments));
}

absl::Status LoopFusionKernelEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  mlir::MLIRContext* context = builder.getContext();

  auto workgroup_ids =
      EmitWorkGroupIds(builder, work_dimensions_.num_work_groups);

  auto indexing = ComputeWorkItemIdToOutputIndexing(context);

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);
  llvm::SmallVector<const Shape*> result_shapes;
  for (const HloInstructionAdaptor& root : fusion_spec_.fusion_roots()) {
    if (root.shape().IsTuple()) {
      for (const auto& shape : root.shape().tuple_shapes()) {
        result_shapes.push_back(&shape);
      }
    } else {
      result_shapes.push_back(&root.shape());
    }
  }

  auto body_builder =
      [&](mlir::ImplicitLocOpBuilder& nested_b, mlir::ValueRange symbol_values,
          mlir::ValueRange map_results,
          mlir::ValueRange output_tensors) -> llvm::SmallVector<mlir::Value> {
    auto root_fn = call_targets(
        fusion.fused_instructions_computation()->root_instruction());
    // Generate the operands for the root function: input tensors +
    // output indices.
    llvm::SmallVector<mlir::Value> operands(
        entry_function.getArguments().take_front(num_inputs));
    absl::c_copy(map_results, std::back_inserter(operands));
    auto result_scalars =
        nested_b.create<PureCallOp>(root_fn, operands).getResults();

    llvm::SmallVector<mlir::Value> result_tensors;
    result_tensors.reserve(output_tensor_args.size());
    for (auto [root_shape, tensor, value] :
         llvm::zip(result_shapes, output_tensors, result_scalars)) {
      llvm::SmallVector<mlir::Value> output_indices = emitters::ApplyIndexing(
          GetBitcastMap(*result_shapes.front(), *root_shape, context),
          map_results, {}, nested_b);
      result_tensors.push_back(nested_b.create<mlir::tensor::InsertOp>(
          value, tensor, output_indices));
    }
    return result_tensors;
  };

  const auto loop_builder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::ValueRange workitem_ids_and_outputs) {
    mlir::ImplicitLocOpBuilder nested_b(loc, builder);

    mlir::ValueRange workitem_ids = workitem_ids_and_outputs.take_front(3);
    mlir::ValueRange outputs = workitem_ids_and_outputs.drop_front(3);

    llvm::SmallVector<mlir::Value, 6> work_dims;
    work_dims.insert(work_dims.end(), workitem_ids.begin(), workitem_ids.end());
    work_dims.insert(work_dims.end(), workgroup_ids.begin(),
                     workgroup_ids.end());

    auto loop_results = emitters::EmitXlaLoopOp(
        nested_b, mlir::ValueRange(work_dims), outputs, indexing, body_builder);
    auto terminator = nested_b.create<mlir::scf::InParallelOp>();
    nested_b.setInsertionPointToStart(terminator.getBody());
    for (auto [result, output] : llvm::zip(loop_results, outputs)) {
      auto output_tensor = mlir::cast<mlir::RankedTensorType>(output.getType());
      llvm::SmallVector<mlir::OpFoldResult> offsets(output_tensor.getRank(),
                                                    nested_b.getIndexAttr(0));
      llvm::SmallVector<mlir::OpFoldResult> sizes =
          mlir::getAsIndexOpFoldResult(context, output_tensor.getShape());
      llvm::SmallVector<mlir::OpFoldResult> strides(output_tensor.getRank(),
                                                    nested_b.getIndexAttr(1));
      nested_b.create<mlir::tensor::ParallelInsertSliceOp>(
          result, output, offsets, sizes, strides);
    }
  };

  const NumWorkItems& num_work_items = work_dimensions_.num_work_items;
  llvm::SmallVector<mlir::OpFoldResult> upper_bounds =
      mlir::getAsIndexOpFoldResult(context,
                                   {static_cast<int64_t>(num_work_items.x),
                                    static_cast<int64_t>(num_work_items.y),
                                    static_cast<int64_t>(num_work_items.z)});
  builder.create<mlir::func::ReturnOp>(
      builder
          .create<mlir::scf::ForallOp>(upper_bounds, output_tensor_args,
                                       std::nullopt, loop_builder)
          .getResults());

  return absl::OkStatus();
}

}  // namespace xla::emitters
