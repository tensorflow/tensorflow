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

#include "xla/codegen/emitters/dynamic_update_slice_kernel_emitter.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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
#include "xla/codegen/ir_emission_utils.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_item.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::emitters {

using ::mlir::MLIRContext;

constexpr int kDUSUpdateIndex = 1;

DynamicUpdateSliceKernelEmitter::DynamicUpdateSliceKernelEmitter(
    MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const HloFusionSpec& fusion_spec, const BufferAssignment* buffer_assignment,
    KernelArguments::BufferAlignment buffer_alignment,
    WorkDimensions work_dimensions, absl::string_view entry_function_name,
    BackendKind backend_kind)
    : mlir_context_(mlir_context),
      fusion_(fusion),
      fusion_spec_(fusion_spec),
      dus_ops_(
          GetOutputDefiningDynamicUpdateSlices(fusion_spec.fusion_roots())),
      buffer_assignment_(buffer_assignment),
      buffer_alignment_(std::move(buffer_alignment)),
      work_dimensions_(std::move(work_dimensions)),
      entry_function_name_(entry_function_name),
      backend_kind_(backend_kind) {}

absl::StatusOr<DynamicUpdateSliceKernelEmitter::KernelDefinition>
DynamicUpdateSliceKernelEmitter::EmitKernelDefinition() {
  mlir::OpBuilder builder(&mlir_context_);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion_.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(
      loc, absl::StrCat(fusion_.name(), "_kernel_module"));

  bool force_64_bit = backend_kind_ == BackendKind::kCpu;
  emitters::SetIndexDataLayout(*module, fusion_, force_64_bit);

  TF_ASSIGN_OR_RETURN(
      mlir::func::FuncOp entry_func,
      emitters::EmitKernelApi(*module, fusion_, buffer_assignment_,
                              buffer_alignment_, entry_function_name_));
  SetBackendKind(&mlir_context_, entry_func, backend_kind_);

  emitters::PartitionedComputations computations(
      fusion_.fused_instructions_computation(), &mlir_context_, GetEpilogues());
  TF_ASSIGN_OR_RETURN(auto call_targets, emitters::EmitPartitionedComputations(
                                             *module, computations));

  TF_RETURN_IF_ERROR(
      EmitEntryFunction(computations, call_targets, entry_func, fusion_));

  TF_ASSIGN_OR_RETURN(auto kernel_spec, GetKernelSpec());

  return KernelDefinition(std::move(kernel_spec),
                          MlirKernelSource(std::move(module)));
}

IndexingMap DynamicUpdateSliceKernelEmitter::ComputeWorkItemIdToInputIndexing(
    MLIRContext* mlir_context) const {
  // It is guaranteed that all DUS ops have the same output shape at this point.
  const auto& update_shape =
      dus_ops_.front().GetOperand(kDUSUpdateIndex).shape();
  return ComputeWorkItemIdToOutputIndexing(work_dimensions_, update_shape,
                                           mlir_context);
}

Shape DynamicUpdateSliceKernelEmitter::GetIndexingShape(
    const HloFusionSpec& fusion_spec) {
  auto dus_ops =
      GetOutputDefiningDynamicUpdateSlices(fusion_spec.fusion_roots());
  return dus_ops.front().GetOperand(kDUSUpdateIndex).shape();
}

IndexingMap DynamicUpdateSliceKernelEmitter::ComputeWorkItemIdToOutputIndexing(
    const WorkDimensions& work_dimensions, const Shape& update_shape,
    MLIRContext* mlir_context) {
  return GetDefaultWorkItemIndexingMap(work_dimensions, update_shape,
                                       mlir_context);
}

absl::StatusOr<KernelSpec> DynamicUpdateSliceKernelEmitter::GetKernelSpec()
    const {
  if (buffer_assignment_ == nullptr) {
    return KernelSpec(entry_function_name_, work_dimensions_,
                      KernelSpec::Buffers(), KernelSpec::Buffers(),
                      absl::flat_hash_set<int64_t>());
  }

  KernelSpec::Buffers result_buffers;
  for (ShapeUtil::IndexedShape& indexed :
       ShapeUtil::GetLeafShapes(fusion_.shape())) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        buffer_assignment_->GetUniqueSlice(&fusion_, indexed.index));
    result_buffers.push_back({slice, indexed.shape});
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
          result_buffers, [&slice](const ShapedSlice& result_slice) {
            return result_slice.slice.OverlapsWith(slice);
          });
      if (invariant) {
        invariant_arguments.insert(operand_index);
      }

      argument_buffers.push_back({slice, indexed.shape});
      ++operand_index;
    }
  }

  return KernelSpec(entry_function_name_, work_dimensions_,
                    std::move(argument_buffers), std::move(result_buffers),
                    std::move(invariant_arguments));
}

absl::Status DynamicUpdateSliceKernelEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  auto indexing = ComputeWorkItemIdToInputIndexing(&mlir_context_);
  indexing.Simplify();
  indexing.RemoveUnusedSymbols();

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());

  auto body_builder =
      [&](mlir::ImplicitLocOpBuilder& nested_b, mlir::ValueRange symbol_values,
          mlir::ValueRange input_indices,
          mlir::ValueRange output_tensors) -> llvm::SmallVector<mlir::Value> {
    llvm::SmallVector<mlir::Value> results;
    for (auto [instr, root, output] :
         llvm::zip(dus_ops_, fusion_spec_.fusion_roots(), output_tensors)) {
      const auto* dus_instr =
          Cast<HloDynamicUpdateSliceInstruction>(&instr.instruction());
      const auto& update_shape = dus_instr->update()->shape();
      llvm::SmallVector<mlir::Value> update_indices;
      auto start_indices = ProvideParameterRange(
          root_computation, dus_instr, dus_instr->first_index_operand_number(),
          update_shape.dimensions().size(), {}, call_targets, entry_function,
          nested_b);
      for (int i = 0; i < update_shape.dimensions().size(); ++i) {
        int64_t update_size = update_shape.dimensions(i);
        auto start_index = ClampIndex(
            start_indices[i],
            primitive_util::IsUnsignedIntegralType(
                dus_instr->operand(i + dus_instr->first_index_operand_number())
                    ->shape()
                    .element_type()),
            dus_instr->shape().dimensions(i) - update_size, nested_b);

        update_indices.push_back(nested_b.create<mlir::arith::AddIOp>(
            input_indices[i], start_index));
      }

      auto updated_value = ProvideParameter(
          root_computation, dus_instr, kDUSUpdateIndex, input_indices,
          call_targets, entry_function, nested_b);
      // Handle bitcasts under the DUS.
      if (dus_instr->shape() != root.shape()) {
        update_indices = ApplyIndexing(
            GetBitcastMap(dus_instr->shape(), root.shape(), &mlir_context_),
            update_indices, {}, nested_b);
      }
      results.push_back(nested_b.create<mlir::tensor::InsertOp>(
          updated_value[0], output, update_indices));
    }
    return results;
  };

  auto workgroup_ids =
      EmitWorkGroupIds(builder, work_dimensions_.num_work_groups);

  const auto forall_builder = [&](mlir::OpBuilder& builder, mlir::Location loc,
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
          mlir::getAsIndexOpFoldResult(&mlir_context_,
                                       output_tensor.getShape());
      llvm::SmallVector<mlir::OpFoldResult> strides(output_tensor.getRank(),
                                                    nested_b.getIndexAttr(1));
      nested_b.create<mlir::tensor::ParallelInsertSliceOp>(
          result, output, offsets, sizes, strides);
    }
  };

  const NumWorkItems& num_work_items = work_dimensions_.num_work_items;
  llvm::SmallVector<mlir::OpFoldResult> upper_bounds =
      mlir::getAsIndexOpFoldResult(&mlir_context_,
                                   {static_cast<int64_t>(num_work_items.x),
                                    static_cast<int64_t>(num_work_items.y),
                                    static_cast<int64_t>(num_work_items.z)});
  builder.create<mlir::func::ReturnOp>(
      builder
          .create<mlir::scf::ForallOp>(upper_bounds, output_tensor_args,
                                       std::nullopt, forall_builder)
          .getResults());
  return absl::OkStatus();
}

std::vector<emitters::EpilogueSpecification>
DynamicUpdateSliceKernelEmitter::GetEpilogues() const {
  std::vector<emitters::EpilogueSpecification> epilogues;
  for (const auto& [dus_op, root] :
       llvm::zip(dus_ops_, fusion_spec_.fusion_roots())) {
    epilogues.push_back(emitters::EpilogueSpecification::FromIdentityIndexing(
        &dus_op.instruction(), &root.instruction(), &mlir_context_));
  }
  return epilogues;
}

}  // namespace xla::emitters
