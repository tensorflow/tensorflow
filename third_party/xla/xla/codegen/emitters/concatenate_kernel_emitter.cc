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

#include "xla/codegen/emitters/concatenate_kernel_emitter.h"

#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
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
#include "xla/codegen/emitters/utils.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/kernel_spec.h"
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

using ::mlir::MLIRContext;

ConcatenateFusionKernelEmitter::ConcatenateFusionKernelEmitter(
    MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const HloFusionSpec& fusion_spec, const BufferAssignment* buffer_assignment,
    KernelArguments::BufferAlignment buffer_alignment,
    WorkDimensions work_dimensions, absl::string_view entry_function_name,
    BackendKind backend_kind)
    : mlir_context_(mlir_context),
      fusion_(fusion),
      fusion_spec_(fusion_spec),
      buffer_assignment_(buffer_assignment),
      buffer_alignment_(std::move(buffer_alignment)),
      work_dimensions_(std::move(work_dimensions)),
      largest_shape_(GetIndexingShape(fusion_spec_)),
      entry_function_name_(entry_function_name),
      backend_kind_(backend_kind) {}

absl::StatusOr<ConcatenateFusionKernelEmitter::KernelDefinition>
ConcatenateFusionKernelEmitter::EmitKernelDefinition() {
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

  std::vector<emitters::EpilogueSpecification> epilogues =
      GetEpilogues(fusion_, &mlir_context_);
  emitters::PartitionedComputations computations(
      fusion_.fused_instructions_computation(), &mlir_context_, epilogues);
  TF_ASSIGN_OR_RETURN(auto call_targets, emitters::EmitPartitionedComputations(
                                             *module, computations));

  TF_RETURN_IF_ERROR(
      EmitEntryFunction(computations, call_targets, entry_func, fusion_));

  TF_ASSIGN_OR_RETURN(auto kernel_spec,
                      GetKernelSpec(entry_function_name_, fusion_,
                                    buffer_assignment_, work_dimensions_));

  return KernelDefinition(std::move(kernel_spec),
                          MlirKernelSource(std::move(module)));
}

const Shape& ConcatenateFusionKernelEmitter::GetIndexingShape(
    const HloFusionSpec& fusion_spec) {
  const HloInstruction& concat = fusion_spec.fusion_hero(0).instruction();
  int64_t dim = concat.concatenate_dimension();
  auto less = [&](const HloInstruction* lhs, const HloInstruction* rhs) {
    return lhs->shape().dimensions(dim) < rhs->shape().dimensions(dim);
  };
  HloInstruction* operand = *absl::c_max_element(concat.operands(), less);
  return operand->shape();
}

int ConcatenateFusionKernelEmitter::GetValidUnrollFactor(
    const HloFusionSpec& fusion_spec, int max_unroll_factor) {
  auto& concat = fusion_spec.fusion_hero(0).instruction();
  int unroll_factor = max_unroll_factor;
  int64_t dim = concat.concatenate_dimension();
  for (const HloInstruction* operand : concat.operands()) {
    if (unroll_factor == 1) {
      return 1;
    }
    unroll_factor = std::gcd(unroll_factor, operand->shape().dimensions(dim));
  }
  return unroll_factor;
}

IndexingMap ConcatenateFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
    const WorkDimensions& work_dimensions, const Shape& largest_shape,
    MLIRContext* ctx) {
  return GetDefaultWorkItemIndexingMap(work_dimensions, largest_shape, ctx);
}

IndexingMap ConcatenateFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
    MLIRContext* ctx) const {
  return ComputeWorkItemIdToOutputIndexing(work_dimensions_, largest_shape_,
                                           ctx);
}

absl::Status ConcatenateFusionKernelEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());

  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  auto workgroup_ids =
      EmitWorkGroupIds(builder, work_dimensions_.num_work_groups);

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  llvm::SmallVector<mlir::Value> input_tensors(
      entry_function.getArguments().take_front(num_inputs));
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  llvm::SmallVector<mlir::Value> result_tensors{output_tensor_args.begin(),
                                                output_tensor_args.end()};

  auto work_item_id_to_input_map =
      ComputeWorkItemIdToOutputIndexing(&mlir_context_);
  auto epilogue_indexing = ComputeEpilogueInputToOutputIndexing(
      fusion_spec_.fusion_hero(0), fusion_spec_.fusion_root(0), &mlir_context_);

  const auto* concat = &fusion_spec_.fusion_hero(0).instruction();

  const auto forall_body_builder = [&](mlir::OpBuilder& builder,
                                       mlir::Location loc,
                                       mlir::ValueRange
                                           workitem_ids_and_outputs) {
    mlir::ImplicitLocOpBuilder nested_b(loc, builder);

    mlir::ValueRange workitem_ids = workitem_ids_and_outputs.take_front(3);
    mlir::ValueRange outputs = workitem_ids_and_outputs.drop_front(3);

    llvm::SmallVector<mlir::Value, 6> work_dims;
    work_dims.insert(work_dims.end(), workitem_ids.begin(), workitem_ids.end());
    work_dims.insert(work_dims.end(), workgroup_ids.begin(),
                     workgroup_ids.end());

    for (auto [operand_index, operand] : llvm::enumerate(concat->operands())) {
      IndexingMap input_to_output_map =
          ComputeInputToOutputIndexing(concat, /*input_id=*/operand_index,
                                       &mlir_context_)
              .indexing_maps.front()
              .begin()
              ->map();
      auto thread_id_to_output_map = ComposeIndexingMaps(
          ComposeIndexingMaps(work_item_id_to_input_map, input_to_output_map),
          epilogue_indexing);
      thread_id_to_output_map.Simplify();

      auto loop_nest_body_builder = [&, operand_index = operand_index](
                                        mlir::ImplicitLocOpBuilder& nested_b,
                                        mlir::ValueRange symbol_values,
                                        mlir::ValueRange output_indices,
                                        mlir::ValueRange output_tensors)
          -> llvm::SmallVector<mlir::Value> {
        auto input_indices = emitters::ApplyIndexing(
            work_item_id_to_input_map, work_dims, symbol_values, nested_b);

        auto result_scalar = emitters::ProvideParameter(
            root_computation, concat, operand_index, input_indices,
            call_targets, entry_function, nested_b);
        absl::flat_hash_map<const HloInstruction*,
                            llvm::SmallVector<mlir::Value>>
            hero_value{{concat, result_scalar}};
        auto result_scalars = EmitEpilogue(
            /*epilogue_index=*/0, computations, entry_function, hero_value,
            output_indices,
            nested_b)[&fusion_spec_.fusion_root(0).instruction()];

        llvm::SmallVector<mlir::Value> result_tensors;
        result_tensors.reserve(output_tensor_args.size());
        for (auto [tensor, value] : llvm::zip(output_tensors, result_scalars)) {
          result_tensors.push_back(
              nested_b
                  .create<mlir::tensor::InsertOp>(value, tensor, output_indices)
                  .getResult());
        }

        return result_tensors;
      };
      result_tensors = emitters::EmitXlaLoopOp(
          nested_b, work_dims, result_tensors, thread_id_to_output_map,
          loop_nest_body_builder);
    }

    auto terminator = nested_b.create<mlir::scf::InParallelOp>();
    nested_b.setInsertionPointToStart(terminator.getBody());
    for (auto [result, output] : llvm::zip(result_tensors, outputs)) {
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
                                       std::nullopt, forall_body_builder)
          .getResults());

  return absl::OkStatus();
}

std::vector<emitters::EpilogueSpecification>
ConcatenateFusionKernelEmitter::GetEpilogues(const HloFusionInstruction& fusion,
                                             MLIRContext* mlir_context) const {
  return {emitters::EpilogueSpecification::FromIdentityIndexing(
      &fusion_spec_.fusion_hero(0).instruction(),
      &fusion_spec_.fusion_root(0).instruction(), mlir_context)};
}

}  // namespace xla::emitters
