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

#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_attrs.h.inc"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/expanders/scatter_simplifier.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::MLIRContext;
using mlir::Value;
using mlir::ValueRange;

namespace ma = ::mlir::arith;
namespace scf = ::mlir::scf;

SmallVector<Value> EmitScatterComputation(
    int64_t num_threads, const HloScatterInstruction* scatter,
    ValueRange indices, ValueRange update_elems, ValueRange output_tensors,
    const emitters::CallTargetProvider& call_targets,
    mlir::ImplicitLocOpBuilder& b) {
  mlir::func::FuncOp reducer =
      call_targets(scatter->called_computations()[0]->root_instruction());
  if (scatter->unique_indices() || num_threads == 1 ||
      scatter->scatter_operand_count() > 1) {
    SmallVector<Value> computation_args;
    computation_args.reserve(output_tensors.size() + update_elems.size());
    for (Value output_tensor : output_tensors) {
      computation_args.push_back(
          mlir::tensor::ExtractOp::create(b, output_tensor, indices));
    }
    computation_args.append(update_elems.begin(), update_elems.end());
    SmallVector<Value> reduced_values =
        emitters::InlineBlock(b, reducer.getBody().front(), computation_args);
    SmallVector<Value> ret;
    ret.reserve(reduced_values.size());
    for (const auto& [reduced_value, output_tensor] :
         llvm::zip(reduced_values, output_tensors)) {
      ret.push_back(mlir::tensor::InsertOp::create(b, reduced_value,
                                                   output_tensor, indices));
    }
    return ret;
  }
  // Multi workgroup CPU scatter path (active when num_threads > 1).
  CHECK_EQ(output_tensors.size(), 1);
  CHECK_EQ(update_elems.size(), 1);
  return {emitters::EmitScatterAtomicUpdate(
      b, scatter->called_computations()[0], reducer, update_elems.front(),
      output_tensors.front(), indices)};
}

int64_t ComputeVectorSize(const HloScatterInstruction* scatter) {
  int64_t num_slice_elements = Product(
      scatter->scatter_updates().front()->shape().dimensions().subspan(1));
  int64_t max_element_bytes = 1;
  for (const HloInstruction* inst : llvm::concat<HloInstruction* const>(
           scatter->scatter_operands(), scatter->scatter_updates())) {
    max_element_bytes = std::max<int64_t>(
        max_element_bytes,
        ShapeUtil::ByteSizeOfPrimitiveType(inst->shape().element_type()));
  }
  constexpr int64_t kMaxVectorizedBytes = 64;
  return std::gcd(kMaxVectorizedBytes / max_element_bytes, num_slice_elements);
}

}  // namespace

std::vector<emitters::EpilogueSpecification> CpuScatterFusion::GetEpilogues(
    const HloFusionInstruction& fusion, MLIRContext* mlir_context) const {
  const HloInstruction* scatter = fusion_->fused_expression_root();
  // We don't actually support epilogues for scatter, but this is how we tell
  // the base class that we don't want it to generate code for the scatter.
  return {emitters::EpilogueSpecification::FromIdentityIndexing(
      scatter, scatter, mlir_context)};
}

CpuScatterFusion::CpuScatterFusion(const BufferAssignment& buffer_assignment,
                                   const HloFusionInstruction* fusion,
                                   MLIRContext* mlir_context)
    : buffer_assignment_(buffer_assignment),
      fusion_(fusion),
      mlir_context_(mlir_context),
      num_threads_(1) {
  if (VLOG_IS_ON(5)) {
    llvm::errs() << "\nnum_threads_: " << num_threads_ << "\n\n";
  }
}

IndexingMap GetScatterIndexingMap(
    absl::Span<const int64_t> updates_operand_shape, int64_t num_threads,
    int64_t vector_size, mlir::MLIRContext* context) {
  CHECK(!updates_operand_shape.empty())
      << "updates_operand_shape must contain at least the scatter dimension";
  for (int64_t dim : updates_operand_shape) {
    CHECK_GE(dim, 0);
  }
  CHECK_GE(num_threads, 1);
  CHECK_GE(vector_size, 1);
  CHECK_NE(context, nullptr);

  SymbolicExpr thread_expr = CreateDimExpr(0, context);
  SymbolicExpr index_id =
      CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/1, context);
  SymbolicExpr slice_linear_index =
      CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/1, context);
  SymbolicExpr vector_element_id =
      CreateSymbolExpr(/*symbol_id=*/2, /*num_dims=*/1, context);

  int64_t num_updates = updates_operand_shape.front();
  int64_t num_updates_per_thread = CeilOfRatio(num_updates, num_threads);
  absl::Span<const int64_t> slice_shape = updates_operand_shape.subspan(1);
  int64_t num_slice_elements = Product(slice_shape);
  int64_t num_vectors_per_slice = CeilOfRatio(num_slice_elements, vector_size);

  // Loop w.r.t. indices.
  SymbolicExpr updates_id_expr =
      thread_expr * num_updates_per_thread + index_id;
  SymbolicExpr slice_linear_index_expr =
      slice_linear_index * vector_size + vector_element_id;
  SmallVector<SymbolicExpr, 4> indices_in_tile =
      DelinearizeInBoundsIndex(slice_linear_index_expr, slice_shape);
  SmallVector<SymbolicExpr> result{updates_id_expr};
  result.append(indices_in_tile.begin(), indices_in_tile.end());

  SmallVector<std::pair<SymbolicExpr, Interval>, 4> constraints{
      {updates_id_expr, {0, num_updates - 1}},
      {slice_linear_index_expr, {0, num_slice_elements - 1}}};

  SymbolicMap symbolic_map = SymbolicMap::Get(context, /*num_dimensions=*/1,
                                              /*num_symbols=*/3, result);
  return IndexingMap(
      symbolic_map, {IndexingMap::Variable({0, num_threads - 1, "thread_id"})},
      {IndexingMap::Variable({0, num_updates_per_thread - 1, "index_id"}),
       IndexingMap::Variable({0, num_vectors_per_slice - 1, "vector_id"}),
       IndexingMap::Variable({0, vector_size - 1, "vector_element_id"})},
      {}, constraints);
}

absl::StatusOr<CpuScatterFusion::KernelDefinition>
CpuScatterFusion::EmitKernelDefinition() {
  mlir::OpBuilder builder(mlir_context_);
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                   CreateNamedMlirModuleOp(*fusion_, builder));

  absl::string_view module_name(mlir_module->getName().value());
  emitters::SetIndexDataLayout(mlir_module.get(), *fusion_);

  mlir::StringAttr disable_loop_unrolling_attr =
      builder.getStringAttr("xla_cpu_disable_loop_unrolling");
  mlir_module->getOperation()->setAttr(
      xla::ExtraBackendOptionsAttr::name,
      builder.getAttr<xla::ExtraBackendOptionsAttr>(
          llvm::ArrayRef{disable_loop_unrolling_attr}));

  mlir_module->getOperation()->setAttr(
      xla::CpuMemoryRegionNameAttr::name,
      builder.getStringAttr(BuildModuleMemoryRegionName(name(), fusion_)));

  ABSL_ASSIGN_OR_RETURN(
      mlir::func::FuncOp entry_func,
      EmitEntryFunctionApi(mlir_module.get(), *fusion_,
                           std::string(module_name), buffer_assignment_));

  std::vector<emitters::EpilogueSpecification> epilogues =
      GetEpilogues(*fusion_, mlir_context_);
  emitters::PartitionedComputations computations(
      fusion_->fused_instructions_computation(), mlir_context_, epilogues);
  ABSL_ASSIGN_OR_RETURN(
      emitters::CallTargetProvider call_targets,
      EmitCallTargets(mlir_module.get(), *fusion_, computations, epilogues));

  ABSL_RETURN_IF_ERROR(
      EmitEntryFunction(computations, call_targets, entry_func, *fusion_));

  WorkDimensions work_dimensions;
  work_dimensions.num_work_groups.x = static_cast<uint64_t>(num_threads_);
  ABSL_ASSIGN_OR_RETURN(
      KernelSpec kernel_spec,
      emitters::GetKernelSpec(module_name, *fusion_, &buffer_assignment_,
                              work_dimensions));

  return KernelDefinition(std::move(kernel_spec),
                          MlirKernelSource(std::move(mlir_module)));
}

absl::Status CpuScatterFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto* scatter = Cast<HloScatterInstruction>(
      fusion.fused_instructions_computation()->root_instruction());
  if (VLOG_IS_ON(5)) {
    llvm::errs() << "\n\nScatter: " << scatter->ToString() << "\n\n";
  }

  absl::Span<HloInstruction* const> scatter_operands =
      scatter->scatter_operands();
  const HloInstruction* scatter_indices = scatter->scatter_indices();
  absl::Span<HloInstruction* const> scatter_updates =
      scatter->scatter_updates();

  mlir::MLIRContext* context = entry_function.getContext();
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());
  ValueRange output_args =
      entry_function.getArguments().take_back(scatter_operands.size());
  SmallVector<Value> output_tensors(output_args.begin(), output_args.end());

  const emitters::PartitionedComputation& root_computation =
      computations.FindPartitionedComputation(
          fusion.fused_instructions_computation());
  const Shape& operand_shape = scatter_operands.front()->shape();
  const Shape& update_shape = scatter_updates.front()->shape();

  for (int64_t i = 1; i < scatter_operands.size(); ++i) {
    TF_RET_CHECK(scatter_operands[i]->shape().dimensions() ==
                 operand_shape.dimensions())
        << "Variadic scatter operands must have matching dimensions.";
    TF_RET_CHECK(scatter_updates[i]->shape().dimensions() ==
                 update_shape.dimensions())
        << "Variadic scatter updates must have matching dimensions.";
  }

  // CopyInsertion shares each result buffer with its scatter operand.
  for (int64_t i = 0; i < scatter_operands.size(); ++i) {
    ShapeIndex result_index =
        fusion.shape().IsTuple() ? ShapeIndex{i} : ShapeIndex{};
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                     buffer_assignment_.GetUniqueSlice(&fusion, result_index));
    const HloInstruction* base = scatter_operands[i];
    while (base->opcode() == HloOpcode::kBitcast) {
      base = base->operand(0);
    }
    bool in_place = false;
    if (base->opcode() == HloOpcode::kParameter) {
      const HloInstruction* fusion_operand =
          fusion.operand(base->parameter_number());
      ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice operand_slice,
                       buffer_assignment_.GetUniqueSlice(fusion_operand, {}));
      in_place = operand_slice == result_slice;
    }
    if (!in_place) {
      return Internal(
          "CpuScatterFusion requires scatter operands to share result buffers");
    }
  }

  auto finish_entry = [&](ValueRange results) {
    mlir::func::ReturnOp::create(b, results);
    if (VLOG_IS_ON(5)) {
      entry_function->getParentOfType<mlir::ModuleOp>().print(
          llvm::errs(), mlir::OpPrintingFlags().useLocalScope());
      llvm::errs() << "\n";
    }
    return absl::OkStatus();
  };

  const ScatterDimensionNumbers& scatter_dims =
      scatter->scatter_dimension_numbers();
  TF_RET_CHECK(ScatterSimplifier::IsSimplifiedScatter(scatter) &&
               scatter_dims.input_batching_dims().empty())
      << "Non-simplified HLO Scatter is not supported.";

  if (ShapeUtil::IsZeroElementArray(update_shape)) {
    return finish_entry(output_tensors);
  }

  const int64_t vector_size = ComputeVectorSize(scatter);
  if (VLOG_IS_ON(5)) {
    llvm::errs() << "\nvector_size: " << vector_size << "\n\n";
  }

  IndexingMap map = GetScatterIndexingMap(update_shape.dimensions(),
                                          num_threads_, vector_size, context);
  map.Simplify();

  WorkGroupIdOp workgroup_id = WorkGroupIdOp::create(b, WorkGroupDimension::x);
  workgroup_id->setAttr("xla.range",
                        b.getIndexArrayAttr({0, num_threads_ - 1}));

  absl::Span<const int64_t> scatter_dims_to_operand_dims =
      scatter_dims.scatter_dims_to_operand_dims();
  int64_t index_vector_dim = scatter_dims.index_vector_dim();
  int64_t operand_rank = operand_shape.dimensions().size();
  int64_t index_vector_size =
      scatter_indices->shape().dimensions(index_vector_dim);

  ValueRange results = emitters::EmitXlaLoopOp(
      b, {workgroup_id}, output_tensors, map,
      [&](ImplicitLocOpBuilder& nested_b, ValueRange iv,
          ValueRange update_indices,
          ValueRange output_tensors) -> SmallVector<Value> {
        Value update_id = update_indices.front();
        Value c0 = ma::ConstantIndexOp::create(nested_b, 0);
        Value in_bounds = ma::ConstantIntOp::create(nested_b, b.getI1Type(), 1);

        SmallVector<Value, 4> update_offsets(operand_rank, c0);
        for (int64_t i = 0; i < index_vector_size; ++i) {
          SmallVector<Value, 4> indices_tensor_indices = {
              update_id, ma::ConstantIndexOp::create(nested_b, i)};
          int indices_index = scatter->scatter_operand_count();
          Value index = ProvideParameter(
              root_computation, scatter, indices_index, indices_tensor_indices,
              call_targets, entry_function, nested_b)[0];
          if (primitive_util::IsUnsignedIntegralType(
                  scatter_indices->shape().element_type())) {
            index =
                ma::IndexCastUIOp::create(nested_b, b.getIndexType(), index);
          } else {
            index = ma::IndexCastOp::create(nested_b, b.getIndexType(), index);
          }
          int64_t remapped_index = scatter_dims_to_operand_dims[i];
          Value ub = ma::ConstantIndexOp::create(
              nested_b, operand_shape.dimensions(remapped_index) -
                            update_shape.dimensions(remapped_index + 1));
          // One bounds check is enough even for signed indices: `sge 0` is
          // implied by `ule ub`, because `ub >= 0`.
          Value dim_in_bounds =
              ma::CmpIOp::create(nested_b, ma::CmpIPredicate::ule, index, ub);
          in_bounds = ma::AndIOp::create(nested_b, in_bounds, dim_in_bounds);
          update_offsets[remapped_index] = index;
        }
        ValueRange predicated_updates =
            scf::IfOp::create(
                nested_b, in_bounds,
                [&](mlir::OpBuilder& then_builder,
                    mlir::Location then_loc) -> void {
                  ImplicitLocOpBuilder implicit_then_builder(then_loc,
                                                             then_builder);
                  SmallVector<Value, 2> update_elems = ProvideParameterRange(
                      root_computation, scatter,
                      /*start=*/scatter->scatter_operand_count() + 1,
                      /*num=*/scatter->scatter_operand_count(), update_indices,
                      call_targets, entry_function, implicit_then_builder);

                  SmallVector<Value, 4> output_indices =
                      std::move(update_offsets);
                  for (int64_t i = 0; i < output_indices.size(); ++i) {
                    output_indices[i] = ma::AddIOp::create(
                        implicit_then_builder, update_indices[i + 1],
                        output_indices[i]);
                  }
                  SmallVector<Value> updated_outputs = EmitScatterComputation(
                      num_threads_, scatter, output_indices, update_elems,
                      output_tensors, call_targets, implicit_then_builder);
                  scf::YieldOp::create(implicit_then_builder, updated_outputs);
                },
                [&](mlir::OpBuilder& else_b, mlir::Location else_loc) {
                  scf::YieldOp::create(else_b, else_loc, output_tensors);
                })
                .getResults();
        return predicated_updates;
      });
  return finish_entry(results);
}

}  // namespace cpu
}  // namespace xla
