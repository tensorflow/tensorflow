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

#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;

namespace ma = ::mlir::arith;
namespace scf = ::mlir::scf;

std::vector<emitters::EpilogueSpecification> CpuScatterFusion::GetEpilogues(
    const HloFusionInstruction& fusion, mlir::MLIRContext* mlir_context) const {
  const auto* scatter = fusion_->fused_expression_root();
  // We don't actually support epilogues for scatter, but this is how we tell
  // the base class that we don't want it to generate code for the scatter.
  return {emitters::EpilogueSpecification::FromIdentityIndexing(
      scatter, scatter, mlir_context)};
}

std::optional<IndexingMap> CpuScatterFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

std::optional<IndexingMap> CpuScatterFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  const auto* scatter =
      DynCast<HloScatterInstruction>(fusion_->fused_expression_root());
  CHECK(ScatterSimplifier::IsSimplifiedScatter(scatter))
      << "Non-simplified HLO Scatter is not supported.";
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

  auto root_shape = scatter->scatter_operands().front()->shape();
  SmallVector<int64_t> outer_dimension_partitions(root_shape.dimensions_size(),
                                                  1);
  auto backend_config = fusion_->backend_config<BackendConfig>();
  if (backend_config.ok() &&
      !backend_config->outer_dimension_partitions().empty()) {
    outer_dimension_partitions.assign(
        backend_config->outer_dimension_partitions().begin(),
        backend_config->outer_dimension_partitions().end());
  }
  SmallVector<int64_t> tile_sizes;
  tile_sizes.reserve(outer_dimension_partitions.size());
  for (auto [count, dim] :
       llvm::zip(root_shape.dimensions(), outer_dimension_partitions)) {
    tile_sizes.push_back(CeilDiv(count, dim));
  }
  return GetDefaultIndexingMap(tile_sizes, root_shape.dimensions(), ctx);
}

int64_t CpuScatterFusion::num_threads() const { return num_threads_; }

std::string CpuScatterFusion::BackendExtraOptions() {
  return "xla_cpu_disable_loop_unrolling";
}

SmallVector<Value> EmitScatterComputation(
    int64_t num_threads, const HloScatterInstruction* scatter,
    ValueRange indices, ValueRange update_elems, ValueRange output_tensors,
    const emitters::PartitionedComputation& root_computation,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function, mlir::ImplicitLocOpBuilder& b) {
  auto reducer =
      call_targets(scatter->called_computations()[0]->root_instruction());
  if (scatter->unique_indices() || num_threads == 1 ||
      scatter->scatter_operand_count() > 1) {
    SmallVector<Value> computation_args =
        ProvideParameterRange(root_computation, scatter, /*start=*/0,
                              /*num=*/scatter->scatter_operand_count(), indices,
                              call_targets, entry_function, b);
    computation_args.append(update_elems.begin(), update_elems.end());
    auto reduced_values =
        emitters::InlineBlock(b, reducer.getBody().front(), computation_args);
    SmallVector<Value> ret;
    ret.reserve(reduced_values.size());
    for (const auto& [reduced_value, output_tensor] :
         llvm::zip(reduced_values, output_tensors)) {
      ret.push_back(b.create<mlir::tensor::InsertOp>(reduced_value,
                                                     output_tensor, indices));
    }
    return ret;
  }
  Value output_tensor = output_tensors.front();
  Value update_elem = update_elems.front();
  auto atomic_rmw = b.create<AtomicRMWOp>(output_tensor, indices);
  mlir::OpBuilder body_builder = atomic_rmw.getBodyBuilder();
  auto reduced_val =
      emitters::InlineBlock(body_builder, reducer.getBody().front(),
                            {atomic_rmw.getCurrentValue(), update_elem})[0];
  body_builder.create<xla::YieldOp>(reducer->getLoc(), reduced_val);
  return {atomic_rmw->getResult(0)};
}

CpuScatterFusion::CpuScatterFusion(mlir::MLIRContext* mlir_context,
                                   llvm::LLVMContext* llvm_context,
                                   const BufferAssignment& buffer_assignment,
                                   const HloFusionInstruction* fusion)
    : CpuFusionEmitterBase{mlir_context, llvm_context, buffer_assignment,
                           fusion} {
  const auto* scatter = Cast<HloScatterInstruction>(
      fusion->fused_instructions_computation()->root_instruction());
  auto update_shape = scatter->scatter_updates().front()->shape();
  auto output_shape = scatter->scatter_operands().front()->shape();

  num_threads_ = 1;
  SmallVector<int64_t, 2> slice_shape(update_shape.dimensions().begin() + 1,
                                      update_shape.dimensions().end());
  int64_t num_elements = Product(slice_shape);

  const int64_t max_vectorized_bytes = 64;
  int64_t max_vectorized_elements =
      max_vectorized_bytes /
      ShapeUtil::ByteSizeOfPrimitiveType(output_shape.element_type());
  vector_size_ = std::gcd(max_vectorized_elements, num_elements);
  if (VLOG_IS_ON(5)) {
    llvm::errs() << "\nvector_size_: " << vector_size_ << "\n\n";
    llvm::errs() << "\num_threads_: " << num_threads_ << "\n\n";
  }
}
IndexingMap GetScatterIndexingMap(
    absl::Span<const int64_t> updates_operand_shape, int64_t num_threads,
    int64_t vector_size, mlir::MLIRContext* mlir_context) {
  using mlir::AffineExpr;

  // Delinearize thread_expr w.r.t. number of thread tiles per dimension.
  auto thread_expr = mlir::getAffineDimExpr(0, mlir_context);
  auto index_id = mlir::getAffineSymbolExpr(0, mlir_context);
  auto slice_linear_index = mlir::getAffineSymbolExpr(1, mlir_context);
  auto vector_element_id = mlir::getAffineSymbolExpr(2, mlir_context);

  int64_t num_updates = updates_operand_shape.front();
  int64_t num_updates_per_thread = CeilOfRatio(num_updates, num_threads);
  SmallVector<int64_t, 2> slice_shape(updates_operand_shape.begin() + 1,
                                      updates_operand_shape.end());
  int64_t num_slice_elements = Product(slice_shape);
  int64_t num_vectors_per_slice = CeilOfRatio(num_slice_elements, vector_size);

  // Loop w.r.t. indices.
  AffineExpr updates_id_expr = thread_expr * num_updates_per_thread + index_id;
  AffineExpr slice_linear_index_expr =
      slice_linear_index * vector_size + vector_element_id;
  llvm::SmallVector<AffineExpr, 4> indices_in_tile =
      DelinearizeInBoundsIndex(slice_linear_index_expr, slice_shape);
  llvm::SmallVector<AffineExpr, 4> result{updates_id_expr};
  result.append(indices_in_tile.begin(), indices_in_tile.end());

  SmallVector<std::pair<AffineExpr, Interval>, 4> constraints{
      {updates_id_expr, {0, num_updates}},
      {slice_linear_index_expr, {0, num_slice_elements - 1}}};

  auto affine_map = mlir::AffineMap::get(/*num_dims=*/1, /*num_symbols=*/3,
                                         result, mlir_context);
  return IndexingMap(
      affine_map, {IndexingMap::Variable({0, num_threads - 1, "thread_id"})},
      {IndexingMap::Variable({0, num_updates_per_thread - 1, "index_id"}),
       IndexingMap::Variable({0, num_vectors_per_slice - 1, "vector_id"}),
       IndexingMap::Variable({0, vector_size - 1, "vector_element_id"})},
      {}, constraints);
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

  mlir::MLIRContext* mlir_context = entry_function.getContext();
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());
  // %arg1 and %arg4 do alias -- they point to the same address!
  // Therefore we simply don't explicitly use %arg4, and perform all
  // input/output accesses on %arg1.
  SmallVector<Value> output_tensors;
  output_tensors.reserve(scatter_operands.size());
  for (int i = 0; i < scatter_operands.size(); ++i) {
    output_tensors.push_back(entry_function.getArgument(1 + i));
  }

  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  CHECK(ScatterSimplifier::IsSimplifiedScatter(scatter))
      << "Non-simplified HLO Scatter is not supported.";

  // For now, thread_id is hardcoded to 0.
  entry_function.setArgAttr(0, "xla.range", b.getIndexArrayAttr({0, 0}));

  const Shape& update_shape = scatter_updates.front()->shape();

  Value thread_id = entry_function.getArgument(0);
  // Set range for the func thread id arg.
  entry_function.setArgAttr(0, "xla.range",
                            b.getIndexArrayAttr({0, num_threads_ - 1}));
  IndexingMap map = GetScatterIndexingMap(
      update_shape.dimensions(), num_threads_, vector_size_, mlir_context);
  map.Simplify();

  auto results = emitters::EmitXlaLoopOp(
      b, {thread_id}, output_tensors, map,
      [&](ImplicitLocOpBuilder nested_b, ValueRange iv,
          ValueRange update_indices,
          ValueRange output_tensors) -> SmallVector<Value> {
        Value update_id = update_indices.front();

        Value c0 = nested_b.create<mlir::arith::ConstantIndexOp>(0);
        Value in_bounds = nested_b.create<ma::ConstantIntOp>(1, b.getI1Type());

        SmallVector<Value, 4> update_offsets(
            scatter_operands.front()->shape().dimensions_size(), c0);
        for (int i = 0; i < scatter_indices->shape().dimensions(1); ++i) {
          SmallVector<Value, 4> indices_tensor_indices = {
              update_id, b.create<ma::ConstantIndexOp>(i)};
          int indices_index = scatter->scatter_operand_count();
          auto index = ProvideParameter(
              root_computation, scatter, indices_index, indices_tensor_indices,
              call_targets, entry_function, nested_b)[0];
          if (primitive_util::IsUnsignedIntegralType(
                  scatter_indices->shape().element_type())) {
            index = nested_b.create<ma::IndexCastUIOp>(b.getIndexType(), index);
          } else {
            index = nested_b.create<ma::IndexCastOp>(b.getIndexType(), index);
          }
          Value ub = nested_b.create<ma::ConstantIndexOp>(
              scatter_operands.front()->shape().dimensions(i) -
              scatter_updates.front()->shape().dimensions(i + 1));
          // One bounds check is enough even for signed indices: `sge 0` is
          // implied by `ule ub`, because `ub >= 0`.
          in_bounds = nested_b.create<ma::AndIOp>(
              in_bounds,
              nested_b.create<ma::CmpIOp>(ma::CmpIPredicate::ule, index, ub));
          update_offsets[i] = index;
        }
        ValueRange predicated_updates =
            nested_b
                .create<scf::IfOp>(
                    in_bounds,
                    [&](mlir::OpBuilder& then_builder,
                        mlir::Location then_loc) -> void {
                      ImplicitLocOpBuilder implicit_then_builder(then_loc,
                                                                 then_builder);
                      // Extract update elements.
                      auto update_elems = ProvideParameterRange(
                          root_computation, scatter,
                          /*start=*/scatter->scatter_operand_count() + 1,
                          /*num=*/scatter->scatter_operand_count(),
                          update_indices, call_targets, entry_function,
                          implicit_then_builder);

                      auto output_indices = std::move(update_offsets);
                      for (int i = 0; i < output_indices.size(); ++i) {
                        output_indices[i] =
                            implicit_then_builder.create<ma::AddIOp>(
                                update_indices[i + 1], output_indices[i]);
                      }
                      SmallVector<Value> updated_outputs =
                          EmitScatterComputation(
                              num_threads(), scatter, output_indices,
                              update_elems, output_tensors, root_computation,
                              call_targets, entry_function,
                              implicit_then_builder);
                      implicit_then_builder.create<scf::YieldOp>(
                          updated_outputs);
                    },
                    [&](mlir::OpBuilder& else_b, mlir::Location else_loc) {
                      else_b.create<scf::YieldOp>(else_loc, output_tensors);
                    })
                .getResults();
        return predicated_updates;
      });
  b.create<mlir::func::ReturnOp>(results);

  if (VLOG_IS_ON(5)) {
    entry_function->getParentOfType<mlir::ModuleOp>().dump();
  }
  return absl::OkStatus();
}

}  // namespace cpu
}  // namespace xla
