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

#include "xla/backends/cpu/codegen/tiled/transforms/vectorized_reduce_emitter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"

namespace xla::cpu {

static absl::StatusOr<mlir::vector::CombiningKind> GetCombiningKind(
    mlir::Block& reduction_body) {
  mlir::Operation* op =
      reduction_body.getTerminator()->getOperand(0).getDefiningOp();
  if (!op) {
    return absl::InternalError("No reduction combiner");
  }

  for (mlir::Value operand : op->getOperands()) {
    if (operand.getDefiningOp()) {
      return absl::InternalError("Non trivial reduction combiner");
    }
  }

  if (auto kind = mlir::linalg::getCombinerOpKind(op)) {
    return *kind;
  }

  return absl::InternalError("Unsupported reduction combiner");
}

static mlir::Value ExtractVector(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value source, mlir::ValueRange indices) {
  return mlir::vector::ExtractOp::create(
      builder, loc, source, llvm::map_to_vector(indices, [](mlir::Value idx) {
        return mlir::OpFoldResult(idx);
      }));
}

static void InsertVectorIntoBuffer(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value value,
                                   mlir::TypedValue<mlir::MemRefType> buffer,
                                   mlir::ValueRange indices) {
  llvm::SmallVector<mlir::Value> padded_indices(indices);
  while (padded_indices.size() < buffer.getType().getRank()) {
    padded_indices.push_back(
        builder.create<mlir::arith::ConstantIndexOp>(loc, 0));
  }

  if (mlir::isa<mlir::VectorType>(value.getType())) {
    mlir::vector::TransferWriteOp::create(builder, loc, value, buffer,
                                          padded_indices);
  } else {
    mlir::memref::StoreOp::create(builder, loc, value, buffer, padded_indices);
  }
}

static mlir::TypedValue<mlir::VectorType> ExtractVectorFromBuffer(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::TypedValue<mlir::MemRefType> buffer, mlir::ValueRange indices = {}) {
  llvm::SmallVector<mlir::Value> padded_indices(indices);
  while (padded_indices.size() < buffer.getType().getRank()) {
    padded_indices.push_back(
        builder.create<mlir::arith::ConstantIndexOp>(loc, 0));
  }
  mlir::VectorType vector_type = mlir::VectorType::get(
      buffer.getType().getShape().drop_front(indices.size()),
      buffer.getType().getElementType());
  return mlir::vector::TransferReadOp::create(builder, loc, vector_type, buffer,
                                              padded_indices,
                                              /*padding=*/std::nullopt);
}

static std::array<llvm::SmallVector<mlir::Value>, 3> GetLoopBounds(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::ArrayRef<int64_t> upper_bounds, int64_t lower_bound = 0) {
  llvm::SmallVector<mlir::Value> lbs(
      upper_bounds.size(),
      builder.create<mlir::arith::ConstantIndexOp>(loc, lower_bound));
  llvm::SmallVector<mlir::Value> ubs =
      llvm::map_to_vector(upper_bounds, [&](int64_t size) -> mlir::Value {
        return builder.create<mlir::arith::ConstantIndexOp>(loc, size);
      });
  llvm::SmallVector<mlir::Value> step(
      upper_bounds.size(),
      builder.create<mlir::arith::ConstantIndexOp>(loc, 1));
  return {lbs, ubs, step};
}

mlir::Value VectorizeBody(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::Block& old_body, mlir::Value lhs_vector,
                          mlir::Value rhs_vector) {
  mlir::IRMapping mapping;

  mapping.map(old_body.getArgument(0), lhs_vector);
  mapping.map(old_body.getArgument(1), rhs_vector);

  for (mlir::Operation& op : old_body.without_terminator()) {
    // TODO(willfroom): Check
    // mlir::OpTrait::hasElementwiseMappableTraits
    auto new_operands = llvm::map_to_vector(
        op.getOperands(),
        [&](mlir::Value operand) { return mapping.lookup(operand); });
    mlir::Operation* new_op = op.create(
        loc, op.getName(), {lhs_vector.getType()}, new_operands, op.getAttrs(),
        op.getPropertiesStorage(), op.getSuccessors(), op.getNumRegions());
    mapping.map(&op, new_op);
    for (auto [old_res, new_res] :
         llvm::zip(op.getResults(), new_op->getResults())) {
      mapping.map(old_res, new_res);
    }
    builder.insert(new_op);
  }
  return mapping.lookup(old_body.getTerminator()->getOperand(0));
}

mlir::Value EmitNonMinorReduction(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::VectorType result_type,
    mlir::TypedValue<mlir::VectorType> source_vector,
    llvm::ArrayRef<int64_t> reduction_dims, mlir::Block& body,
    bool minor_dim_reduced) {
  mlir::VectorType source_vector_type = source_vector.getType();
  int64_t rank = source_vector_type.getRank();
  int64_t minor_dim = rank - 1;
  int64_t minor_dim_size = source_vector_type.getDimSize(minor_dim);
  llvm::SmallVector<int64_t> non_reduced_dims(rank);
  absl::c_iota(non_reduced_dims, 0);
  non_reduced_dims.erase(
      std::remove_if(non_reduced_dims.begin(), non_reduced_dims.end(),
                     [&](int64_t dim) {
                       return absl::c_find(reduction_dims, dim) !=
                              reduction_dims.end();
                     }),
      non_reduced_dims.end());

  // The set of non-reduced dimensions that are not the minor dimension.
  llvm::SmallVector<int64_t> non_reduced_non_minor_dims(non_reduced_dims);
  if (auto itr = absl::c_find(non_reduced_non_minor_dims, minor_dim);
      itr != non_reduced_non_minor_dims.end()) {
    non_reduced_non_minor_dims.erase(itr);
  }

  // The set of reduced dimensions that are not the minor dimension.
  llvm::SmallVector<int64_t> non_minor_reduced_dims(reduction_dims);
  if (auto itr = absl::c_find(non_minor_reduced_dims, minor_dim);
      itr != non_minor_reduced_dims.end()) {
    non_minor_reduced_dims.erase(itr);
  }

  // The shape of the of the non-minor-reduced output.
  llvm::SmallVector<int64_t> output_shape(result_type.getShape());
  if (minor_dim_reduced) {
    output_shape.push_back(minor_dim_size);
  }
  auto output_buffer_shape =
      mlir::MemRefType::get(output_shape, result_type.getElementType());
  auto buffer = CreateBufferOfShape(builder, loc, output_buffer_shape);

  auto [lbs, ubs, step] = GetLoopBounds(builder, loc, output_shape);
  // Outer loop is non-reduced dimensions.
  mlir::scf::buildLoopNest(
      builder, loc, lbs, ubs, step,
      [&](mlir::OpBuilder& builder, mlir::Location loc,
          mlir::ValueRange outer_induction_vars) {
        auto [lbs, ubs, step] = GetLoopBounds(
            builder, loc,
            llvm::map_to_vector(non_minor_reduced_dims,
                                [&](int64_t dim) {
                                  return source_vector_type.getDimSize(dim);
                                }),
            1);

        llvm::SmallVector<mlir::Value> zeroth_step_indices(
            rank - 1, mlir::arith::ConstantIndexOp::create(builder, loc, 0));
        for (auto [idx, var] :
             llvm::zip(non_reduced_non_minor_dims, outer_induction_vars)) {
          zeroth_step_indices[idx] = var;
        }
        // Get the first iteration
        mlir::Value minor_accumilator =
            ExtractVector(builder, loc, source_vector, zeroth_step_indices);
        // Inner loop is the non-minor reduced dimension.
        mlir::scf::LoopNest loop_nest = mlir::scf::buildLoopNest(
            builder, loc, lbs, ubs, step, minor_accumilator,
            [&](mlir::OpBuilder& builder, mlir::Location loc,
                mlir::ValueRange inner_induction_vars,
                mlir::ValueRange minor_accumilator)
                -> mlir::SmallVector<mlir::Value> {
              llvm::SmallVector<mlir::Value> indices(rank - 1);
              for (auto [idx, var] : llvm::zip(non_reduced_non_minor_dims,
                                               outer_induction_vars)) {
                indices[idx] = var;
              }
              for (auto [idx, var] :
                   llvm::zip(non_minor_reduced_dims, inner_induction_vars)) {
                indices[idx] = var;
              }

              mlir::Value vector_slice =
                  ExtractVector(builder, loc, source_vector, indices);

              return {VectorizeBody(builder, loc, body, vector_slice,
                                    minor_accumilator.front())};
            });

        InsertVectorIntoBuffer(builder, loc, loop_nest.results.front(), buffer,
                               outer_induction_vars);
        return;
      });

  // If the minor dimension is also reduced then it extracts directly from the
  // buffer to avoid the additional vector -> subvector operation.
  if (minor_dim_reduced) {
    return buffer;
  }

  return ExtractVectorFromBuffer(builder, loc, buffer);
}

mlir::TypedValue<mlir::VectorType> EmitMinorReduction(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::VectorType result_type,
    mlir::Value input, mlir::Value init_value, mlir::Block& body) {
  absl::StatusOr<mlir::vector::CombiningKind> kind_or = GetCombiningKind(body);
  if (!kind_or.ok()) {
    body.getParentOp()->emitRemark() << kind_or.status().ToString();
  }

  // TODO(willfroom): we could reuse the non minor result buffer.
  auto minor_result_buffer = CreateBufferOfShape(builder, loc, result_type);
  auto maybe_input_buffer =
      mlir::dyn_cast<mlir::TypedValue<mlir::MemRefType>>(input);

  auto maybe_input_type =
      llvm::TypeSwitch<mlir::Type, std::optional<mlir::ShapedType>>(
          input.getType())
          .Case<mlir::MemRefType>([&](auto op) { return input.getType(); })
          .Case<mlir::VectorType>([&](auto op) { return input.getType(); })
          .Default([&](auto op) { return std::nullopt; });

  if (!maybe_input_type.has_value()) {
    return nullptr;
  }

  int64_t minor_dim_size = maybe_input_type->getShape().back();

  auto [lbs, ubs, step] = GetLoopBounds(builder, loc, result_type.getShape());

  mlir::scf::buildLoopNest(
      builder, loc, lbs, ubs, step,
      [&](mlir::OpBuilder& builder, mlir::Location loc,
          mlir::ValueRange induction_vars) {
        mlir::Value vector_slice =
            maybe_input_buffer
                ? ExtractVectorFromBuffer(builder, loc, maybe_input_buffer,
                                          induction_vars)
                : ExtractVector(builder, loc, input, induction_vars);

        if (kind_or.ok()) {
          mlir::Value reduced_scalar =
              builder.create<mlir::vector::ReductionOp>(
                  loc, *kind_or, vector_slice, init_value);
          InsertVectorIntoBuffer(builder, loc, reduced_scalar,
                                 minor_result_buffer, induction_vars);
          return;
        }

        auto [lbs, ubs, step] = GetLoopBounds(builder, loc, {minor_dim_size});
        mlir::scf::LoopNest minor_reduction_loop = mlir::scf::buildLoopNest(
            builder, loc, lbs, ubs, step, {init_value},
            [&](mlir::OpBuilder& builder, mlir::Location loc,
                mlir::ValueRange index, mlir::ValueRange carry_value)
                -> mlir::SmallVector<mlir::Value> {
              mlir::Value element =
                  ExtractVector(builder, loc, vector_slice, index);
              return {VectorizeBody(builder, loc, body, element,
                                    carry_value.front())};
            });

        InsertVectorIntoBuffer(builder, loc,
                               minor_reduction_loop.results.front(),
                               minor_result_buffer, induction_vars);
        return;
      });

  return ExtractVectorFromBuffer(builder, loc, minor_result_buffer);
}

mlir::Value EmitVectorizedReduction(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::VectorType result_type,
    mlir::TypedValue<mlir::VectorType> source, mlir::Value init_value,
    llvm::ArrayRef<int64_t> reduction_dims, mlir::Block& body) {
  int64_t rank = source.getType().getRank();
  int64_t minor_dim = rank - 1;

  bool minor_dim_reduced = reduction_dims.back() == minor_dim;
  bool non_minor_dim_reduced = reduction_dims.size() > 1 || !minor_dim_reduced;

  mlir::Value non_minor_result;
  if (non_minor_dim_reduced) {
    non_minor_result =
        EmitNonMinorReduction(builder, loc, result_type, source, reduction_dims,
                              body, minor_dim_reduced);
  }
  if (!minor_dim_reduced) {
    // We add the init value during the minor reduction loop, if that wasn't
    // done then we must apply it here.
    mlir::Value init_value_vector =
        builder.create<mlir::vector::BroadcastOp>(loc, result_type, init_value);

    return VectorizeBody(builder, loc, body, non_minor_result,
                         init_value_vector);
  }

  return EmitMinorReduction(builder, loc, result_type,
                            non_minor_result ? non_minor_result : source,
                            init_value, body);
}

}  // namespace xla::cpu
