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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
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

static void InsertValue(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value value,
                        mlir::TypedValue<mlir::MemRefType> buffer,
                        mlir::ValueRange indices) {
  llvm::SmallVector<mlir::Value> padded_indices(indices);
  while (padded_indices.size() < buffer.getType().getRank()) {
    padded_indices.push_back(
        mlir::arith::ConstantIndexOp::create(builder, loc, 0));
  }

  if (mlir::isa<mlir::VectorType>(value.getType())) {
    mlir::vector::TransferWriteOp::create(builder, loc, value, buffer,
                                          padded_indices);
  } else {
    mlir::memref::StoreOp::create(builder, loc, value, buffer, padded_indices);
  }
}

static mlir::TypedValue<mlir::VectorType> ExtractVector(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::TypedValue<mlir::ShapedType> input, mlir::ValueRange indices = {}) {
  llvm::SmallVector<mlir::Value> padded_indices(indices);
  while (padded_indices.size() < input.getType().getRank()) {
    padded_indices.push_back(
        mlir::arith::ConstantIndexOp::create(builder, loc, 0));
  }
  mlir::VectorType vector_type = mlir::VectorType::get(
      input.getType().getShape().drop_front(indices.size()),
      input.getType().getElementType());
  return mlir::vector::TransferReadOp::create(builder, loc, vector_type, input,
                                              padded_indices,
                                              /*padding=*/std::nullopt);
}

static std::array<llvm::SmallVector<mlir::Value>, 3> GetLoopBounds(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::ArrayRef<int64_t> upper_bounds, int64_t lower_bound = 0) {
  llvm::SmallVector<mlir::Value> lbs(
      upper_bounds.size(),
      mlir::arith::ConstantIndexOp::create(builder, loc, lower_bound));
  llvm::SmallVector<mlir::Value> ubs =
      llvm::map_to_vector(upper_bounds, [&](int64_t size) -> mlir::Value {
        return mlir::arith::ConstantIndexOp::create(builder, loc, size);
      });
  llvm::SmallVector<mlir::Value> step(
      upper_bounds.size(),
      mlir::arith::ConstantIndexOp::create(builder, loc, 1));
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

// Reduce a 1D vector to a scalar with the given body.
mlir::Value EmitMinorReduction(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::RankedTensorType result_type,
                               mlir::TypedValue<mlir::VectorType> input,
                               mlir::Value init_value, mlir::Block& body) {
  absl::StatusOr<mlir::vector::CombiningKind> kind_or = GetCombiningKind(body);
  if (!kind_or.ok()) {
    body.getParentOp()->emitRemark() << kind_or.status().ToString();
  }

  auto input_type = input.getType();
  int64_t minor_dim_size = input_type.getShape().back();

  if (kind_or.ok()) {
    // TODO(willfroom): Investigate tree-reduction to split the reduction
    // op into natural sizes (2, 4, 8, 16, ...) and then remove the
    // reassociation flag.
    mlir::Value reduced_scalar = mlir::vector::ReductionOp::create(
        builder, loc, *kind_or, input, init_value,
        mlir::arith::FastMathFlags::reassoc);

    return reduced_scalar;
  }

  mlir::Value lbs = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value ubs =
      mlir::arith::ConstantIndexOp::create(builder, loc, minor_dim_size);
  mlir::Value step = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  auto loop = mlir::scf::ForOp::create(
      builder, loc, lbs, ubs, step, {init_value},
      [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value index,
          mlir::ValueRange carry_value) {
        mlir::TypedValue<mlir::VectorType> element_vector =
            ExtractVector(builder, loc, input, index);
        mlir::Value element =
            mlir::vector::ExtractOp::create(builder, loc, element_vector);

        mlir::Value result =
            VectorizeBody(builder, loc, body, element, carry_value.front());

        mlir::scf::YieldOp::create(builder, loc, result);
      });

  return loop.getResult(0);
}

mlir::TypedValue<mlir::MemRefType> EmitReductionLoop(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType result_type,
    mlir::TypedValue<mlir::RankedTensorType> source_tensor,
    llvm::ArrayRef<int64_t> reduction_dims, mlir::Block& body,
    mlir::Value init_value) {
  mlir::RankedTensorType source_tensor_type = source_tensor.getType();
  int64_t rank = source_tensor_type.getRank();
  int64_t minor_dim = rank - 1;
  bool minor_dim_reduced = reduction_dims.back() == minor_dim;

  // The set of non-reduced dimensions that are not the minor dimension.
  llvm::SmallVector<int64_t> non_reduced_non_minor_dims(rank);
  absl::c_iota(non_reduced_non_minor_dims, 0);
  non_reduced_non_minor_dims.erase(
      std::remove_if(
          non_reduced_non_minor_dims.begin(), non_reduced_non_minor_dims.end(),
          [&](int64_t dim) {
            return absl::c_find(reduction_dims, dim) != reduction_dims.end() ||
                   dim == minor_dim;
          }),
      non_reduced_non_minor_dims.end());

  // The set of reduced dimensions that are not the minor dimension.
  llvm::SmallVector<int64_t> non_minor_reduced_dims(reduction_dims);
  if (auto itr = absl::c_find(non_minor_reduced_dims, minor_dim);
      itr != non_minor_reduced_dims.end()) {
    non_minor_reduced_dims.erase(itr);
  }

  auto buffer = CreateBufferOfShape(builder, loc, result_type);

  auto get_source_vector_dim_size = [&](llvm::ArrayRef<int64_t> dims) {
    return llvm::map_to_vector(
        dims, [&](int64_t dim) { return source_tensor_type.getDimSize(dim); });
  };

  // Outer loop is non-minor non-reduced dimensions.
  auto [lbs, ubs, step] = GetLoopBounds(
      builder, loc, get_source_vector_dim_size(non_reduced_non_minor_dims));

  mlir::scf::buildLoopNest(
      builder, loc, lbs, ubs, step,
      [&](mlir::OpBuilder& builder, mlir::Location loc,
          mlir::ValueRange outer_induction_vars) {
        auto [lbs, ubs, step] = GetLoopBounds(
            builder, loc, get_source_vector_dim_size(non_minor_reduced_dims),
            1);

        llvm::SmallVector<mlir::Value> zeroth_step_indices(
            rank - 1, mlir::arith::ConstantIndexOp::create(builder, loc, 0));
        for (auto [idx, var] :
             llvm::zip(non_reduced_non_minor_dims, outer_induction_vars)) {
          zeroth_step_indices[idx] = var;
        }
        // Get the first iteration
        mlir::Value minor_accumilator =
            ExtractVector(builder, loc, source_tensor, zeroth_step_indices);
        // Inner loop is the non-minor reduced dimension.
        mlir::scf::LoopNest loop_nest = mlir::scf::buildLoopNest(
            builder, loc, lbs, ubs, step, minor_accumilator,
            [&](mlir::OpBuilder& builder, mlir::Location loc,
                mlir::ValueRange inner_induction_vars,
                mlir::ValueRange minor_accumilator)
                -> mlir::SmallVector<mlir::Value> {
              // Handle the case when there are no non-minor reduced dimensions.
              if (inner_induction_vars.empty()) {
                return {minor_accumilator.front()};
              }

              llvm::SmallVector<mlir::Value> indices = zeroth_step_indices;
              for (auto [idx, var] :
                   llvm::zip(non_minor_reduced_dims, inner_induction_vars)) {
                indices[idx] = var;
              }

              mlir::Value vector_slice =
                  ExtractVector(builder, loc, source_tensor, indices);

              return {VectorizeBody(builder, loc, body, vector_slice,
                                    minor_accumilator.front())};
            });

        auto non_minor_reduced_result =
            mlir::cast<mlir::TypedValue<mlir::VectorType>>(
                loop_nest.results.front());

        if (minor_dim_reduced) {
          mlir::Value reduced_scalar =
              EmitMinorReduction(builder, loc, result_type,
                                 non_minor_reduced_result, init_value, body);

          InsertValue(builder, loc, reduced_scalar, buffer,
                      outer_induction_vars);
        } else {
          InsertValue(builder, loc, non_minor_reduced_result, buffer,
                      outer_induction_vars);
        }
      });

  return buffer;
}

mlir::Value EmitVectorizedReduction(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType result_type,
    mlir::TypedValue<mlir::RankedTensorType> source, mlir::Value init_value,
    llvm::ArrayRef<int64_t> reduction_dims, mlir::Block& body) {
  mlir::TypedValue<mlir::ShapedType> result;
  result = EmitReductionLoop(builder, loc, result_type, source, reduction_dims,
                             body, init_value);

  auto to_tensor = mlir::bufferization::ToTensorOp::create(builder, loc,
                                                           result_type, result);
  // This is a local allocation so we know it doesn't alias.
  to_tensor.setRestrict(true);
  to_tensor.setWritable(true);
  return to_tensor;
}

}  // namespace xla::cpu
