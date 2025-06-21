/* Copyright 2024 The JAX Authors.

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

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"
#include "xla/mosaic/dialect/tpu/vreg_util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_CANONICALIZEMOSAICPASS
#define GEN_PASS_DEF_CANONICALIZEMOSAICPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

struct CanonicalizeContext {
  // see Note: Compatibility mode
  bool compatibility_mode;

  int hardware_generation;

  std::array<int64_t, 2> target_shape;
};

Value create_transpose_op(const CanonicalizeContext &ctx,
                          ImplicitLocOpBuilder &builder, VectorType input_ty,
                          Value input, ArrayRef<int64_t> permutation);

bool need_elementwise_canonicalization(const CanonicalizeContext &ctx,
                                       Operation &op);

LogicalResult tpu_matmul_rule(const CanonicalizeContext &ctx,
                              tpu::MatmulOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

  auto transpose_lhs = op.getTransposeLhs();
  auto transpose_rhs = op.getTransposeRhs();

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto acc = op.getAcc();

  const VectorType lhs_ty = lhs.getType();
  const VectorType rhs_ty = rhs.getType();
  const VectorType acc_ty = acc.getType();

  auto lhs_element_type = lhs_ty.getElementType();
  auto rhs_element_type = rhs_ty.getElementType();
  auto acc_element_type = acc_ty.getElementType();

  // there are a few primary paths for dimension_numbers in matmul
  // 1) No dimension numbers provided -> set to default
  // 2) defined and not default -> verify and apply
  // 3) defined and matching defaultDimensionNumbers -> no-op for
  // canonicalization of dims
  std::optional<int64_t> batch_size = std::nullopt;

  // MKN matmul - no dims or transpositions set
  if (!op.getDimensionNumbers().has_value()) {
    // Legacy API - convert it to dimension numbers
    op.setDimensionNumbersAttr(
        defaultDimensionNumbers(builder, transpose_lhs, transpose_rhs));
  } else if (
      // Dot dim API - dimensions are provided and are not default
      (op.getDimensionNumbers().value() !=
       defaultDimensionNumbers(builder, false, false))) {
    auto dimension_numbers = op.getDimensionNumbers();
    auto lhs_contracting_dims = dimension_numbers->getLhsContractingDims();
    auto rhs_contracting_dims = dimension_numbers->getRhsContractingDims();

    auto lhs_batch_dims = dimension_numbers->getLhsBatchDims();
    auto rhs_batch_dims = dimension_numbers->getRhsBatchDims();

    // Invariant in matmul verifier: <= 1 batch dim atm, and that lhs and rhs
    // are the same
    // Invariant in matmul verifier: Exactly one contracting and non contracting
    // dim in each of lhs and rhs for now.
    batch_size =
        lhs_batch_dims.empty()
            ? std::nullopt
            : std::optional<int64_t>(lhs_ty.getShape()[lhs_batch_dims[0]]);
    // Lower each dim in contracting dims by size(batch_dims)
    auto batch_adjusted_lhs_contracting_dim =
        lhs_contracting_dims[0] - lhs_batch_dims.size();
    auto batch_adjusted_rhs_contracting_dim =
        rhs_contracting_dims[0] - rhs_batch_dims.size();

    if (batch_adjusted_lhs_contracting_dim != 1) {
      transpose_lhs = true;
    }
    if (batch_adjusted_rhs_contracting_dim != 0) {
      transpose_rhs = true;
    }
  }

  auto extsi_sitofp = [&builder, &op](TypedValue<VectorType> element) {
    const VectorType ty = element.getType();
    auto shape = ty.getShape();
    CHECK(ty.getElementType().isInteger());
    TypedValue<VectorType> ext_ele;
    if (ty.getElementType().getIntOrFloatBitWidth() == 32) {
      ext_ele = element;
    } else {
      ext_ele = cast<TypedValue<VectorType>>(
          builder
              .create<arith::ExtSIOp>(
                  VectorType::get(shape, builder.getI32Type()), element)
              .getResult());
    }
    // TODO(mvoz): Go to bf16 when hardware supported, requires adding support
    // for 16 bitwidth in extsiop in infer/apply.
    auto ele_as_fp = builder.create<arith::SIToFPOp>(
        op.getLoc(), VectorType::get(shape, builder.getF32Type()), ext_ele);
    return ele_as_fp;
  };

  if (lhs_element_type != rhs_element_type) {
    if (!ctx.compatibility_mode) {
      return op->emitOpError(
          "Mosaic matmul invoked with mixed element types, but compatibility "
          "mode is disabled.");
    }
    if (lhs_element_type.isInteger() && rhs_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/int matmul.
      op->emitOpError("Mix int/int - NYI");
      return failure();
    }
    if (acc_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/float matmul with int acc.
      // Should be pretty straightforward.
      op->emitOpError("acc is int in mixed matmul. Expected float.");
      return failure();
    }
    if (lhs_element_type.isInteger()) {
      auto float_lhs = extsi_sitofp(lhs);
      op->setOperand(0, float_lhs);
      lhs = cast<TypedValue<VectorType>>(float_lhs.getResult());
    }
    if (rhs_element_type.isInteger()) {
      auto float_rhs = extsi_sitofp(rhs);
      op->setOperand(1, float_rhs);
      rhs = cast<TypedValue<VectorType>>(float_rhs.getResult());
    }
  }
  // TODO(mvoz): Add more invariants.
  if (acc_element_type.isInteger()) {
    if (!op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float lhs. Expected int lhs.");
      return failure();
    }
    if (!op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float rhs. Expected int rhs.");
      return failure();
    }
  } else {
    if (op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int lhs. Expected float lhs.");
      return failure();
    }
    if (op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int rhs. Expected float rhs.");
      return failure();
    }
  }

  // Attempt to canonicalize matmul(x, transpose(y)) to a matmul with the
  // dimension numbers changed which will later be lowered into a more efficient
  // operation that fuses the transpose into the matmul.
  auto transpose_op =
      dyn_cast_if_present<tpu::TransposeOp>(rhs.getDefiningOp());
  auto dimension_numbers = op.getDimensionNumbers();
  if (transpose_op && transpose_op->hasOneUse() &&
      dimension_numbers->getRhsContractingDims().size() == 1 &&
      dimension_numbers->getRhsNonContractingDims().size() == 1) {
    auto rhs_non_contracting_dim =
        dimension_numbers->getRhsNonContractingDims()[0];
    auto rhs_contracting_dim = dimension_numbers->getRhsContractingDims()[0];
    auto permutation = transpose_op.getPermutation();
    if (permutation[rhs_contracting_dim] == rhs_non_contracting_dim &&
        permutation[rhs_non_contracting_dim] == rhs_contracting_dim &&
        std::all_of(dimension_numbers->getRhsBatchDims().begin(),
                    dimension_numbers->getRhsBatchDims().end(),
                    [&](long batch_dim) {
                      return permutation[batch_dim] == batch_dim;
                    })) {
      if (auto transpose_op_vector_operand =
              dyn_cast<TypedValue<VectorType>>(transpose_op.getOperand())) {
        // The transpose is DCE'ed away at a later point.
        rhs = transpose_op_vector_operand;
        transpose_rhs = !transpose_rhs;
      } else {
        return op->emitOpError("Unexpected operand type for transpose op.");
      }
    }
  }

  auto dot_dim_matmul = [&](Value lhs, auto rhs, auto acc) {
    auto precision_attr = op.getPrecisionAttr();

    // If we are transposing the lhs, we need to transpose the lhs before
    // matmul here, as we don't have lhs fusion implemented in apply.
    if (transpose_lhs) {
      auto lhs_ty = cast<VectorType>(lhs.getType());
      auto rank = lhs_ty.getShape().size();

      // This transposition must run on vectors with rank >= 2
      CHECK_GE(rank, 2);

      std::vector<int64_t> perm(rank);
      std::iota(perm.begin(), perm.end(), 0);
      std::swap(perm[rank - 2], perm[rank - 1]);

      std::vector<int64_t> shape(lhs_ty.getShape());
      std::swap(shape[rank - 2], shape[rank - 1]);

      VectorType lhs_ty_transposed =
          VectorType::get(shape, lhs_ty.getElementType());

      const SmallVector<int64_t> perm_vec =
          SmallVector<int64_t>(perm.begin(), perm.end());
      lhs = create_transpose_op(ctx, builder, lhs_ty_transposed, lhs, perm_vec);
    }
    auto ddn = defaultDimensionNumbers(builder, /*transpose_lhs=*/false,
                                       transpose_rhs);
    // transpose flags are always false here, because ddn takes precedence
    // after this pass.
    auto matmul_res = builder.create<tpu::MatmulOp>(
        op.getLoc(), acc.getType(), lhs, rhs, acc,
        /*transpose_lhs=*/false,
        /*transpose_rhs=*/false, precision_attr, ddn);
    return matmul_res;
  };

  // If we have a batch_size, we want to slice rhs and lhs [:batch_size],
  // and then do O[i] = A[i] @ B[i]
  // Produce an output shape of [batch_size, m, n]
  if (batch_size.has_value()) {
    std::vector<Value> outputs;

    for (int64_t i = 0; i < batch_size; ++i) {
      auto sliced_lhs = builder.create<vector::ExtractOp>(op.getLoc(), lhs,
                                                          ArrayRef<int64_t>{i});
      auto sliced_rhs = builder.create<vector::ExtractOp>(op.getLoc(), rhs,
                                                          ArrayRef<int64_t>{i});

      auto sliced_acc = builder.create<vector::ExtractOp>(op.getLoc(), acc,
                                                          ArrayRef<int64_t>{i});

      auto matmul_res =
          dot_dim_matmul(sliced_lhs.getResult(), sliced_rhs.getResult(),
                         sliced_acc.getResult());
      auto res_ty = cast<VectorType>(matmul_res.getType());
      auto res_shape = res_ty.getShape();
      // reshape to 1x[prior_shape]
      auto reshape_shape = llvm::to_vector(res_shape);
      reshape_shape.insert(reshape_shape.begin(), 1);
      auto shape_cast = builder.create<vector::ShapeCastOp>(
          op.getLoc(), VectorType::get(reshape_shape, res_ty.getElementType()),
          matmul_res);
      outputs.push_back(shape_cast);
    }
    // Technically almost identical to the case where batch_size is 1, but
    // we want to avoid the spurious concat here.
    if (batch_size == 1) {
      op.replaceAllUsesWith(outputs[0]);
      op.erase();
      return success();
    }
    auto output = builder
                      .create<tpu::ConcatenateOp>(op.getLoc(), acc_ty, outputs,
                                                  /*dimension=*/0)
                      .getResult();
    op.replaceAllUsesWith(output);
    op.erase();
  } else {
    auto matmul_res = dot_dim_matmul(lhs, rhs, acc).getResult();
    op.replaceAllUsesWith(matmul_res);
    op.erase();
  }
  return success();
};

LogicalResult canonicalize_elementwise(const CanonicalizeContext &ctx,
                                       Operation &op) {
  OpBuilder builder(&op);
  auto operands = op.getOperands();
  auto res_ty = dyn_cast<VectorType>(op.getResult(0).getType());
  if (op.getNumResults() != 1) {
    op.emitOpError("Invariant violated: Unexpected number of results");
    return failure();
  }
  if (!res_ty) {
    // scalar
    // TODO(mvoz): Add canonicalization and invariants for scalar elementwise
    // ops.
    return success();
  }
  auto shape = res_ty.getShape();
  std::vector<Value> new_operands;
  new_operands.reserve(operands.size());

  bool should_rewrite_op = false;
  auto target_f32_ty = VectorType::get(shape, builder.getF32Type());
  for (int i = 0; i < operands.size(); ++i) {
    auto operand = operands[i];
    auto ty = dyn_cast<VectorType>(operand.getType());
    if (ty) {
      if (ty.getShape() != shape) {
        // Should already be checked my MLIR verification, but let's be safe.
        op.emitOpError("Mismatched shapes in elementwise op.");
        return failure();
      }
      auto element_type = ty.getElementType();
      // There's an annoying hodgepodge of elementwise ops that need to be
      // rewritten to f32 on later hardware.
      if (element_type.isBF16()) {
        if (ctx.compatibility_mode) {
          auto target_f32 =
              builder.create<arith::ExtFOp>(op.getLoc(), target_f32_ty, operand)
                  .getResult();
          should_rewrite_op = true;
          new_operands.push_back(target_f32);
        } else {
          op.emitOpError(
              "Compatibility mode disabled. Unsupported element type in "
              "elementwise op on hardware generation: ")
              << ctx.hardware_generation
              << ". Use hardware generation after 5 or cast to f32.";
          return failure();
        }
      } else {
        new_operands.push_back(operand);
      }
    } else {
      // Should already be checked my MLIR verification, but let's be safe.
      op.emitOpError("MLIR unsupported - mix scalar and vec elementwise ops");
      return failure();
    }
  }
  if (should_rewrite_op) {
    if (!res_ty) {
      op.emitOpError("Not implemented: Unexpected result type");
      return failure();
    }
    // Do the new op in f32, then truncate to the original element type if
    // needed. For example, result of arith::CmpF is i1 and doesn't need to be
    // truncated.
    bool should_truncate = !isa<arith::CmpFOp>(op);
    auto new_res_ty =
        VectorType::get(shape, should_truncate ? builder.getF32Type()
                                               : res_ty.getElementType());
    auto new_op = builder.create(op.getLoc(), op.getName().getIdentifier(),
                                 new_operands, new_res_ty, op.getAttrs());
    if (should_truncate) {
      new_op = builder.create<arith::TruncFOp>(op.getLoc(), res_ty,
                                               new_op->getResult(0));
    }
    op.replaceAllUsesWith(new_op);
    op.erase();
  }
  return success();
}

LogicalResult canonicalize_multi_dim_reduction(const CanonicalizeContext &ctx,
                                               Operation &operation) {
  ImplicitLocOpBuilder builder(operation.getLoc(), &operation);
  auto op = cast<vector::MultiDimReductionOp>(operation);
  auto source_ty = op.getSourceVectorType();
  auto result_ty = dyn_cast<VectorType>(op.getDestType());
  if (!result_ty) {
    return op->emitOpError() << "Only vector reductions supported";
  }

  auto element_type = source_ty.getElementType();
  if (element_type.isF32()) {
    return success();
  } else if (element_type.isBF16()) {
    bool reduces_sublanes = false;
    for (int64_t dim : op.getReductionDims()) {
      if (dim == source_ty.getRank() - 2) {
        reduces_sublanes = true;
      }
    }
    if (ctx.hardware_generation <= 5) {
      auto new_source = builder.create<arith::ExtFOp>(
          VectorType::get(source_ty.getShape(), builder.getF32Type()),
          op.getSource());

      auto result_ty_f32 =
          VectorType::get(result_ty.getShape(), builder.getF32Type());
      auto acc_ext = builder.create<arith::ExtFOp>(result_ty_f32, op.getAcc());
      Value new_acc = acc_ext.getResult();
      // Try to constant fold.
      if (auto const_acc = op.getAcc().getDefiningOp<arith::ConstantOp>()) {
        auto result =
            acc_ext.fold(arith::ExtFOp::FoldAdaptor(const_acc.getValue()));
        if (!result.isNull() && result.is<Attribute>()) {
          acc_ext->erase();
          new_acc = builder.create<arith::ConstantOp>(
              op.getLoc(), result_ty_f32,
              cast<TypedAttr>(result.get<Attribute>()));
        }
      }
      auto new_op = builder.create<vector::MultiDimReductionOp>(
          op.getLoc(), new_acc.getType(), op.getKindAttr(), new_source, new_acc,
          DenseI64ArrayAttr::get(builder.getContext(), op.getReductionDims()));
      auto new_result = builder.create<arith::TruncFOp>(op.getLoc(), result_ty,
                                                        new_op.getResult());
      op.replaceAllUsesWith(new_result.getResult());
      op.erase();
    }
    return success();
  } else if (element_type.isSignlessInteger(32) &&
             // TODO(b/384774084): Add support for u32 reductions.
             (op.getKind() == vector::CombiningKind::ADD ||
              op.getKind() == vector::CombiningKind::MAXSI ||
              op.getKind() == vector::CombiningKind::MINSI)) {
    return success();
  }
  op.emitOpError("Unsupported element type for the selected reduction");
  return failure();
}

LogicalResult canonicalize_matmul(const CanonicalizeContext &ctx,
                                  Operation &op) {
  auto matmul_op = dyn_cast<tpu::MatmulOp>(op);
  if (!matmul_op) {
    op.emitOpError("Invariant violated: Not a matmul");
    return failure();
  }
  return tpu_matmul_rule(ctx, matmul_op);
};

LogicalResult canonicalize_contraction(const CanonicalizeContext &ctx,
                                       Operation &op) {
  auto contraction_op = dyn_cast<vector::ContractionOp>(op);
  if (!contraction_op) {
    op.emitOpError("Invariant violated: Not a contraction");
    return failure();
  }
  // Rewrite the contraction as a matmul
  auto lhs = contraction_op.getLhs();
  auto rhs = contraction_op.getRhs();
  auto acc = contraction_op.getAcc();
  VectorType acc_ty;
  if (!(acc_ty = dyn_cast<VectorType>(acc.getType()))) {
    contraction_op->emitOpError("Not implemented: acc must be a vector");
    return failure();
  }

  if (contraction_op.getKind() != vector::CombiningKind::ADD) {
    contraction_op->emitOpError("Only ADD supported");
    return failure();
  }

  ImplicitLocOpBuilder builder(contraction_op->getLoc(),
                               contraction_op.getOperation());

  MLIRContext *const mlir_ctx = contraction_op->getContext();

  auto getMapAttr = [&](const unsigned first, const unsigned second) {
    return AffineMapAttr::get(AffineMap::get(
        3, 0,
        {getAffineDimExpr(first, mlir_ctx), getAffineDimExpr(second, mlir_ctx)},
        mlir_ctx));
  };

  const ArrayAttr matmul_indexing_maps = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(2, 1), getMapAttr(0, 1)});
  const ArrayAttr matmul_indexing_maps_transposed = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(1, 2), getMapAttr(0, 1)});
  const auto indexing_maps = contraction_op.getIndexingMaps();
  if (indexing_maps != matmul_indexing_maps &&
      indexing_maps != matmul_indexing_maps_transposed) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul or unsupported indexing_maps");
  }
  const bool transpose_rhs = indexing_maps == matmul_indexing_maps_transposed;

  const ArrayAttr matmul_iterator_types =
      builder.getArrayAttr({builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::reduction)});
  if (contraction_op->getAttr("iterator_types") != matmul_iterator_types) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
  }
  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      contraction_op->getAttrOfType<tpu::ContractPrecisionAttr>("precision");

  const auto dot_dimension_numbers_attr =
      defaultDimensionNumbers(builder, false, transpose_rhs);

  auto matmul_op = builder.create<tpu::MatmulOp>(
      contraction_op->getLoc(), acc_ty, lhs, rhs, acc,
      /*transpose_lhs=*/false,
      /*transpose_rhs=*/false, precision_attr, dot_dimension_numbers_attr);
  contraction_op.replaceAllUsesWith(matmul_op.getResult());
  contraction_op.erase();
  auto result = tpu_matmul_rule(ctx, matmul_op);
  return result;
}

LogicalResult canonicalize_extract(const CanonicalizeContext &ctx,
                                   Operation &raw_op) {
  auto op = dyn_cast<vector::ExtractOp>(raw_op);
  Type result_ty = op.getResult().getType();
  if (!isa<VectorType>(result_ty)) {
    bool is_supported = result_ty.isSignlessIntOrFloat() &&
                        result_ty.getIntOrFloatBitWidth() == 32;
    if (!is_supported) {
      return op.emitOpError(
          "Only 32-bit scalar vector.extracts supported. Cast your input to a "
          "32-bit type first.");
    }
  }
  return success();
}

LogicalResult canonicalize_broadcast(const CanonicalizeContext &ctx,
                                     Operation &raw_op) {
  auto op = dyn_cast<vector::BroadcastOp>(raw_op);
  auto src_ty = op.getSource().getType();
  auto src_vty = dyn_cast<VectorType>(src_ty);
  if ((src_vty && src_vty.getElementType().isSignlessInteger(1)) ||
      op.getSource().getType().isSignlessInteger(1)) {
    // Canonicalize i1 broadcast.
    // i1 represents vmsk in Mosaic and TPU doesn't support vmsk replication
    // directly.
    // Instead, convert i1 to i32 vector, broadcast i32, and then convert it
    // back to i1.
    ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
    Value i32_src;
    if (src_vty) {
      i32_src = builder.create<arith::ExtUIOp>(
          VectorType::get(src_vty.getShape(), builder.getI32Type()),
          op.getSource());
    } else {
      i32_src =
          builder.create<arith::ExtUIOp>(builder.getI32Type(), op.getSource());
    }
    auto i32_res_vty =
        VectorType::get(op.getType().getShape(), builder.getI32Type());
    auto bcast = builder.create<vector::BroadcastOp>(i32_res_vty, i32_src);
    auto ones = builder.create<arith::ConstantOp>(
        i32_res_vty,
        SplatElementsAttr::get(i32_res_vty,
                               builder.getOneAttr(builder.getI32Type())));
    auto cmp =
        builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bcast, ones);
    op.replaceAllUsesWith(cmp.getResult());
    op.erase();
    return success();
  }
  return success();
}

LogicalResult canonicalize_select(const CanonicalizeContext &ctx,
                                  Operation &raw_op) {
  auto op = dyn_cast<arith::SelectOp>(raw_op);
  if (!isa<VectorType>(op.getType()) ||
      isa<VectorType>(op.getCondition().getType())) {
    return success();
  }
  // Canonicalize `i1 ? v1 : v2` -> `broadcast(i1) ? v1 : v2`.
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto cond_ty = VectorType::get(cast<VectorType>(op.getType()).getShape(),
                                 op.getCondition().getType());
  auto cond = builder.create<vector::BroadcastOp>(cond_ty, op.getCondition());
  auto new_op = builder.create<arith::SelectOp>(
      op.getLoc(), cond, op.getTrueValue(), op.getFalseValue());
  op.replaceAllUsesWith(new_op.getResult());
  op.erase();
  if (need_elementwise_canonicalization(ctx, *new_op.getOperation())) {
    return canonicalize_elementwise(ctx, *new_op.getOperation());
  }
  return success();
}

// All conversions that change bitwidth must be canonicalized to tpu.fptosi.
LogicalResult canonicalize_fptosi(const CanonicalizeContext &ctx,
                                  Operation &raw_op) {
  auto op = cast<arith::FPToSIOp>(raw_op);
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto src_vty = dyn_cast<VectorType>(op.getIn().getType());
  auto dst_vty = dyn_cast<VectorType>(op.getType());
  if (static_cast<bool>(src_vty) != static_cast<bool>(dst_vty)) {
    return op.emitOpError("Vector/scalar mismatch between input and output");
  }
  bool is_vector = static_cast<bool>(src_vty);
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned src_bitwidth,
                             getElementTypeBitwidth(op.getIn().getType()));
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned dst_bitwidth,
                             getElementTypeBitwidth(op.getType()));
  if (dst_bitwidth > 32) {
    return op.emitOpError("Target bitwidth too large");
  }
  // We have low-level optimized code for bf16->s8 and bf16->s4 casts on v6.
  if (ctx.hardware_generation >= 6 && is_vector &&
      src_vty.getElementType().isBF16() &&
      (dst_vty.getElementType().isSignlessInteger(8) ||
       dst_vty.getElementType().isSignlessInteger(4))) {
    auto new_op = builder.create<tpu::FPToSIOp>(
        op.getType(), op.getIn(), tpu::RoundingMode::kTowardsZero);
    op.replaceAllUsesWith(new_op.getResult());
    op.erase();
    return success();
  }

  if ((src_bitwidth < 32 || dst_bitwidth < 32) && !ctx.compatibility_mode) {
    return op.emitOpError(
        "On this target float-to-integer conversions can only happen on "
        "32-bit values. Enable compatibility mode or upcast to float32, cast "
        "to int32 and truncate to desired bitwidth.");
  }

  Value x = op.getIn();
  // Upcast the input to f32.
  if (src_bitwidth < 32) {
    if (is_vector) {
      x = builder.create<arith::ExtFOp>(
          VectorType::get(src_vty.getShape(), builder.getF32Type()), x);
    } else {
      x = builder.create<arith::ExtFOp>(builder.getF32Type(), x);
    }
  }
  if (dst_bitwidth < 32) {
    // Need to clip values to match XLA
    auto clip = [&](Value x, Value low, Value high) {
      x = builder.create<arith::MaximumFOp>(x, low);
      x = builder.create<arith::MinimumFOp>(x, high);
      return x;
    };
    auto minval = builder.getF32FloatAttr(
        APInt::getSignedMinValue(dst_bitwidth).getSExtValue());
    auto maxval = builder.getF32FloatAttr(
        APInt::getSignedMaxValue(dst_bitwidth).getSExtValue());
    if (is_vector) {
      auto x_vty = cast<VectorType>(x.getType());
      x = clip(x, getFullVector(builder, x_vty, minval),
               getFullVector(builder, x_vty, maxval));
    } else {
      auto f32 = builder.getF32Type();
      x = clip(x, builder.create<arith::ConstantOp>(f32, minval),
               builder.create<arith::ConstantOp>(f32, maxval));
    }
  }
  if (is_vector) {
    x = builder.create<arith::FPToSIOp>(
        VectorType::get(src_vty.getShape(), builder.getI32Type()), x);
  } else {
    x = builder.create<arith::FPToSIOp>(builder.getI32Type(), x);
  }
  if (dst_bitwidth < 32) {
    x = builder.create<arith::TruncIOp>(op.getType(), x);
  }
  op.replaceAllUsesWith(x);
  op.erase();
  return success();
}

LogicalResult canonicalize_sitofp(const CanonicalizeContext &ctx,
                                  Operation &raw_op) {
  auto op = cast<arith::SIToFPOp>(raw_op);
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto src_vty = dyn_cast<VectorType>(op.getIn().getType());
  auto dst_vty = dyn_cast<VectorType>(op.getType());
  if (static_cast<bool>(src_vty) != static_cast<bool>(dst_vty)) {
    return op.emitOpError("Vector/scalar mismatch between input and output");
  }
  bool is_vector = static_cast<bool>(src_vty);
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned src_bitwidth,
                             getElementTypeBitwidth(op.getIn().getType()));
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned dst_bitwidth,
                             getElementTypeBitwidth(op.getType()));

  // We have low-level optimized code for s8->bf16 and s4->bf16 casts on v6.
  if (ctx.hardware_generation >= 6 && is_vector &&
      (src_vty.getElementType().isSignlessInteger(8) ||
       src_vty.getElementType().isSignlessInteger(4)) &&
      dst_vty.getElementType().isBF16()) {
    auto new_op = builder.create<tpu::SIToFPOp>(
        op.getType(), op.getIn(), tpu::RoundingMode::kToNearestEven);
    op.replaceAllUsesWith(new_op.getResult());
    op.erase();
    return success();
  }

  if ((src_bitwidth < 32 || dst_bitwidth < 32) && !ctx.compatibility_mode) {
    return op.emitOpError(
        "On this target integer-to-float conversions can only happen on "
        "32-bit values. Enable compatibility mode or upcast to int32, cast to "
        "float32 and truncate to desired bitwidth.");
  }

  // Canonicalize (intX -> floatY) to (intX -> int32 -> float32 -> floatY).
  Value x = op.getIn();
  if (src_bitwidth < 32) {
    if (is_vector) {
      x = builder.create<arith::ExtSIOp>(
          VectorType::get(src_vty.getShape(), builder.getI32Type()), x);
    } else {
      x = builder.create<arith::ExtSIOp>(builder.getI32Type(), x);
    }
  }
  if (is_vector) {
    x = builder.create<tpu::SIToFPOp>(
        VectorType::get(src_vty.getShape(), builder.getF32Type()), x,
        tpu::RoundingMode::kToNearestEven);
  } else {
    x = builder.create<tpu::SIToFPOp>(builder.getF32Type(), x,
                                      tpu::RoundingMode::kToNearestEven);
  }
  if (dst_bitwidth < 32) {
    x = builder.create<arith::TruncFOp>(op.getType(), x);
  }
  op.replaceAllUsesWith(x);
  op.erase();
  return success();
}

LogicalResult canonicalize_repeat(const CanonicalizeContext &ctx,
                                  Operation &raw_op) {
  auto op = dyn_cast<tpu::RepeatOp>(raw_op);
  if (!isa<VectorType>(op.getType())) {
    return op.emitOpError("Only vector types supported");
  }
  auto operand = op.getSource();
  auto times = op.getTimes();
  if (times == 1) {
    // A true no op - kind of an odd edge case, but this does come up in
    // flash_attention_backward tests.
    op.replaceAllUsesWith(operand);
    op.erase();
    return success();
  }
  auto operands = std::vector<Value>(times, operand);
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto concat = builder.create<tpu::ConcatenateOp>(op.getLoc(), op.getType(),
                                                   operands, op.getDimension());
  op.replaceAllUsesWith(concat.getResult());
  op.erase();
  return success();
}

LogicalResult canonicalize_vector_transpose(const CanonicalizeContext &ctx,
                                            Operation &raw_op) {
  auto op = cast<vector::TransposeOp>(raw_op);
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto new_op = builder.create<tpu::TransposeOp>(op.getType(), op.getVector(),
                                                 op.getPermutation());
  op.replaceAllUsesWith(new_op.getResult());
  op.erase();
  return success();
}

LogicalResult canonicalize_reshape(const CanonicalizeContext &ctx,
                                   Operation &raw_op) {
  auto op = cast<vector::ShapeCastOp>(raw_op);
  // We can canonicalize some reshape(load(x)) -> strided load + ALU ops.
  auto src = op.getSource();
  auto src_ty = src.getType();
  auto tgt_ty = op.getType();
  if (auto load_op = src.getDefiningOp<vector::LoadOp>()) {
    // Pattern match (..., M, N, 128) -> (..., M, N * 128).
    // This reshape can be folded into the load for any dtype and tiling
    // as long as the minormost dim is 128 and N is aligned to packing. The
    // pseudo code is:
    // ```
    // src_ref: (M, N, 128) with src_ty
    //
    // def load_to_reshape(src_ref):
    //   b_ref = src_ref.bitcast(i32) # i32[M, N / packing, 128]
    //   r_ref = b_ref.reshape(M * N / packing, 128)
    //   chunks = []
    //   for i in range(N / packing):
    //     v = r_ref[i::N / packing, :] # i32[M, 128]
    //     for j in range(packing):
    //       chunk = v >> (j * bitwidth)
    //       chunks.append(chunk)
    //   res = concat(chunks, axis=-1) # i32[M, N * 128]
    //   # int_src_ty refers to int type with the same bitwidth as src_ty.
    //   res = res.astype(int_src_ty) # Trigger i32 -> int_src_ty packing.
    //   return bitcast(res, src_ty) # src_ty[M, N * 128]
    // ```
    // TODO(jevinjiang): we can extend this to support folding more dims to last
    // dim not just last 2 dims.
    auto bitwidth = src_ty.getElementTypeBitWidth();
    auto packing = 32 / bitwidth;
    if (packing <= 0) {
      return op.emitOpError("Unsupported bitwidth = ") << bitwidth;
    }
    // Memref bitcast is not supported if HW generation is below 4. We don't
    // return failure because we will rely on vector reshape.
    if ((ctx.hardware_generation < 4 && packing > 1) ||
        (ctx.hardware_generation == 4 && packing > 2)) {
      return success();
    }
    auto ref = load_op.getBase();
    auto indices = load_op.getIndices();
    auto ref_shape = ref.getType().getShape();
    auto src_shape = src_ty.getShape();
    auto tgt_shape = tgt_ty.getShape();
    int ref_rank = ref_shape.size();
    int src_rank = src_shape.size();
    int tgt_rank = tgt_shape.size();
    if (ref_rank != src_rank) {
      return op.emitOpError("Loaded vector rank and memref rank mismatch");
    }
    // Check the memref's eligibility.
    if (!isContiguousMemref(ref) || ref_rank <= 2 ||
        // TODO(jevinjiang): add support for partial load on last 2 dims where
        // last 2 indices are not necessarily 0 or load shape is not full.
        getIntConst(indices[ref_rank - 1]) != 0 ||
        getIntConst(indices[ref_rank - 2]) != 0 ||
        ref_shape[ref_rank - 1] != src_shape[src_rank - 1] ||
        ref_shape[ref_rank - 2] != src_shape[src_rank - 2]) {
      return success();
    }
    // Check the reshape's eligibility.
    if (src_rank != tgt_rank + 1 || src_shape[src_rank - 2] % packing != 0 ||
        src_shape[src_rank - 1] != ctx.target_shape[1] ||
        src_shape[src_rank - 2] * src_shape[src_rank - 1] !=
            tgt_shape[tgt_rank - 1]) {
      return success();
    }
    // At this point, the pattern is matched.
    ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
    auto loc = op.getLoc();
    // First, we bitcast and reshape src ref from (..., M, N, 128) to
    // i32(..., M * N / packing, 128).
    SmallVector<int64_t> bitcast_shape(ref_shape);
    // TODO(jevinjiang): once we have memref pad op, we can use ceiling
    // division to ref_shape[ref_rank - 2] and packing to get sublane_cnt.
    CHECK_EQ(ref_shape[ref_rank - 2] % packing, 0);
    auto i32_2nd_minor_size = ref_shape[ref_rank - 2] / packing;
    bitcast_shape[ref_rank - 2] = i32_2nd_minor_size;
    auto i32_ref = builder.create<tpu::MemRefBitcastOp>(
        MemRefType::get(bitcast_shape, builder.getI32Type()), ref);

    SmallVector<int64_t> reshape_shape(ref_shape.begin(),
                                       ref_shape.begin() + tgt_rank);
    reshape_shape[tgt_rank - 1] = ctx.target_shape[1];
    reshape_shape[tgt_rank - 2] = ref_shape[ref_rank - 3] * i32_2nd_minor_size;
    auto reshape_ref = builder.create<tpu::MemRefReshapeOp>(
        MemRefType::get(reshape_shape, builder.getI32Type()), i32_ref);

    // We also need to transform the indices while transforming the memref.
    SmallVector<Value> new_indices(indices.begin(), indices.begin() + tgt_rank);
    new_indices[tgt_rank - 1] = IdxConst(0, builder, loc);
    new_indices[tgt_rank - 2] = builder.create<arith::MulIOp>(
        builder.getIndexType(), indices[ref_rank - 3],
        IdxConst(i32_2nd_minor_size, builder, loc));
    // Then, we strided load the bitcasted ref by stride (N / packing).
    int stride = i32_2nd_minor_size;
    // Expect to hold src_shape[src_rank - 2] number of chunks which have the
    // shape (..., src_shape[src_rank - 3], 128) and wait to be concatenated
    // along the last dim.
    SmallVector<Value> chunks(src_shape[src_rank - 2]);
    SmallVector<int64_t> chunk_shape(tgt_shape);
    chunk_shape[tgt_rank - 1] = ctx.target_shape[1];
    SmallVector<int32_t> strides(tgt_rank, 1);
    strides[tgt_rank - 2] = stride;
    auto tgt_2nd_minor_idx = new_indices[tgt_rank - 2];
    for (int i = 0; i < stride; ++i) {
      new_indices[tgt_rank - 2] = builder.create<arith::AddIOp>(
          builder.getIndexType(), tgt_2nd_minor_idx, IdxConst(i, builder, loc));
      auto chunk = builder.create<tpu::StridedLoadOp>(
          VectorType::get(chunk_shape, builder.getI32Type()), reshape_ref,
          new_indices, strides);
      for (int j = 0; j < packing; ++j) {
        int idx = i * packing + j;
        chunks[idx] = builder.create<arith::ShRUIOp>(
            chunk.getType(), chunk,
            I32Const(j * bitwidth, chunk_shape, builder, loc));
      }
    }
    // Concatenate the chunks along the last dim to get i32(..., M, N * 128).
    CHECK_GT(chunks.size(), 0);
    Value i32_tgt = chunks[0];
    if (chunks.size() > 1) {
      i32_tgt = builder.create<tpu::ConcatenateOp>(
          VectorType::get(tgt_shape, builder.getI32Type()), chunks,
          /*dimension=*/tgt_rank - 1);
    }
    Value tgt = i32_tgt;
    // Convert to target dtype.
    if (packing > 1) {
      tgt = builder.create<arith::TruncIOp>(
          VectorType::get(tgt_shape, builder.getIntegerType(bitwidth)),
          i32_tgt);
    }
    tgt = builder.create<arith::BitcastOp>(tgt_ty, tgt);
    op.replaceAllUsesWith(tgt);
    op.erase();
  }
  return success();
}

namespace {
// TODO(mvoz): We can refactor a lot of other canonicalization rules to use
// these functions.
// TODO(mvoz): I think we can eventually do direct conversion to bf16
// without going through f32?
Value upcastInt8ToBf16(ImplicitLocOpBuilder &builder, Value input) {
  auto vty = cast<VectorType>(input.getType());
  auto shape = vty.getShape();
  auto int_ty = cast<IntegerType>(vty.getElementType());

  auto i32_vty = VectorType::get(shape, builder.getI32Type());
  auto val_i32 = int_ty.isUnsigned()
                     ? builder.create<arith::ExtUIOp>(i32_vty, input)
                     : builder.create<arith::ExtSIOp>(i32_vty, input);

  auto f32_vty = VectorType::get(shape, builder.getF32Type());
  auto val_f32 = builder.create<tpu::SIToFPOp>(
      f32_vty, val_i32->getResult(0), tpu::RoundingMode::kToNearestEven);

  auto bf16_vty = VectorType::get(shape, builder.getBF16Type());
  return builder.create<arith::TruncFOp>(bf16_vty, val_f32);
}

Value downcastBf16ToInt8(ImplicitLocOpBuilder &builder, Value input_bf16,
                         Type target_vty) {
  auto shape = cast<VectorType>(input_bf16.getType()).getShape();

  auto f32_vty = VectorType::get(shape, builder.getF32Type());
  auto val_f32 = builder.create<arith::ExtFOp>(f32_vty, input_bf16);

  auto i32_vty = VectorType::get(shape, builder.getI32Type());
  auto val_i32 = builder.create<arith::FPToSIOp>(i32_vty, val_f32);

  return builder.create<arith::TruncIOp>(target_vty, val_i32);
}

Value upcastFp8ToBf16(ImplicitLocOpBuilder &builder, Value input) {
  auto shape = cast<VectorType>(input.getType()).getShape();
  auto f32_vty = VectorType::get(shape, builder.getF32Type());
  auto val_f32 = builder.create<arith::ExtFOp>(f32_vty, input);
  auto bf16_vty = VectorType::get(shape, builder.getBF16Type());
  return builder.create<arith::TruncFOp>(bf16_vty, val_f32);
}

Value downcastBf16ToFp8(ImplicitLocOpBuilder &builder, Value input_bf16,
                        Type target_vty) {
  auto shape = cast<VectorType>(input_bf16.getType()).getShape();
  auto f32_vty = VectorType::get(shape, builder.getF32Type());
  auto val_f32 = builder.create<arith::ExtFOp>(f32_vty, input_bf16);
  return builder.create<arith::TruncFOp>(target_vty, val_f32);
}
}  // namespace

// Note(mvoz): Returns optional to signal no replacement, simplifying downstream
// .replace() and .erase() calls.
std::optional<Value> canonicalize_transpose_impl(const CanonicalizeContext &ctx,
                                                 ImplicitLocOpBuilder &builder,
                                                 tpu::TransposeOp op) {
  auto input_ty = dyn_cast<VectorType>(op.getOperand().getType());
  auto element_type = input_ty.getElementType();
  // TODO(mvoz): Even gen 7 support is spotty on all test targets.
  if (element_type.getIntOrFloatBitWidth() == 8 && ctx.compatibility_mode &&
      ctx.hardware_generation > 3) {
    Value val_bf16;
    if (isa<IntegerType>(element_type)) {
      val_bf16 = upcastInt8ToBf16(builder, op.getOperand());
    } else {
      val_bf16 = upcastFp8ToBf16(builder, op.getOperand());
    }

    auto original_output_ty = cast<VectorType>(op.getType());
    auto post_transpose_bf16_vty =
        VectorType::get(original_output_ty.getShape(), builder.getBF16Type());

    auto new_t = builder.create<tpu::TransposeOp>(
        post_transpose_bf16_vty, val_bf16, op.getPermutation());

    Value final_val;
    if (isa<IntegerType>(element_type)) {
      final_val = downcastBf16ToInt8(builder, new_t.getResult(), op.getType());
    } else {
      final_val = downcastBf16ToFp8(builder, new_t.getResult(), op.getType());
    }
    return final_val;
  }
  return std::nullopt;
}

Value create_transpose_op(const CanonicalizeContext &ctx,
                          ImplicitLocOpBuilder &builder, VectorType input_ty,
                          Value input, ArrayRef<int64_t> permutation) {
  auto t = builder.create<tpu::TransposeOp>(input_ty, input, permutation);
  auto new_op_opt = canonicalize_transpose_impl(ctx, builder, t);
  if (new_op_opt.has_value()) {
    return new_op_opt.value();
  }
  return t;
}

LogicalResult canonicalize_transpose(const CanonicalizeContext &ctx,
                                     Operation &raw_op) {
  auto op = cast<tpu::TransposeOp>(raw_op);
  auto builder = ImplicitLocOpBuilder(op->getLoc(), op.getOperation());
  auto new_op_opt = canonicalize_transpose_impl(ctx, builder, op);
  if (new_op_opt.has_value()) {
    op.replaceAllUsesWith(new_op_opt.value());
    op.erase();
  }
  return success();
}

using canonicalize_rule_type =
    std::function<LogicalResult(const CanonicalizeContext &ctx, Operation &op)>;

const llvm::StringMap<canonicalize_rule_type> &rules() {
  static auto rules = new llvm::StringMap<canonicalize_rule_type>{
      {tpu::MatmulOp::getOperationName(), canonicalize_matmul},
      {vector::ContractionOp::getOperationName(), canonicalize_contraction},
      {vector::ExtractOp::getOperationName(), canonicalize_extract},
      {vector::MultiDimReductionOp::getOperationName(),
       canonicalize_multi_dim_reduction},
      {vector::TransposeOp::getOperationName(), canonicalize_vector_transpose},
      {vector::ShapeCastOp::getOperationName(), canonicalize_reshape},
      {vector::BroadcastOp::getOperationName(), canonicalize_broadcast},
      {arith::SelectOp::getOperationName(), canonicalize_select},
      {arith::FPToSIOp::getOperationName(), canonicalize_fptosi},
      {arith::SIToFPOp::getOperationName(), canonicalize_sitofp},
      {tpu::TransposeOp::getOperationName(), canonicalize_transpose},
      {tpu::RepeatOp::getOperationName(), canonicalize_repeat}};
  return *rules;
}

const llvm::StringMap<int> &bf16_ops_min_supported_versions() {
  constexpr int kAlwaysUpcast = std::numeric_limits<int>::max();
  static const auto m = new llvm::StringMap<int>{
      {arith::DivFOp::getOperationName(), 4},
      {arith::SelectOp::getOperationName(), 5},
      {arith::CmpFOp::getOperationName(), 5},
      {arith::MulFOp::getOperationName(), 6},
      {arith::AddFOp::getOperationName(), 6},
      {arith::SubFOp::getOperationName(), 6},
      {arith::MaximumFOp::getOperationName(), 6},
      {arith::MinimumFOp::getOperationName(), 6},
      {math::PowFOp::getOperationName(), kAlwaysUpcast},
      {math::TanhOp::getOperationName(), 6},
      {math::ExpOp::getOperationName(), 6},
      {math::Exp2Op::getOperationName(), 6},
      {math::LogOp::getOperationName(), 6},
  };
  return *m;
}

bool need_elementwise_canonicalization(const CanonicalizeContext &ctx,
                                       Operation &op) {
  // Only rewrite when the hardware generation is below the minimum supported
  // version.
  auto it = bf16_ops_min_supported_versions().find(op.getName().getStringRef());
  if (it == bf16_ops_min_supported_versions().end() ||
      ctx.hardware_generation >= it->second) {
    return false;
  }
  return llvm::any_of(op.getOperands(), [](Value operand) {
    auto vty = dyn_cast<VectorType>(operand.getType());
    return vty && vty.getElementType().isBF16();
  });
}

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer(int hardware_generation, bool compatibility_mode,
                      std::array<int64_t, 2> target_shape)
      : hardware_generation_(hardware_generation),
        compatibility_mode_(compatibility_mode),
        target_shape_(target_shape) {}

  int hardware_generation_;
  bool compatibility_mode_;
  std::array<int64_t, 2> target_shape_;

  LogicalResult canonicalize(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return canonicalizeBlock(op.getBody().front());
  }

  LogicalResult canonicalizeBlock(Block &block) {
    // make_early_inc_range is utilized due to op mutation.
    for (Operation &any_op : make_early_inc_range(block)) {
      if (canonicalizeOp(any_op).failed()) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult canonicalizeOp(Operation &any_op) {
    CanonicalizeContext ctx(
        {compatibility_mode_, hardware_generation_, target_shape_});
    // We must iterate over the op first, because canonicalization can cause
    // us to .erase() an op, and accessing getRegions on it after is not
    // sound. Invariant - top level ops with regions may never be invalidated.
    for (Region &region : any_op.getRegions()) {
      for (Block &block : region) {
        if (canonicalizeBlock(block).failed()) {
          return failure();
        }
      }
    }
    if (need_elementwise_canonicalization(ctx, any_op)) {
      return canonicalize_elementwise(ctx, any_op);
    }
    if (auto rule_it = rules().find(any_op.getName().getStringRef());
        rule_it != rules().end()) {
      const canonicalize_rule_type &rule = rule_it->getValue();
      return rule(ctx, any_op);
    }
    return success();
  }
};

struct CanonicalizeMosaicPass
    : public impl::CanonicalizeMosaicPassBase<CanonicalizeMosaicPass> {
  CanonicalizeMosaicPass(int hardware_generation_p, bool compatibility_mode_p,
                         std::array<int64_t, 2> target_shape)
      : compatibility_mode_(compatibility_mode_p) {
    this->hardware_generation = hardware_generation_p;
    this->sublane_count = target_shape[0];
    this->lane_count = target_shape[1];
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc(hardware_generation, compatibility_mode_,
                            {sublane_count, lane_count});
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };

  bool compatibility_mode_;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass(
    int hardware_generation, bool compatibility_mode,
    std::array<int64_t, 2> target_shape) {
  return std::make_unique<CanonicalizeMosaicPass>(
      hardware_generation, compatibility_mode, target_shape);
}

}  // namespace mlir::tpu
