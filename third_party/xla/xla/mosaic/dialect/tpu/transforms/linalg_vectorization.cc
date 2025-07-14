/* Copyright 2023 The JAX Authors.

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

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_LINALGVECTORIZATIONPASS
#define GEN_PASS_DEF_LINALGVECTORIZATIONPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {
struct VectorizationPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return vectorize(rewriter, op,
                     /*inputVectorSizes=*/{},
                     /*inputScalableVecDims=*/{},
                     /*vectorizeNDExtract=*/false);
  }
};

// Check preconditions for `vector.transfer_read` rewrite patterns.
LogicalResult checkPreconditions(vector::TransferReadOp op,
                                 PatternRewriter &rewriter) {
  if (op.hasOutOfBoundsDim()) {
    return rewriter.notifyMatchFailure(op, "out of bounds transfer dim");
  }
  if (op.getMask()) {
    return rewriter.notifyMatchFailure(op, "masked transfer");
  }
  if (!op.getPermutationMap().isIdentity()) {
    return rewriter.notifyMatchFailure(op, "non identity permutation map");
  }
  SmallVector<Value> indices = {op.getIndices().begin(), op.getIndices().end()};
  if (absl::c_any_of(
          indices, [](Value index) { return !isConstantIntValue(index, 0); })) {
    return rewriter.notifyMatchFailure(op, "non zero indices");
  }
  return success();
}

// Create a `vector.transfer_read` based on the original |op|, which succeeds
// the checkPreconditions() call.
vector::TransferReadOp createTransferReadOp(vector::TransferReadOp op,
                                            Value source,
                                            RankedTensorType source_ty,
                                            PatternRewriter &rewriter) {
  // We know from preconditions that there are no out of bound dims.
  SmallVector<bool> in_bounds(source_ty.getRank(), true);
  return rewriter.create<vector::TransferReadOp>(
      op.getLoc(),
      VectorType::get(source_ty.getShape(), source_ty.getElementType()), source,
      SmallVector<Value>(
          source_ty.getRank(),
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0)),
      AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(source_ty.getRank(),
                                                           op->getContext())),
      rewriter.getBoolArrayAttr(in_bounds));
}

template <typename DefiningOp>
LogicalResult matchAndRewriteTransferOfExpandOrCollapseShape(
    vector::TransferReadOp op, PatternRewriter &rewriter) {
  if (failed(checkPreconditions(op, rewriter))) {
    return failure();
  }
  auto expand = op.getSource().template getDefiningOp<DefiningOp>();
  if (!expand) {
    return rewriter.notifyMatchFailure(
        op, "not a tensor.expand_shape/collapse_shape");
  }
  if (auto result_type = dyn_cast<ShapedType>(op.getType());
      !result_type ||
      result_type.getShape() != expand.getResultType().getShape()) {
    return rewriter.notifyMatchFailure(op, "output type mismatch");
  }
  auto expand_src_type = expand.getSrcType();
  // We know from preconditions that there are no out of bound dims.
  SmallVector<bool> in_bounds(expand_src_type.getRank(), true);
  rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
      op, op.getType(),
      createTransferReadOp(op, expand.getSrc(), expand_src_type, rewriter));
  return success();
}

// Rewrite `vector.transfer_read(tensor.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfExpandShape
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteTransferOfExpandOrCollapseShape<
        tensor::ExpandShapeOp>(op, rewriter);
  }
};

// Rewrite `vector.transfer_read(tensor.collapse_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfCollapseShape
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteTransferOfExpandOrCollapseShape<
        tensor::CollapseShapeOp>(op, rewriter);
  }
};

// Rewrite a `vector.transfer_read` of a dense tensor constant as a dense
// vector constant.
struct TransferReadOfConstant
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr constant_elements;
    Attribute constant_value;
    if (matchPattern(op.getSource(), m_Constant(&constant_elements)) &&
        constant_elements.isSplat()) {
      constant_value = constant_elements.getSplatValue<Attribute>();
    } else {
      return rewriter.notifyMatchFailure(op, "not an arith.constant");
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, op.getVectorType(),
        DenseElementsAttr::get(op.getVectorType(), constant_value));
    return success();
  }
};

// Rewrite `vector.transfer_read(arith.select)` as `arith.select` with
// `transfer_read` applied to its operands.
struct TransferReadOfSelect : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(op, rewriter))) {
      return failure();
    }
    auto select = op.getSource().getDefiningOp<arith::SelectOp>();
    if (!select) {
      return rewriter.notifyMatchFailure(op, "source not an arith.select");
    }
    auto true_value_ty =
        dyn_cast<RankedTensorType>(select.getTrueValue().getType());
    if (!true_value_ty) {
      return rewriter.notifyMatchFailure(
          op, "true value is not a ranked tensor type");
    }
    // We do not check the type of the false_value since the verifier enforces
    // that types of true_value, false_value, and result match.
    auto false_value_ty =
        dyn_cast<RankedTensorType>(select.getFalseValue().getType());
    auto condition_type =
        dyn_cast<RankedTensorType>(select.getCondition().getType());
    if (!condition_type) {
      return rewriter.notifyMatchFailure(
          op, "condition is not a ranked tensor type");
    }
    auto transfer_read = [&](Value value, RankedTensorType type) {
      return createTransferReadOp(op, value, type, rewriter);
    };
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, transfer_read(select.getCondition(), condition_type),
        transfer_read(select.getTrueValue(), true_value_ty),
        transfer_read(select.getFalseValue(), false_value_ty));
    return success();
  }
};

// Rewrite `vector.transfer_read(arith.cmpi)` as `arith.cmpi` with
// `transfer_read` applied to its operands.
struct TransferReadOfCmpI : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(op, rewriter))) {
      return failure();
    }
    auto cmp = op.getSource().getDefiningOp<arith::CmpIOp>();
    if (!cmp) {
      return rewriter.notifyMatchFailure(op, "source not an arith.cmpi");
    }
    auto lhs_type = dyn_cast<RankedTensorType>(cmp.getLhs().getType());
    if (!lhs_type) {
      return rewriter.notifyMatchFailure(op, "lhs is not a ranked tensor type");
    }
    auto rhs_type = dyn_cast<RankedTensorType>(cmp.getRhs().getType());
    if (!rhs_type) {
      return rewriter.notifyMatchFailure(op, "rhs is not a ranked tensor type");
    }
    auto transfer_read = [&](Value value, RankedTensorType type) {
      return createTransferReadOp(op, value, type, rewriter);
    };
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, cmp.getPredicate(), transfer_read(cmp.getLhs(), lhs_type),
        transfer_read(cmp.getRhs(), rhs_type));
    return success();
  }
};

// Rewrite `vector.transfer_read(tensor.splat)` as `vector.broadcast`.
struct TransferReadOfSplat : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(op, rewriter))) {
      return failure();
    }
    auto splat = op.getSource().getDefiningOp<tensor::SplatOp>();
    if (!splat) {
      return rewriter.notifyMatchFailure(op, "source not a tensor.splat");
    }
    if (!splat.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "not statically shaped");
    }
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, op.getVectorType(),
                                                     splat.getInput());
    return success();
  }
};

// List of operations that are covered by the supports_bf16_alu_instructions.
const auto kSupportedBf16Ops = absl::flat_hash_set<absl::string_view>(
    {arith::AddFOp::getOperationName(), arith::SubFOp::getOperationName(),
     arith::MulFOp::getOperationName(), arith::MaximumFOp::getOperationName(),
     arith::MinimumFOp::getOperationName()});

// Rewrite operation with bf16 inputs/outputs into an operation with f32
// inputs/outputs, where the inputs are extended and the outputs truncated.
// Non-bf16 operands remain unchanged.
// TODO(b/324596736): Extend the functionality to int8 and int16.
class GenericBitwidthConvert : public RewritePattern {
 public:
  explicit GenericBitwidthConvert(llvm::StringRef operation_name,
                                  MLIRContext *ctx,
                                  bool supports_bf16_alu_instructions)
      : RewritePattern(operation_name, 0, ctx),
        supports_bf16_alu_instructions_(supports_bf16_alu_instructions) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (supports_bf16_alu_instructions_ &&
        kSupportedBf16Ops.contains(op->getName().getStringRef())) {
      return rewriter.notifyMatchFailure(op, "target supports bf16 operands");
    }
    llvm::SmallVector<Value> extended_operands;
    extended_operands.reserve(op->getOperands().size());
    Location loc = op->getLoc();
    bool has_bf16_operand = false;
    for (Value operand : op->getOperands()) {
      auto operand_type = dyn_cast<VectorType>(operand.getType());
      if (!operand_type) {
        return rewriter.notifyMatchFailure(op, "operand not a vector");
      }
      if (!operand_type.getElementType().isBF16()) {
        // Add the operand as is and continue, since not all operands must be
        // bf16, for example in the case of a select op.
        extended_operands.push_back(operand);
        continue;
      }
      has_bf16_operand = true;
      extended_operands.push_back(rewriter.create<arith::ExtFOp>(
          loc, VectorType::get(operand_type.getShape(), rewriter.getF32Type()),
          operand));
    }
    // If there are no bf16 operands, then we do not need to rewrite the op.
    if (!has_bf16_operand) {
      return rewriter.notifyMatchFailure(op, "no bf16 operands");
    }
    llvm::SmallVector<Type> new_results;
    new_results.reserve(op->getResultTypes().size());
    for (Type result_ty : op->getResultTypes()) {
      auto result_type = dyn_cast<VectorType>(result_ty);
      if (!result_type) {
        return rewriter.notifyMatchFailure(op, "result is not a vector");
      }
      if (!result_type.getElementType().isBF16()) {
        return rewriter.notifyMatchFailure(op,
                                           "result element type is not bf16");
      }
      new_results.push_back(
          VectorType::get(result_type.getShape(), rewriter.getF32Type()));
    }
    OperationState state(loc, op->getName().getStringRef(), extended_operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    Operation *new_op = rewriter.create(state);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, op->getResultTypes(),
                                                 new_op->getResults());
    return success();
  }

 private:
  // Whether the target supports bf16 ALU instructions.
  const bool supports_bf16_alu_instructions_;
};

// Rewrite `vector.contraction` with bf16 accumulator and output into a
// contraction with f32 accumulator and output, where the accumulator is
// extended and the output truncated. For targets that do not support bf16
// matmul, the lhs and rhs are extended to f32.
struct ContractionBitwidthConvert
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  ContractionBitwidthConvert(bool supports_bf16_matmul, MLIRContext *ctx)
      : OpRewritePattern(ctx), supports_bf16_matmul_(supports_bf16_matmul) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // The ContractionOp contract is that (1) lhs and rhs have same element
    // type, and (2) the accumulator and result have the same element type.

    // If the target does not support bf16 matmul and we have bf16 operands, we
    // need to extend the lhs and rhs to f32.
    const bool extend_operands =
        op.getLhsType().getElementType().isBF16() && !supports_bf16_matmul_;
    // Determine if the accumulator is bf16 and hence needs to be extended to
    // f32.
    ShapedType acc_ty = dyn_cast<ShapedType>(op.getAccType());
    if (acc_ty == nullptr) {
      return rewriter.notifyMatchFailure(op,
                                         "accumulator is not a shaped type");
    }
    const bool extend_acc = acc_ty.getElementType().isBF16();

    if (!extend_operands && !extend_acc) {
      return rewriter.notifyMatchFailure(op, "no bf16 operands or accumulator");
    }

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (extend_operands) {
      lhs = rewriter.create<arith::ExtFOp>(
          op.getLoc(),
          VectorType::get(op.getLhsType().getShape(), rewriter.getF32Type()),
          lhs);
      rhs = rewriter.create<arith::ExtFOp>(
          op.getLoc(),
          VectorType::get(op.getRhsType().getShape(), rewriter.getF32Type()),
          rhs);
    }

    Value acc = op.getAcc();
    if (extend_acc) {
      acc = rewriter.create<arith::ExtFOp>(
          op.getLoc(),
          VectorType::get(acc_ty.getShape(), rewriter.getF32Type()),
          op.getAcc());
    }

    vector::ContractionOp contraction = rewriter.create<vector::ContractionOp>(
        op.getLoc(), lhs, rhs, acc, op.getIndexingMaps(), op.getIteratorTypes(),
        op.getKind());

    if (extend_acc) {
      rewriter.replaceOpWithNewOp<arith::TruncFOp>(
          op, dyn_cast<ShapedType>(op.getResultType()), contraction);
    } else {
      rewriter.replaceOp(op, contraction);
    }
    return success();
  }

 private:
  const bool supports_bf16_matmul_;
};

// Rewrite `vector.multi_dim_reduction` with bf16 source/accumulator/output into
// a multi_dim_reduction with f32 source/accumulator/output, where the source
// and accumulator are extended and the result is truncated.
// TODO(b/324596736): Make the rewrite conditional on the target supporting
// bf16 reductions.
struct MultiDimReductionBitwidthConvert
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    // Below we rely on the contract that the source operand, accumulator, and
    // result have the same element type.
    auto src_ty = op.getSourceVectorType();
    if (!src_ty.getElementType().isBF16()) {
      return rewriter.notifyMatchFailure(op, "not bf16 reduction");
    }

    auto res_ty = dyn_cast<VectorType>(op.getResult().getType());
    if (!res_ty) {
      return rewriter.notifyMatchFailure(op, "not vector reduction");
    }

    auto reduction = rewriter.create<vector::MultiDimReductionOp>(
        op.getLoc(),
        rewriter.create<arith::ExtFOp>(
            op.getLoc(),
            VectorType::get(src_ty.getShape(), rewriter.getF32Type()),
            op.getSource()),
        rewriter.create<arith::ExtFOp>(
            op.getLoc(),
            VectorType::get(res_ty.getShape(), rewriter.getF32Type()),
            op.getAcc()),
        op.getReductionMask(), op.getKind());
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, res_ty, reduction);
    return success();
  }
};

struct LinalgVectorizationPass
    : public impl::LinalgVectorizationPassBase<LinalgVectorizationPass> {
  explicit LinalgVectorizationPass(
      const LinalgVectorizationPassOptions &options)
      : impl::LinalgVectorizationPassBase<LinalgVectorizationPass>(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<VectorizationPattern>(ctx);
    // Pull in patterns to shuffle broadcast/transpose ops around in order to
    // cancel them or embed into contract ops. Embedding in the flexible
    // contract ops will help to sustain the structure through various
    // transformations.
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);
    // Pull in patterns to canonicalize transfer ops.
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
    patterns.add<TransferReadOfCmpI, TransferReadOfCollapseShape,
                 TransferReadOfConstant, TransferReadOfExpandShape,
                 TransferReadOfSelect, TransferReadOfSplat>(ctx);
    // Pull in patterns to convert bf16 ops to f32 ops.
    for (::llvm::StringLiteral unary_op_name :
         {arith::NegFOp::getOperationName(), math::TanhOp::getOperationName(),
          math::ExpOp::getOperationName(), math::AbsFOp::getOperationName(),
          math::SinOp::getOperationName(), math::CosOp::getOperationName(),
          math::SqrtOp::getOperationName(), math::RsqrtOp::getOperationName(),
          math::LogOp::getOperationName(), math::Log1pOp::getOperationName(),
          math::RoundOp::getOperationName(),
          math::RoundEvenOp::getOperationName()}) {
      patterns.add<GenericBitwidthConvert>(unary_op_name, ctx,
                                           supports_bf16_alu_instructions);
    }
    for (::llvm::StringLiteral binary_op_name :
         {arith::MulFOp::getOperationName(), arith::DivFOp::getOperationName(),
          arith::AddFOp::getOperationName(), arith::SubFOp::getOperationName(),
          arith::MaximumFOp::getOperationName(),
          arith::MinimumFOp::getOperationName(),
          math::PowFOp::getOperationName()}) {
      patterns.add<GenericBitwidthConvert>(binary_op_name, ctx,
                                           supports_bf16_alu_instructions);
    }
    for (::llvm::StringLiteral ternary_op_name :
         {arith::SelectOp::getOperationName()}) {
      patterns.add<GenericBitwidthConvert>(ternary_op_name, ctx,
                                           supports_bf16_alu_instructions);
    }
    patterns.add<ContractionBitwidthConvert>(supports_bf16_matmul, ctx);
    patterns.add<MultiDimReductionBitwidthConvert>(ctx);

    // We do not want to apply the vector patterns above to the ops that are
    // unrelated to the original linalg op.
    SmallVector<Operation *> linalgOps;
    func.walk([&](Operation *op) {
      if (dyn_cast<arith::SelectOp>(op) || dyn_cast<linalg::LinalgOp>(op) ||
          dyn_cast<vector::TransferReadOp>(op) ||
          dyn_cast<vector::TransferWriteOp>(op) ||
          dyn_cast<vector::ContractionOp>(op) ||
          dyn_cast<vector::MultiDimReductionOp>(op)) {
        linalgOps.push_back(op);
      }
    });
    if (failed(applyOpPatternsAndFold(linalgOps, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgVectorizationPass(
    bool supports_bf16_alu_instructions, bool supports_bf16_matmul) {
  LinalgVectorizationPassOptions options;
  options.supports_bf16_alu_instructions = supports_bf16_alu_instructions;
  options.supports_bf16_matmul = supports_bf16_matmul;
  return std::make_unique<LinalgVectorizationPass>(options);
}

}  // namespace mlir::tpu
