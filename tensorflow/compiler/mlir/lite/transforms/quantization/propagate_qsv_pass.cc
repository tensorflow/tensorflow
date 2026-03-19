/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass propagates QSV information through the model.

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_interface.h.inc"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/transforms/quantization/quant_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/shape_and_size_utils.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_PROPAGATEQSVPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

//-------------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Returns true if any of the op's operands are per-axis quantized.
bool HasPerAxisQuantizedOperand(mlir::Operation* op) {
  for (const auto& operand : op->getOperands()) {
    auto qtype = GetQTypeFromDefiningDequantize(operand);
    if (qtype.has_value() &&
        dyn_cast<quant::UniformQuantizedPerAxisType>(*qtype)) {
      return true;
    }
  }
  return false;
}

// Propagates the quantized type `qtype` to all float operands and results of
// `same_scales_op` by inserting QDQ pairs. This is only used for ops that have
// the SameScalesOpInterface.
LogicalResult PropagateQsvAcrossOperandsAndResults(
    SameScalesOpInterface same_scales_op, quant::QuantizedType qtype,
    PatternRewriter& rewriter) {
  mlir::Operation* op = same_scales_op.getOperation();
  bool changed = false;

  auto propagate_across = [&](auto&& values, auto get_qtype_fn,
                              mlir::Operation* target_op) -> LogicalResult {
    for (mlir::Value value : values) {
      if (!isa<FloatType>(getElementTypeOrSelf(value))) continue;
      std::optional<quant::QuantizedType> current_qtype = get_qtype_fn(value);
      if (!current_qtype || *current_qtype != qtype) {
        if (failed(InsertQDQ(value, qtype, rewriter, target_op))) {
          return failure();
        }
        changed = true;
      }
    }
    return success();
  };

  if (failed(propagate_across(op->getOperands(),
                              &GetQTypeFromDefiningDequantize, op))) {
    return failure();
  }

  // ops like `tfl.split` can have multiple results, and we should propagate
  // qsv to all of them. If ops are found for which some known result indices
  // only need propagation, we could add an method to the SameScalesOpInterface
  // to provide that information.
  if (failed(propagate_across(op->getResults(), &GetQTypeFromConsumingQuantize,
                              nullptr))) {
    return failure();
  }

  return changed ? success()
                 : rewriter.notifyMatchFailure(same_scales_op, "No change.");
}

// Calculates the new quantization dimension for a per-axis quantized tensor
// that has been transposed. `transpose_op` is the op to inspect, `quant_dim`
// is the original quantization dimension. The new dimension is returned via
// `new_quant_dim`. Returns a `LogicalResult` to indicate success or failure of
// parsing the transpose op.
LogicalResult GetQuantDimensionAfterTranspose(TFL::TransposeOp transpose_op,
                                              const int32_t quant_dim,
                                              PatternRewriter& rewriter,
                                              int32_t& new_quant_dim) {
  auto input_type =
      mlir::cast<mlir::ShapedType>(transpose_op.getInput().getType());
  auto perm_type =
      mlir::cast<mlir::ShapedType>(transpose_op.getPerm().getType());
  if (input_type.hasStaticShape() && perm_type.hasStaticShape()) {
    if (perm_type.getNumElements() != input_type.getRank()) {
      return transpose_op.emitOpError(
          "perm tensor elements size is not equal to input tensor rank");
    }
  }

  // Get permutation axes of the TransposeOp
  DenseIntElementsAttr perm;
  if (!matchPattern(transpose_op.getPerm(), m_Constant(&perm))) {
    return rewriter.notifyMatchFailure(transpose_op, "perm is not a constant");
  }

  SmallVector<int64_t, 4> axes;
  axes.reserve(perm.getNumElements());
  absl::flat_hash_set<int64_t> seen_axes;
  for (const auto& axis_int : perm.getValues<APInt>()) {
    int64_t axis = axis_int.getSExtValue();
    if (axis < 0) {
      axis += input_type.getRank();
    }
    if (axis < 0 || (input_type.hasRank() && axis >= input_type.getRank())) {
      return transpose_op.emitOpError("perm must be in [-rank, rank)");
    }
    if (!seen_axes.insert(axis).second) {
      return transpose_op.emitOpError("perm cannot have duplicated axis");
    }
    axes.push_back(axis);
  }

  // Find what the quantized dimension has been transposed to
  const auto it = std::find(axes.begin(), axes.end(), quant_dim);
  if (it == axes.end()) {
    llvm_unreachable(
        "quantized dimension should be present in a valid permutation");
  }
  new_quant_dim = std::distance(axes.begin(), it);
  return success();
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

class PropagateReshapedPerAxisQuantDim
    : public OpRewritePattern<TFL::ReshapeOp> {
 public:
  explicit PropagateReshapedPerAxisQuantDim(MLIRContext* context)
      : OpRewritePattern<TFL::ReshapeOp>(context) {}
  LogicalResult matchAndRewrite(TFL::ReshapeOp reshape_op,
                                PatternRewriter& rewriter) const override {
    std::optional<quant::QuantizedType> qtype =
        GetQTypeFromDefiningDequantize(reshape_op.getOperand(0));
    if (!qtype.has_value()) {
      return rewriter.notifyMatchFailure(reshape_op,
                                         "input is not a dequantize op");
    }
    auto per_axis_quant = dyn_cast<quant::UniformQuantizedPerAxisType>(*qtype);
    if (!per_axis_quant) {
      return rewriter.notifyMatchFailure(reshape_op,
                                         "input is not per-axis quantized");
    }

    // Return if the result of ReshapeOp is already quantized
    if (GetQTypeFromConsumingQuantize(reshape_op.getResult())) {
      return rewriter.notifyMatchFailure(reshape_op,
                                         "output is already quantized");
    }

    // Get the new quantization dimension
    absl::StatusOr<int32_t> new_quant_dim = GetQuantDimensionAfterReshape(
        reshape_op.getInput().getType().getShape(),
        reshape_op.getType().getShape(),
        per_axis_quant.getQuantizedDimension());

    if (!new_quant_dim.ok()) {
      return rewriter.notifyMatchFailure(
          reshape_op, "Invalid quantization dimension after ReshapeOp");
    }

    // Insert a QDQ pair with the new quantized dimension after ReshapeOp
    auto new_element_type =
        mlir::quant::UniformQuantizedPerAxisType::getChecked(
            reshape_op.getLoc(), per_axis_quant.getFlags(),
            per_axis_quant.getStorageType(), per_axis_quant.getExpressedType(),
            per_axis_quant.getScales(), per_axis_quant.getZeroPoints(),
            *new_quant_dim, per_axis_quant.getStorageTypeMin(),
            per_axis_quant.getStorageTypeMax());

    if (failed(InsertQDQ(reshape_op.getResult(), new_element_type, rewriter))) {
      return failure();
    }

    return success();
  }
};

class PropagateTransposedPerAxisQuantDim
    : public OpRewritePattern<TFL::TransposeOp> {
 public:
  explicit PropagateTransposedPerAxisQuantDim(MLIRContext* context)
      : OpRewritePattern<TFL::TransposeOp>(context) {}
  LogicalResult matchAndRewrite(TFL::TransposeOp transpose_op,
                                PatternRewriter& rewriter) const override {
    std::optional<quant::QuantizedType> qtype =
        GetQTypeFromDefiningDequantize(transpose_op.getOperand(0));
    if (!qtype.has_value()) {
      return rewriter.notifyMatchFailure(transpose_op,
                                         "input is not a dequantize op");
    }
    auto per_axis_quant = dyn_cast<quant::UniformQuantizedPerAxisType>(*qtype);
    if (!per_axis_quant) {
      return rewriter.notifyMatchFailure(transpose_op,
                                         "input is not per-axis quantized");
    }

    // Return if the result of TransposeOp is already quantized
    if (GetQTypeFromConsumingQuantize(transpose_op.getResult())) {
      return rewriter.notifyMatchFailure(transpose_op,
                                         "output is already quantized");
    }

    int32_t new_out_quant_dim;
    if (failed(GetQuantDimensionAfterTranspose(
            transpose_op, per_axis_quant.getQuantizedDimension(), rewriter,
            new_out_quant_dim))) {
      return failure();
    }

    // Insert a QDQ pair with the new quantized dimension after TransposeOp
    auto new_element_type =
        mlir::quant::UniformQuantizedPerAxisType::getChecked(
            transpose_op.getLoc(), per_axis_quant.getFlags(),
            per_axis_quant.getStorageType(), per_axis_quant.getExpressedType(),
            per_axis_quant.getScales(), per_axis_quant.getZeroPoints(),
            new_out_quant_dim, per_axis_quant.getStorageTypeMin(),
            per_axis_quant.getStorageTypeMax());

    if (failed(
            InsertQDQ(transpose_op.getResult(), new_element_type, rewriter))) {
      return failure();
    }

    return success();
  }
};

class PropagateQsv : public OpInterfaceRewritePattern<SameScalesOpInterface> {
  using OpInterfaceRewritePattern<
      SameScalesOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(SameScalesOpInterface op,
                                PatternRewriter& rewriter) const final {
    // The per-axis quantized ops that don't directly transfer the quantized
    // types from input to output (e.g. TransposeOp, ReshapeOp), need dedicated
    // propagation patterns.
    if (!op.RequiredSameQuantizedAxes() && HasPerAxisQuantizedOperand(op)) {
      return rewriter.notifyMatchFailure(
          op, "requires dedicated propagation pattern.");
    }

    std::optional<quant::QuantizedType> propagated_type = GetPropagatedType(op);
    if (!propagated_type) {
      return rewriter.notifyMatchFailure(op, "No propagated type found.");
    }
    return PropagateQsvAcrossOperandsAndResults(op, *propagated_type, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct PropagateQsvPass : public impl::PropagateQsvPassBase<PropagateQsvPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateQsvPass)

  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
#include "tensorflow/compiler/mlir/lite/transforms/quantization/generated_strict_quantize.inc"

void PropagateQsvPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<PropagateQsv>(ctx);

  // Dedicated propagation patterns.
  patterns.add<PropagateTransposedPerAxisQuantDim,
               PropagateReshapedPerAxisQuantDim>(ctx);

  GreedyRewriteConfig greedy_config;
  greedy_config.enableFolding(false);
  if (failed(
          applyPatternsGreedily(module, std::move(patterns), greedy_config))) {
    module.emitError("Failed to apply PropagateQsvPass patterns.");
    signalPassFailure();
    return;
  }

  // Optimize to Requantize and clean up redundant Q-DQs.
  RewritePatternSet cleanup_patterns(ctx);
  cleanup_patterns.add<FuseDqQToRequant, FuseQQToRequant, RemoveNoOpQ>(ctx);
  if (failed(applyPatternsGreedily(module, std::move(cleanup_patterns),
                                   greedy_config))) {
    module.emitError(
        "Failed to apply requant and clean up patterns during QSV "
        "propagation.");
    signalPassFailure();
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> CreatePropagateQsvPass() {
  return std::make_unique<PropagateQsvPass>();
}

}  // namespace TFL
}  // namespace mlir
