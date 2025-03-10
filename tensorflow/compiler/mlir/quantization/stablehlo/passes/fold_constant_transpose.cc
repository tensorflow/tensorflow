/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/permutation.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_FOLDCONSTANTTRANSPOSEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// Returns contiguous offset (address) of the position represented by `indices`
// in a `shape` shaped tensor. Assumes row-major order. `indices` and `shape`
// should have the same size.
// Example: Index (2, 3) of a (4, 5)-shaped tensor has the contiguous offset of
// 2 * 5 + 3 = 13.
int64_t GetContiguousOffset(const ArrayRef<int64_t> indices,
                            const ArrayRef<int64_t> shape) {
  int64_t contiguous_offset = 0;
  int64_t base_offset = 1;
  for (auto [i, dimension] : llvm::reverse(llvm::zip_equal(indices, shape))) {
    contiguous_offset += base_offset * i;
    base_offset *= dimension;
  }

  return contiguous_offset;
}

// Performs transposition of a tensor represented as a contiguous element array.
// Assumes row-major order. The shape of the input tensor and the desired
// permutation is registered during construction, and calling `TransposeValues`
// returns the transposed tensor values.
class DenseElementsTransposer {
 public:
  DenseElementsTransposer(const ArrayRef<int64_t> original_shape,
                          const ArrayRef<int64_t> permutation)
      : rank_(original_shape.size()),
        original_shape_(original_shape),
        target_shape_(Permute<int64_t>(original_shape, permutation)),
        permutation_(permutation) {}

  // Transposes `values` with the permutation. Returns the transposed values.
  SmallVector<float> TransposeValues(const ArrayRef<float> values) const {
    SmallVector<float> transposed_values(values.size());
    SmallVector<int64_t> current_indices = {};
    TransposeRecursively(values, transposed_values, current_indices);

    return transposed_values;
  }

  // Returns the shape after permutation.
  SmallVector<int64_t> GetTargetShape() const { return target_shape_; }

 private:
  // Helper function that performs transposition recursively by mapping each set
  // of indices from the original values to the target values.
  void TransposeRecursively(const ArrayRef<float> original_values,
                            const MutableArrayRef<float> target_values,
                            SmallVector<int64_t>& current_indices) const {
    // Map an element from `original_values` to `target_values` when a set of
    // indices is formed.
    if (current_indices.size() == rank_) {
      const int64_t original_index =
          GetContiguousOffset(current_indices, original_shape_);

      const SmallVector<int64_t> target_indices =
          Permute<int64_t>(current_indices, permutation_);
      const int64_t target_index =
          GetContiguousOffset(target_indices, target_shape_);

      target_values[target_index] = original_values[original_index];
      return;
    }

    // Recursively iterate by selecting the index of the next dimension.
    const int next_shape_idx = current_indices.size();
    for (int i = 0; i < original_shape_[next_shape_idx]; ++i) {
      current_indices.push_back(i);
      TransposeRecursively(original_values, target_values, current_indices);
      current_indices.pop_back();
    }
  }

  int rank_;                             // Rank of the input values.
  SmallVector<int64_t> original_shape_;  // Shape of the original tensor.
  SmallVector<int64_t> target_shape_;    // Shape of the target tensor.
  SmallVector<int64_t> permutation_;
};

class FoldTransposedConstantOp
    : public OpRewritePattern<
          mlir::stablehlo::TransposeOp>::SplitMatchAndRewrite {
 public:
  using SplitMatchAndRewrite::SplitMatchAndRewrite;

  LogicalResult match(mlir::stablehlo::TransposeOp op) const override {
    Value operand = op.getOperand();
    auto const_op =
        dyn_cast_or_null<mlir::stablehlo::ConstantOp>(operand.getDefiningOp());
    if (!const_op) return failure();

    // Only support float tensors.
    auto tensor_type = mlir::dyn_cast_or_null<TensorType>(const_op.getType());
    if (!tensor_type || !tensor_type.getElementType().isF32()) {
      return failure();
    }

    return success(
        mlir::isa_and_nonnull<DenseFPElementsAttr>(const_op.getValue()));
  }

  void rewrite(mlir::stablehlo::TransposeOp op,
               PatternRewriter& rewriter) const override {
    auto const_op =
        cast<mlir::stablehlo::ConstantOp>(op.getOperand().getDefiningOp());

    const auto value_attr =
        mlir::cast<DenseFPElementsAttr>(const_op.getValue());
    const ArrayRef<int64_t> original_shape =
        value_attr.getShapedType().getShape();

    const SmallVector<float> original_values =
        llvm::to_vector(value_attr.getValues<float>());

    // Fold the constant value by transposing the values according to the
    // `TransposeOp`'s permutation attribute.
    const DenseElementsTransposer transposer(original_shape,
                                             op.getPermutation());
    SmallVector<float> transposed_values =
        transposer.TransposeValues(original_values);

    // Create a new constant op with the transposed values.
    const Location combined_loc =
        rewriter.getFusedLoc({const_op.getLoc(), op.getLoc()});
    auto new_value_type =
        RankedTensorType::getChecked(combined_loc, transposer.GetTargetShape(),
                                     /*elementType=*/rewriter.getF32Type());
    auto new_value_attr =
        DenseFPElementsAttr::get(new_value_type, std::move(transposed_values));
    auto new_const_op = rewriter.create<mlir::stablehlo::ConstantOp>(
        combined_loc, new_value_attr);

    rewriter.replaceAllUsesWith(op, new_const_op);
  };
};

}  // namespace

class FoldConstantTransposePass
    : public impl::FoldConstantTransposePassBase<FoldConstantTransposePass> {
 public:
  using impl::FoldConstantTransposePassBase<
      FoldConstantTransposePass>::FoldConstantTransposePassBase;

 private:
  void runOnOperation() override;
};

void FoldConstantTransposePass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<FoldTransposedConstantOp>(&ctx);
  if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
    func_op.emitError("Failed to fold constant->transpose pattern.");
    signalPassFailure();
  }
}

}  // namespace mlir::quant::stablehlo
