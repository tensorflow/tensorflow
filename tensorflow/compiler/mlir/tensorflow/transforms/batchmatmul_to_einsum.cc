/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/batchmatmul_to_einsum.h"

#include <climits>
#include <cstdint>
#include <numeric>

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/OpImplementation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/Functional.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

namespace {
// Replace TF BatchMatMul by TF Einsum
struct BatchMatMulToEinsumPass : public FunctionPass<BatchMatMulToEinsumPass> {
  void runOnFunction() override;
};

void BatchMatMulToEinsumPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  patterns.insert<ConvertTFBatchMatMulToEinsumOp<TF::BatchMatMulOp>,
                  ConvertTFBatchMatMulToEinsumOp<TF::BatchMatMulV2Op>>(
      &getContext());
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Transposes the last two dimensions of and conjugates tensor matrix
template <typename BatchMatMulOpType>
TF::TransposeOp
ConvertTFBatchMatMulToEinsumOp<BatchMatMulOpType>::createTransposeOp(
    Value value, Location loc, PatternRewriter& rewriter) {
  auto value_type = value.getType().cast<RankedTensorType>();
  auto shape = value_type.getShape();
  int dims = shape.size();

  std::vector<int32_t> perm(dims);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[dims - 1], perm[dims - 2]);

  auto perm_type = RankedTensorType::get({static_cast<int32_t>(perm.size())},
                                         rewriter.getIntegerType(32));

  auto perm_attr = DenseElementsAttr::get(perm_type, llvm::makeArrayRef(perm));
  auto perm_op = rewriter.create<TF::ConstOp>(loc, perm_type, perm_attr);

  std::vector<int64_t> transposed_shape(shape.begin(), shape.end());
  std::swap(transposed_shape[dims - 1], transposed_shape[dims - 2]);

  auto transposed_type =
      RankedTensorType::get(transposed_shape, value_type.getElementType());
  return rewriter.create<TF::TransposeOp>(loc, transposed_type, value, perm_op);
}

template <typename BatchMatMulOpType>
PatternMatchResult
ConvertTFBatchMatMulToEinsumOp<BatchMatMulOpType>::matchAndRewrite(
    BatchMatMulOpType op, PatternRewriter& rewriter) const {
  Value input_lhs = op.x();
  Value input_rhs = op.y();

  if (!input_lhs.getType().isa<RankedTensorType>()) {
    // LHS must be a ranked tensor type
    return this->matchFailure();
  }
  if (!input_rhs.getType().isa<RankedTensorType>()) {
    // RHS must be a ranked tensor type
    return this->matchFailure();
  }

  auto lhs_type = input_lhs.getType().cast<RankedTensorType>();
  auto rhs_type = input_rhs.getType().cast<RankedTensorType>();

  auto element_type = lhs_type.getElementType();

  if (element_type != rhs_type.getElementType()) {
    // The element type of LHS must be the same with element type of RHS
    return this->matchFailure();
  }

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();

  Location loc = op.getLoc();

  // Ensure that input ranks are at least 2.
  const int dims_a = lhs_shape.size();
  const int dims_b = rhs_shape.size();
  if (dims_a < 2 || dims_b < 2) {
    // Both inputs must have rank >= 2
    return this->matchFailure();
  }

  // Transpose LHS input if necessary.
  if (op.adj_x()) {
    input_lhs = createTransposeOp(input_lhs, loc, rewriter);

    lhs_type = input_lhs.getType().cast<RankedTensorType>();
    lhs_shape = lhs_type.getShape();
  }

  // Transpose RHS input if necessary.
  if (op.adj_y()) {
    input_rhs = createTransposeOp(input_rhs, loc, rewriter);

    rhs_type = input_rhs.getType().cast<RankedTensorType>();
    rhs_shape = rhs_type.getShape();
  }

  if (lhs_shape[dims_a - 1] != rhs_shape[dims_b - 2]) {
    // Input dimensions must be compatible for multiplication.
    return this->matchFailure();
  }

  // einsum equation for batchmatmul
  std::string equation("...mk,...kn->...mn");

  std::vector<Value> inputs = {input_lhs, input_rhs};
  rewriter.replaceOpWithNewOp<TF::EinsumOp>(op, op.getType(),
                                            /*inputs=*/ValueRange(inputs),
                                            /*equation=*/equation);

  return this->matchSuccess();
}

static PassRegistration<BatchMatMulToEinsumPass> pass(
    "tf-batch-matmul-to-tf-einsum",
    "Replace TF BatchMatMul op by TF Einsum op.");

std::unique_ptr<OpPassBase<FuncOp>> CreateBatchMatMulToEinsumPass() {
  return std::make_unique<BatchMatMulToEinsumPass>();
}

}  // namespace TF
}  // namespace mlir
