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
#include "mlir/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

namespace {
// Replace TF BatchMatMul by TF Einsum
struct BatchMatMulToEinsumPass
    : public PassWrapper<BatchMatMulToEinsumPass, FunctionPass> {
  void runOnFunction() override;
};

void BatchMatMulToEinsumPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  patterns.insert<ConvertTFBatchMatMulToEinsumOp<TF::BatchMatMulOp>,
                  ConvertTFBatchMatMulToEinsumOp<TF::BatchMatMulV2Op>>(
      &getContext());
  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

template <typename BatchMatMulOpType>
LogicalResult
ConvertTFBatchMatMulToEinsumOp<BatchMatMulOpType>::matchAndRewrite(
    BatchMatMulOpType op, PatternRewriter& rewriter) const {
  Value input_lhs = op.x();
  Value input_rhs = op.y();

  if (!input_lhs.getType().isa<RankedTensorType>()) {
    // LHS must be a ranked tensor type
    return failure();
  }
  if (!input_rhs.getType().isa<RankedTensorType>()) {
    // RHS must be a ranked tensor type
    return failure();
  }

  auto lhs_type = input_lhs.getType().dyn_cast<RankedTensorType>();
  auto rhs_type = input_rhs.getType().dyn_cast<RankedTensorType>();

  if (!lhs_type || !rhs_type) {
    return failure();
  }

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();

  Location loc = op.getLoc();

  // Ensure that input ranks are at least 2.
  const int dims_a = lhs_shape.size();
  const int dims_b = rhs_shape.size();
  if (dims_a < 2 || dims_b < 2) {
    // Both inputs must have rank >= 2
    return failure();
  }

  // einsum equation for batchmatmul
  std::string equation("...mk,...kn->...mn");

  if (op.adj_x()) {
    std::swap(equation[3], equation[4]);
  }
  if (op.adj_y()) {
    std::swap(equation[6 + 3], equation[6 + 4]);
  }

  llvm::SmallVector<Value, 2> inputs = {input_lhs, input_rhs};
  rewriter.replaceOpWithNewOp<TF::EinsumOp>(op, op.getType(),
                                            /*inputs=*/ValueRange(inputs),
                                            /*equation=*/equation);

  return success();
}

static PassRegistration<BatchMatMulToEinsumPass> pass(
    "tf-batch-matmul-to-tf-einsum",
    "Replace TF BatchMatMul op by TF Einsum op.");

std::unique_ptr<OperationPass<FuncOp>> CreateBatchMatMulToEinsumPass() {
  return std::make_unique<BatchMatMulToEinsumPass>();
}

}  // namespace TF
}  // namespace mlir
