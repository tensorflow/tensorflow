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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_

#include "mlir/Analysis/Dominance.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Builders.h"   // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project

namespace mlir {
namespace xla {

/// Prepares a buffer assignment phase. It can place (user-defined) alloc
/// nodes. This simplifies the integration of the actual buffer-assignment
/// pass. Sample usage:
///   BufferAssignmentPlacer baHelper(regionOp);
///   -> determine alloc positions
///   auto allocPosition = baHelper.computeAllocPosition(value);
///   -> place alloc
///   allocBuilder.setInsertionPoint(positions.getAllocPosition());
///   <create alloc>
///   alternatively:
///   -> place alloc
///   baHelper.insertAlloc<AllocOp>(...);
/// Note: this class is intended to be used during legalization. In order
/// to move alloc and dealloc nodes into the right places you can use the
/// createBufferAssignmentPass() function.
class BufferAssignmentPlacer {
 public:
  /// Creates a new assignment builder.
  explicit BufferAssignmentPlacer(Operation* op);

  /// Returns the operation this analysis was constructed from.
  Operation* getOperation() const { return operation; }

  /// Computes the actual position to place allocs for the given value.
  OpBuilder::InsertPoint computeAllocPosition(Value value);

 private:
  /// The operation this analysis was constructed from.
  Operation* operation;

  /// The dominator analysis to place allocs in the appropriate blocks.
  DominanceInfo dominators;
};

/// Helper conversion pattern that encapsulates a BufferAssignmentPlacer
/// instance.
template <typename SourceOp>
class BufferAssignmentOpConversionPattern
    : public OpConversionPattern<SourceOp> {
 public:
  explicit BufferAssignmentOpConversionPattern(
      MLIRContext* context_,
      xla::BufferAssignmentPlacer* bufferAssignment_ = nullptr,
      PatternBenefit benefit_ = 1)
      : OpConversionPattern<SourceOp>(context_, benefit_),
        bufferAssignment(bufferAssignment_) {}

 protected:
  xla::BufferAssignmentPlacer* bufferAssignment;
};

// Converts only the tensor-type function and block arguments to memref-type.
class FunctionAndBlockSignatureConverter
    : public BufferAssignmentOpConversionPattern<FuncOp> {
 public:
  using BufferAssignmentOpConversionPattern<
      FuncOp>::BufferAssignmentOpConversionPattern;

  // Adding functions whose arguments are memref type to the set of legal
  // operations.
  static void addDynamicallyLegalFuncOp(ConversionTarget& target);

  // Performs the actual signature rewriting step.
  LogicalResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final;
};

// This pattern converter transforms a non-void ReturnOpSourceTy into a void
// return of type ReturnOpTargetTy. It uses a copy operation of type CopyOpTy to
// copy the results to the output buffer.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy>
class NonVoidToVoidReturnOpConverter
    : public BufferAssignmentOpConversionPattern<ReturnOpSourceTy> {
 public:
  using BufferAssignmentOpConversionPattern<
      ReturnOpSourceTy>::BufferAssignmentOpConversionPattern;

  // Performs the actual return-op conversion step.
  LogicalResult matchAndRewrite(
      ReturnOpSourceTy returnOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto numReturnValues = returnOp.getNumOperands();
    auto funcOp = returnOp.template getParentOfType<FuncOp>();
    auto numFuncArgs = funcOp.getNumArguments();
    auto loc = returnOp.getLoc();

    // Find the corresponding output buffer for each operand.
    for (auto operand : llvm::enumerate(operands)) {
      auto returnArgNumber = numFuncArgs - numReturnValues + operand.index();
      auto dstBuffer = funcOp.getArgument(returnArgNumber);
      if (dstBuffer == operand.value()) {
        continue;
      }

      // Insert the copy operation to copy before the return.
      rewriter.setInsertionPoint(
          returnOp.getOperation()->getBlock()->getTerminator());
      rewriter.create<CopyOpTy>(loc, operand.value(),
                                funcOp.getArgument(returnArgNumber));
    }
    // Insert the new target return operation.
    rewriter.replaceOpWithNewOp<ReturnOpTargetTy>(returnOp);
    return success();
  }
};

}  // namespace xla
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_
