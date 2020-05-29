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

// This file implements logic for testing buffer assignment including its
// utility converters.

#include "tensorflow/compiler/mlir/xla/transforms/buffer_assignment.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Function.h"                 // TF:llvm-project
#include "mlir/IR/Operation.h"                // TF:llvm-project
#include "mlir/Pass/Pass.h"                   // TF:llvm-project
#include "mlir/Pass/PassManager.h"            // TF:llvm-project
#include "absl/memory/memory.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace xla {
namespace {

/// This dialect independent unary operation has been defined only for testing
/// buffer assignment.
class BufferAssignmentTestUnaryOp
    : public Op<BufferAssignmentTestUnaryOp, OpTrait::OneResult,
                OpTrait::OneOperand> {
 public:
  using Op::Op;
  static StringRef getOperationName() { return "buffer_assignment_test.unary"; }
  static void build(OpBuilder& b, OperationState& state, Value source) {
    state.addOperands(source);
  }
};

/// This dialect independent lowered unary operation has been defined only for
/// testing buffer assignment.
class BufferAssignmentTestUnaryLoweredOp
    : public Op<BufferAssignmentTestUnaryLoweredOp, OpTrait::ZeroResult,
                OpTrait::NOperands<2>::Impl> {
 public:
  using Op::Op;
  static StringRef getOperationName() {
    return "buffer_assignment_test.unary_lowered";
  }
  static void build(OpBuilder& b, OperationState& state, Value source,
                    Value target) {
    state.addOperands(source);
    state.addOperands(target);
  }
};

/// This dialect independent copy operation has been defined only for testing
/// NonVoidToVoidReturnOpConverter
class BufferAssignmentTestCopyOp
    : public Op<BufferAssignmentTestCopyOp, OpTrait::ZeroResult,
                OpTrait::NOperands<2>::Impl> {
 public:
  using Op::Op;
  static StringRef getOperationName() { return "buffer_assignment_test.copy"; }
  static void build(OpBuilder& b, OperationState& state, Value from, Value to) {
    state.addOperands(from);
    state.addOperands(to);
  }
};

class BufferAssignmentTestDialect : public Dialect {
 public:
  explicit BufferAssignmentTestDialect(MLIRContext* context)
      : Dialect(getDialectNamespace(), context) {
    addOperations<BufferAssignmentTestCopyOp, BufferAssignmentTestUnaryOp,
                  BufferAssignmentTestUnaryLoweredOp>();
  }
  static StringRef getDialectNamespace() { return "buffer_assignment_test"; }
};

/// This pass tests two provided operation converters,
/// FunctionAndBlockSignatureConverter and NonVoidToVoidReturnOpConverter, for
/// Buffer Assignment.
struct BufferAssignmentPreparationTestPass
    : mlir::PassWrapper<BufferAssignmentPreparationTestPass, FunctionPass> {
  /// A simple converter that legalizes a BufferAssignmentTestUnaryOp to a
  /// BufferAssignmentTestUnaryLoweredOp and creates buffer allocation for
  /// the result of the computation.
  class TestUnaryOpConverter : public BufferAssignmentOpConversionPattern<
                                   BufferAssignmentTestUnaryOp> {
   public:
    using BufferAssignmentOpConversionPattern<
        BufferAssignmentTestUnaryOp>::BufferAssignmentOpConversionPattern;

    // Performs the actual legalization conversion step.
    LogicalResult matchAndRewrite(
        BufferAssignmentTestUnaryOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const final {
      // Create a new buffer allocation using the current BufferAssignmentPlacer
      // instance.
      auto result = op.getResult();
      auto result_type = result.getType().dyn_cast<ShapedType>();
      auto memref_type =
          MemRefType::get(result_type.getShape(), result_type.getElementType());
      rewriter.restoreInsertionPoint(
          bufferAssignment->computeAllocPosition(result));
      auto alloc = rewriter.create<AllocOp>(op.getLoc(), memref_type);

      // Create the lowered operation and replace the old operation with a
      // reference to the allocated buffer.
      rewriter.create<BufferAssignmentTestUnaryLoweredOp>(op.getLoc(),
                                                          operands[0], alloc);
      rewriter.replaceOp(op, {alloc});
      return success();
    }
  };

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto funcOp = getOperation();
    auto context = funcOp.getContext();
    ConversionTarget target(*context);
    BufferAssignmentPlacer bufferAssignmentPlacer(funcOp);

    // Specifying the legal and illegal operations.
    context->allowUnregisteredDialects(true);
    target.addIllegalOp<BufferAssignmentTestUnaryOp>();
    target.addLegalOp<BufferAssignmentTestUnaryLoweredOp>();
    target.addLegalOp<BufferAssignmentTestCopyOp>();
    target.addLegalOp<AllocOp>();
    target.addLegalOp<DeallocOp>();
    // TODO(dfki): ReturnOp can also be changed to TestReturnOp like
    // BufferAssignmentTestCopyOp.
    target.addDynamicallyLegalOp<ReturnOp>(
        [](ReturnOp returnOp) { return returnOp.getNumOperands() == 0; });
    FunctionAndBlockSignatureConverter::addDynamicallyLegalFuncOp(target);

    // Adding patterns for testing this pass.
    // clang-format off
    patterns.insert<
        FunctionAndBlockSignatureConverter,
        TestUnaryOpConverter,
        NonVoidToVoidReturnOpConverter
          <ReturnOp, ReturnOp, BufferAssignmentTestCopyOp>
    >(context, &bufferAssignmentPlacer);
    // clang-format on

    if (failed(applyPartialConversion(funcOp, target, patterns, nullptr))) {
      funcOp.emitOpError()
          << "Failed to apply buffer assignment preparation steps";
    }
  };
};

}  // namespace

static mlir::DialectRegistration<BufferAssignmentTestDialect>
    buffer_assignment_test_ops;

/// This pass tests helper methods such as computeAllocPosition,
/// FunctionAndBlockSignatureConverter, NonVoidToVoidReturnOpConverter
/// conversion patterns. Furthermore, it checks buffer-assignment pass that
/// moves existing Alloc and Dealloc operations to their proper positions, and
/// insert missing Dealloc operations.
static PassPipelineRegistration<> buffer_assignment_test_pass(
    "test-buffer-assignment",
    "Tests buffer assignment helper methods and buffer assignment pass.",
    [](mlir::OpPassManager& pm) {
      pm.addPass(absl::make_unique<BufferAssignmentPreparationTestPass>());
      pm.addPass(createBufferAssignmentPass());
    });

}  // namespace xla
}  // namespace mlir
