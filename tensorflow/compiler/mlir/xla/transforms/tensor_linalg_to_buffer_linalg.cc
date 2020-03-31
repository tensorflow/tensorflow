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

// This file implements logic for transforming Linalg operations with tensor
// types to memref type and allocate and deallocate buffers using the
// BufferAssignmentPlacer.

#include "absl/memory/memory.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"    // TF:llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"     // TF:llvm-project
#include "mlir/IR/AffineExpr.h"                  // TF:llvm-project
#include "mlir/IR/Attributes.h"                  // TF:llvm-project
#include "mlir/IR/Builders.h"                    // TF:llvm-project
#include "mlir/IR/Function.h"                    // TF:llvm-project
#include "mlir/IR/Location.h"                    // TF:llvm-project
#include "mlir/IR/MLIRContext.h"                 // TF:llvm-project
#include "mlir/IR/Operation.h"                   // TF:llvm-project
#include "mlir/IR/PatternMatch.h"                // TF:llvm-project
#include "mlir/IR/StandardTypes.h"               // TF:llvm-project
#include "mlir/Pass/Pass.h"                      // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"   // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/buffer_assignment.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla {
namespace {

class GenericOpConverter
    : public xla::BufferAssignmentOpConversionPattern<linalg::GenericOp> {
 public:
  using xla::BufferAssignmentOpConversionPattern<
      linalg::GenericOp>::BufferAssignmentOpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    SmallVector<Value, 4> args(operands.begin(), operands.end());

    // Update all types to memref types.
    auto result = op.getOperation()->getResult(0);
    auto type = result.getType().cast<ShapedType>();
    auto memref_type = MemRefType::get(type.getShape(), type.getElementType());
    auto position = bufferAssignment->computeAllocPosition(result);

    // Compute alloc position and insert a custom allocation node.
    OpBuilder allocBuilder(result.getDefiningOp());
    allocBuilder.restoreInsertionPoint(position);
    auto alloc = allocBuilder.create<AllocOp>(loc, memref_type);
    args.push_back(alloc);

    // Generate a new linalg operation that works on buffers.
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, llvm::None, args, rewriter.getI64IntegerAttr(operands.size()),
        rewriter.getI64IntegerAttr(1), op.indexing_maps(), op.iterator_types(),
        /*doc=*/nullptr,
        /*fun=*/nullptr, /*library_call=*/nullptr);

    // Move regions from the old operation to the new one.
    auto& region = linalgOp.region();
    rewriter.inlineRegionBefore(op.region(), region, region.end());

    // TODO(dfki): verify the internal memref-based linalg functionality.
    region.front().addArgument(type.getElementType());

    // Replace the old linalg version with the new allocation.
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

void populateTensorLinalgToBufferLinalgConversionPattern(
    MLIRContext* context, xla::BufferAssignmentPlacer* placer,
    OwningRewritePatternList* patterns) {
  patterns->insert<xla::FunctionAndBlockSignatureConverter, GenericOpConverter,
                   xla::NonVoidToVoidReturnOpConverter<
                       mlir::ReturnOp, mlir::ReturnOp, linalg::CopyOp>>(context,
                                                                        placer);
}

struct TensorLinalgToBufferLinalg
    : public FunctionPass<TensorLinalgToBufferLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto& context = getContext();
    ConversionTarget target(context);

    // Make all linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [](Operation* op) {
      auto isIllegalValue = [](Value operand) {
        return operand.getType().isa<TensorType>();
      };
      auto operands = op->getOperands();
      auto results = op->getResults();
      return std::none_of(operands.begin(), operands.end(), isIllegalValue) &
             std::none_of(results.begin(), results.end(), isIllegalValue);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            isLegalOperation));

    // Mark return operations illegal as long as they return values.
    target.addDynamicallyLegalOp<mlir::ReturnOp>(
        [](mlir::ReturnOp returnOp) { return returnOp.getNumOperands() == 0; });

    auto function = getFunction();
    xla::BufferAssignmentPlacer placer(function);
    xla::FunctionAndBlockSignatureConverter::addDynamicallyLegalFuncOp(target);
    populateTensorLinalgToBufferLinalgConversionPattern(function.getContext(),
                                                        &placer, &patterns);

    // Do partial conversion so we can have unknown ops in tests.
    if (failed(applyPartialConversion(function, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};
}  // namespace

static PassRegistration<TensorLinalgToBufferLinalg>
    tensor_linalg_to_buffer_linalg_pass(
        "tensor-linalg-to-buffer-linalg",
        "Legalize linalg operations with tensor type operands to memref type "
        "ones");

}  // namespace xla
}  // namespace mlir
