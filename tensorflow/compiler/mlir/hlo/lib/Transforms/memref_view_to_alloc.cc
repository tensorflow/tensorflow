/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Transforms/passes.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class MemRefViewToAllocOp : public OpConversionPattern<memref::ViewOp> {
public:
  using OpConversionPattern<memref::ViewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ViewOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    memref::ViewOpAdaptor adaptor(operands, op->getAttrDictionary());
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, getTypeConverter()->convertType(op.getType()).cast<MemRefType>(),
        adaptor.sizes());
    return success();
  }
};

/// A helper type converter class that registers a type to type conversion.
class MemRefViewTypeConverter : public TypeConverter {
public:
  MemRefViewTypeConverter() {
    addConversion([](Type type) { return type; });
  }
};
} // end anonymous namespace

namespace {
/// Pass to convert memref views to allocates.
struct MemRefViewToAllocatePass
    : public MemRefViewToAllocateBase<MemRefViewToAllocatePass> {
  void runOnOperation() override {
    // Parse memref.views to allocates
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    MemRefViewTypeConverter typeConverter;

    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalOp<memref::ViewOp>();

    RewritePatternSet patterns(&context);
    patterns.add<MemRefViewToAllocOp>(typeConverter, patterns.getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

/// Create a pass to transform memref views to allocates.
std::unique_ptr<OperationPass<FuncOp>> mlir::createMemRefViewToAllocatePass() {
  return std::make_unique<MemRefViewToAllocatePass>();
}
