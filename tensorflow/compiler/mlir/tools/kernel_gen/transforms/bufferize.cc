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

// This file implements logic for translating mixed IR to buffer form.

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"
#include "tensorflow/compiler/xla/mlir_hlo/stablehlo/stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

struct BufferizeJITExecuteOp
    : public OpConversionPattern<tf_framework::JITExecuteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tf_framework::JITExecuteOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type result_ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tf_framework::JITExecuteOp>(
        op, result_ty, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

}  // namespace

void populateExtraBufferizePatterns(
    ConversionTarget &target, MLIRContext *context,
    bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns) {
  target.addLegalDialect<tf_framework::TFFrameworkDialect>();
  auto typesAreLegal = [converter](Operation *op) {
    return converter->isLegal(op->getOperandTypes()) &&
           converter->isLegal(op->getResultTypes());
  };
  target.addDynamicallyLegalOp<tf_framework::JITExecuteOp>(typesAreLegal);
  // clang-format off
  patterns->add<
      BufferizeJITExecuteOp
  >(*converter, context);
  // clang-format on
}

void populateExtraBufferizeDialects(DialectRegistry &registry) {
  registry.insert<tf_framework::TFFrameworkDialect>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
