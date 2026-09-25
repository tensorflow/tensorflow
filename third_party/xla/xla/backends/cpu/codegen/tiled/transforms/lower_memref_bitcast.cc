/* Copyright 2026 The OpenXLA Authors.

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

#include <utility>

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DEF_LOWERMEMREFBITCASTPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

struct LowerMemRefBitcast
    : public mlir::ConvertOpToLLVMPattern<xtile::MemRefBitcastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::MemRefBitcastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSource());
    return mlir::success();
  }
};

class LowerMemRefBitcastPass
    : public impl::LowerMemRefBitcastPassBase<LowerMemRefBitcastPass> {
 public:
  using LowerMemRefBitcastPassBase::LowerMemRefBitcastPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::LowerToLLVMOptions options(context, mlir::DataLayout(module));
    mlir::LLVMTypeConverter type_converter(context, options);

    mlir::ConversionTarget target(*context);
    target.addIllegalOp<xtile::MemRefBitcastOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerMemRefBitcast>(type_converter);
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace xla::cpu
