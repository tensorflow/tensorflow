/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/round_trip_common/import_constants.h"

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mhlo = ::mlir::mhlo;

namespace xla {
namespace sdy {

namespace {

using ::mlir::ConversionPatternRewriter;
using ::mlir::LogicalResult;
using ::mlir::OpConversionPattern;
using ::mlir::StringRef;
using ::mlir::func::FuncOp;

using ::mlir::sdy::ConstantOp;

class ConstantPattern : public OpConversionPattern<mhlo::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // We use the generic op builder so that unregistered attributes will be
    // added to the new op.
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, op->getResultTypes(), adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

class ImportConstantsPass
    : public mlir::PassWrapper<ImportConstantsPass,
                               mlir::OperationPass<FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportConstantsPass)

  void runOnOperation() final {
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addIllegalOp<mhlo::ConstantOp>();
    target.addLegalOp<ConstantOp>();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<ConstantPattern>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "xla-sdy-import-constants"; }

  StringRef getDescription() const override {
    return "Converts an `mhlo.constant` into an `sdy.constant`.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createImportConstantsPass() {
  return std::make_unique<ImportConstantsPass>();
}

void registerImportConstantsPass() {
  mlir::registerPass(createImportConstantsPass);
}

}  // namespace sdy
}  // namespace xla
