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

#include "xla/service/spmd/shardy/round_trip_common/import_backend_func_calls.h"

#include <iterator>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::MLIRContext;
using ::mlir::OpConversionPattern;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::NamedComputationOp;

class BackendFuncCallPattern : public OpConversionPattern<CallOp> {
 public:
  explicit BackendFuncCallPattern(MLIRContext* context,
                                  const SymbolTable& symbolTable)
      : OpConversionPattern<CallOp>(context), symbolTable(symbolTable) {}

  mlir::LogicalResult matchAndRewrite(
      CallOp callOp, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!hasFrontendAttr(callOp, kXlaBackendConfigAttr)) {
      return mlir::failure();
    }

    FuncOp func = symbolTable.lookup<FuncOp>(adaptor.getCallee());
    CHECK(func) << "Failed to lookup function: "
                << absl::string_view(adaptor.getCallee());
    mlir::SmallVector<mlir::NamedAttribute> namedCompAttrs;
    llvm::copy_if(callOp->getDiscardableAttrs(),
                  std::back_inserter(namedCompAttrs),
                  [](const mlir::NamedAttribute& attr) {
                    return attr.getName() != kShardingAttr;
                  });

    auto namedCompOp = rewriter.replaceOpWithNewOp<NamedComputationOp>(
        callOp, callOp->getResultTypes(), adaptor.getCallee(),
        adaptor.getOperands(), /*inShardings=*/nullptr,
        /*outShardings=*/mlir::sdy::getShardingPerValue(callOp));
    namedCompOp->setAttrs(namedCompAttrs);
    if (func.getBody().empty()) {
      return rewriter.notifyMatchFailure(callOp, [](mlir::Diagnostic& diag) {
        diag << "Tried to use an already inlined FuncOp. Expected each CallOp "
                "with backend_config to have a unique FuncOp.";
      });
    }

    mlir::sdy::inlineRegionAndConvertTerminatorOp<mlir::sdy::ReturnOp>(
        func.getBody(), namedCompOp.getRegion(), rewriter);
    rewriter.eraseOp(func);

    return mlir::success();
  }

 private:
  const SymbolTable& symbolTable;
};

// Converts a `CallOp` with `backend_config` into a `NamedComputationOp`.
class ImportBackendFuncCallsPass
    : public mlir::PassWrapper<ImportBackendFuncCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportBackendFuncCallsPass)

  void runOnOperation() final {
    // NOTE: Assume that there is a unique callee for each caller. So no need to
    // do a walk and copy the callees if there are multiple callers for the
    // callee.
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addLegalOp<NamedComputationOp, mlir::sdy::ReturnOp>();
    SymbolTable symbolTable(getOperation());
    target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
      // In case the assumption that each host-callback caller has a unique
      // callee is not true, and an optimized build is being run without
      // verification, make sure that the callee is a function that exists.
      return !hasFrontendAttr(op, kXlaBackendConfigAttr);
    });
    mlir::RewritePatternSet patterns(&context);
    patterns.add<BackendFuncCallPattern>(&context, symbolTable);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-import-backend-func-calls";
  }

  StringRef getDescription() const override {
    return "Creates a pass that converts a `CallOp` with a `backend_config` "
           "attr to a `NamedComputationOp` with the function body inlined and "
           "name of the callee.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createImportBackendFuncCallsPass() {
  return std::make_unique<ImportBackendFuncCallsPass>();
}

void registerImportBackendFuncCallsPass() {
  mlir::registerPass(createImportBackendFuncCallsPass);
}

}  // namespace sdy
}  // namespace xla
