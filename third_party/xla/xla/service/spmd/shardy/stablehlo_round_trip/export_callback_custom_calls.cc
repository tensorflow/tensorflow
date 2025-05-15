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

#include "xla/service/spmd/shardy/stablehlo_round_trip/export_callback_custom_calls.h"

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::OperationPass;
using ::mlir::PassWrapper;
using ::mlir::StringRef;

using ::mlir::stablehlo::CustomCallOp;

// Attempts to replace the `CustomCallOp` with a tuple version of it, and a
// `GetTupleElementOp` that gets the first element of the tuple.
//
// This only happens if the op has a single result and the result type is not
// a tuple.
void replaceCallbackWithTupleVersion(CustomCallOp customCall) {
  if (customCall.getNumResults() != 1 ||
      mlir::isa<mlir::TupleType>(customCall->getResultTypes().front())) {
    return;
  }
  mlir::IRRewriter rewriter(customCall);
  CustomCallOp tupleCustomCall = cloneCustomCallWithNewResultTypes(
      customCall,
      mlir::TupleType::get(customCall->getContext(),
                           {customCall->getResultTypes()}),
      rewriter);
  auto getTupleElement = rewriter.create<mlir::stablehlo::GetTupleElementOp>(
      customCall.getLoc(), customCall->getResultTypes().front(),
      tupleCustomCall.getResult(0), rewriter.getI32IntegerAttr(0));
  getTupleElement->setAttr(kXlaShardingAttr,
                           customCall->getAttr(kXlaShardingAttr));
  rewriter.replaceOp(customCall, getTupleElement);
}

class StablehloRoundTripExportCallbackCustomCallsPass
    : public PassWrapper<StablehloRoundTripExportCallbackCustomCallsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      StablehloRoundTripExportCallbackCustomCallsPass)

  void runOnOperation() final {
    getOperation().walk([&](CustomCallOp customCall) {
      if (!isPythonCallbackCustomCall(customCall) || customCall->use_empty()) {
        return;
      }
      replaceCallbackWithTupleVersion(customCall);
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-stablehlo-round-trip-export-callback-custom-calls";
  }

  StringRef getDescription() const override {
    return "Converts the `CustomCallOp`s for host callbacks in XLA into the "
           "pattern that the XLA compiler recognizes.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
createStablehloRoundTripExportCallbackCustomCallsPass() {
  return std::make_unique<StablehloRoundTripExportCallbackCustomCallsPass>();
}

void registerStablehloRoundTripExportCallbackCustomCallsPass() {
  mlir::registerPass(createStablehloRoundTripExportCallbackCustomCallsPass);
}

}  // namespace sdy
}  // namespace xla
