/* Copyright 2022 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct StablehloLegalizeToHloPass
    : public impl::StablehloLegalizeToHloPassBase<StablehloLegalizeToHloPass> {
  using StablehloLegalizeToHloPassBase::StablehloLegalizeToHloPassBase;
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();

    // Allow injecting legal ops to permit gradual migration.
    if (!convert_xla_supported_stablehlo_) {
      target.addLegalOp<stablehlo::ConstantOp>();
    }

    stablehlo::StablehloToHloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateStablehloToHloPatterns(&patterns, &converter,
                                              &getContext());
    stablehlo::registerFuncOpsForTypeConversion(target, patterns, converter);

    // Our guiding principle is to support all StableHLO functionality in MHLO.
    // This check is here only for exceptional situations, e.g. when we added
    // a new StableHLO op and forgot to update the conversion patterns.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

}  // namespace mhlo
}  // namespace mlir
