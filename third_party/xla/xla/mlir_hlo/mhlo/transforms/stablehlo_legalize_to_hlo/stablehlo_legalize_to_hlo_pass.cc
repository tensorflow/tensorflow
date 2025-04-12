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
#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

// AddDependencyOp is the only op that doesn't exist in StableHLO but uses
// token types. This led to two options (1) support either token type in
// AddDependencyOp or (2) Design a token conversion (or unrealized cast) between
// MHLO and StableHLO. Option (1) seems safer, and we can hopefully obsolete
// mhlo::TokenType all together and just use StableHLO tokens everywhere.
//
// Note: Only the second argument needs to be converted. All token creation and
// propagation is already handled by existing conversions.
struct AddDependencyOpToMhoTokenConverter
    : public OpConversionPattern<mhlo::AddDependencyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::AddDependencyOp op, mhlo::AddDependencyOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only convert if input token type is MHLO token
    if (!llvm::isa<mhlo::TokenType>(adaptor.getToken().getType()))
      return rewriter.notifyMatchFailure(op, "nothing to convert");
    rewriter.replaceOpWithNewOp<mhlo::AddDependencyOp>(op, adaptor.getOperand(),
                                                       adaptor.getToken());
    return success();
  }
};

void legalDirectStablehloToHloConversionOps(ConversionTarget& target) {
  target.addLegalOp<
      // go/keep-sorted start
      stablehlo::AbsOp, stablehlo::CbrtOp, stablehlo::SqrtOp, stablehlo::TanOp,
      stablehlo::AddOp, stablehlo::BroadcastInDimOp, stablehlo::BroadcastOp,
      stablehlo::CeilOp, stablehlo::ClzOp, stablehlo::ConvertOp,
      stablehlo::ConstantOp, stablehlo::ConvolutionOp, stablehlo::CosineOp,
      stablehlo::DynamicSliceOp, stablehlo::FloorOp, stablehlo::ImagOp,
      stablehlo::ExpOp, stablehlo::Expm1Op, stablehlo::DynamicBroadcastInDimOp,
      stablehlo::IsFiniteOp, stablehlo::Log1pOp, stablehlo::LogOp,
      stablehlo::LogisticOp, stablehlo::NegOp, stablehlo::NotOp,
      stablehlo::PopulationCountOp, stablehlo::RealOp,
      stablehlo::RoundNearestEvenOp, stablehlo::RoundOp, stablehlo::RsqrtOp,
      stablehlo::SignOp, stablehlo::SineOp, stablehlo::SliceOp,
      stablehlo::TanhOp
      // go/keep-sorted end
      >();
}

struct StablehloLegalizeToHloPass
    : public impl::StablehloLegalizeToHloPassBase<StablehloLegalizeToHloPass> {
  using StablehloLegalizeToHloPassBase::StablehloLegalizeToHloPassBase;
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addDynamicallyLegalOp<mhlo::AddDependencyOp>(
        [](mhlo::AddDependencyOp op) {
          return llvm::isa<mhlo::TokenType>(op.getToken().getType());
        });

    // Allow injecting legal ops to permit gradual migration.
    if (!convert_xla_supported_stablehlo_) {
      legalDirectStablehloToHloConversionOps(target);
    }

    stablehlo::StablehloToHloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    patterns.add<AddDependencyOpToMhoTokenConverter>(&getContext());
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
