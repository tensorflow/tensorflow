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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_HLOLEGALIZETOSTABLEHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

bool hasMhloTypes(TypeRange types) {
  bool hasMhloType = false;
  for (Type type : types) {
    type.walk([&](Type t) {
      if (auto tuple = dyn_cast<TupleType>(t)) {
        hasMhloType = hasMhloType || hasMhloTypes(tuple.getTypes());
      } else if (auto bundle = dyn_cast<mhlo::AsyncBundleType>(t)) {
        hasMhloType = hasMhloType || hasMhloTypes(bundle.getTypes());
      } else if (auto rankedTensor = dyn_cast<RankedTensorType>(t)) {
        hasMhloType =
            hasMhloType || llvm::isa_and_nonnull<mhlo::TypeExtensionsAttr>(
                               rankedTensor.getEncoding());
      } else if (llvm::isa<mhlo::MhloDialect>(t.getDialect())) {
        hasMhloType = true;
      }
      if (hasMhloType) return WalkResult::interrupt();
      return WalkResult::advance();
    });
  }
  return hasMhloType;
}
struct UpdateOperandsInUnknownOp : public ConversionPattern {
  UpdateOperandsInUnknownOp(TypeConverter& converter, MLIRContext* context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}
  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // Input types already converted to MHLO.
    if (llvm::isa<mhlo::MhloDialect, stablehlo::StablehloDialect>(
            op->getDialect()))
      return rewriter.notifyMatchFailure(op, "op is not an unknown op");

    if (!hasMhloTypes(op->getOperandTypes()))
      return rewriter.notifyMatchFailure(op, "op has no mhlo operands");

    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(operands); });
    return success();
  }
};

struct HloLegalizeToStablehloPass
    : public impl::HloLegalizeToStablehloPassBase<HloLegalizeToStablehloPass> {
  HloLegalizeToStablehloPass()
      : HloLegalizeToStablehloPassBase<HloLegalizeToStablehloPass>() {}
  explicit HloLegalizeToStablehloPass(
      const HloLegalizeToStablehloPassOptions& opts)
      : HloLegalizeToStablehloPassBase<HloLegalizeToStablehloPass>(opts) {}

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();

    stablehlo::HloToStablehloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateHloToStablehloPatterns(
        &patterns, &converter, &getContext(), allow_experimental_features_,
        allow_xla_features_);
    stablehlo::registerFuncOpsForTypeConversion(target, patterns, converter);

    if (allow_xla_features_) {
      // These ops do not exist in StableHLO. (They do exist in CHLO, a slightly
      // higher-level dialect wrapping StableHLO, but we leave them as MHLO here
      // since we're specifically legalizing to StableHLO, not to CHLO.)
      target.addLegalOp<  //
          mhlo::AcosOp, mhlo::AcoshOp, mhlo::AsinOp, mhlo::AsinhOp,
          mhlo::AtanhOp, mhlo::CoshOp, mhlo::ErfOp, mhlo::RaggedDotOp,
          mhlo::ScanOp, mhlo::SinhOp, mhlo::TopKOp>();

      // These ops do not exist in StableHLO. (They don't exist in CHLO, either;
      // MHLO is the appropriate dialect for expressing XLA-specific features
      // such as these.)
      target.addLegalOp<
          mhlo::AsyncDoneOp, mhlo::AsyncStartOp, mhlo::AsyncUpdateOp,
          mhlo::BitcastOp, mhlo::CopyOp, mhlo::DomainOp, mhlo::FusionOp,
          mhlo::MinimumBroadcastShapesOp, mhlo::StochasticConvertOp,
          mhlo::TraceOp, mhlo::XlaRngGetAndUpdateStateOp>();

      target.addDynamicallyLegalOp<mhlo::AddDependencyOp>(
          [](mhlo::AddDependencyOp op) {
            return !hasMhloTypes(op->getOperandTypes());
          });
      target.addDynamicallyLegalOp<mhlo::AsyncStartOp>(
          [](mhlo::AsyncStartOp op) {
            return !hasMhloTypes(op->getResultTypes());
          });
      target.addDynamicallyLegalOp<mhlo::AsyncUpdateOp>(
          [](mhlo::AsyncUpdateOp op) {
            return !hasMhloTypes(op->getResultTypes());
          });
      target.addDynamicallyLegalOp<mhlo::AsyncDoneOp>([](mhlo::AsyncDoneOp op) {
        return !hasMhloTypes(op->getResultTypes());
      });
      target.addDynamicallyLegalOp<mhlo::CustomCallOp>(
          [](mhlo::CustomCallOp op) {
            return !!op.getCustomCallScheduleAttr();
          });
      // TODO: StableHLO AllToAll has different semantics than MHLO AllToAll.
      target.addDynamicallyLegalOp<mhlo::AllToAllOp>(
          [](mhlo::AllToAllOp op) { return op.getNumOperands() > 1; });
    }

    // Handle non-MHLO ops that may have bounded dynamism or token types.
    target.markUnknownOpDynamicallyLegal(
        [](Operation* op) { return !hasMhloTypes(op->getOperandTypes()); });
    patterns.add<UpdateOperandsInUnknownOp>(converter, &getContext());

    ConversionConfig config;
    config.foldingMode = DialectConversionFoldingMode::Never;

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns), config)))
      return signalPassFailure();
  }
};

}  // namespace

}  // namespace mhlo
}  // namespace mlir
