/* Copyright 2022 The StableHLO Authors.
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

#include <optional>
#include <string>
#include <utility>

#include "llvm/Support/Debug.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/api/PortableApi.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloTypes.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/lite/core/macros.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace odml {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOTOVHLOPASS
#define GEN_PASS_DEF_LEGALIZEVHLOTOSTABLEHLOPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// StableHLO --> VHLO types
//===----------------------------------------------------------------------===//

std::optional<Value> MaterializeIllegalCast(OpBuilder &builder, Type type,
                                            ValueRange inputs, Location loc) {
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
      ->getResult(0);
}

class StablehloToOdmlTypeConverter : public vhlo::VhloTypeConverter {
 public:
  StablehloToOdmlTypeConverter() : vhlo::VhloTypeConverter() {
    addConversion([](Type type) {
      if (type.getDialect().getNamespace() ==
          vhlo::VhloDialect::getDialectNamespace()) {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return Type();
    });
    addConversion([](stablehlo::TokenType token) {
      return vhlo::TokenV1Type::get(token.getContext());
    });
    addBuiltinToVhloConversions();

    addArgumentMaterialization(MaterializeIllegalCast);
    addSourceMaterialization(MaterializeIllegalCast);
    addTargetMaterialization(MaterializeIllegalCast);
  }

  Attribute convertEncoding(Attribute attr) const final {
    LLVM_DEBUG(llvm::dbgs() << "Converting encoding.\n" << attr << '\n');
    // Must be VHLO encoding, or convertible to VHLO encoding.
    if (attr.getDialect().getNamespace() ==
        vhlo::VhloDialect::getDialectNamespace())
      return attr;

    if (auto stablehlo_attr =
            mlir::dyn_cast_or_null<stablehlo::TypeExtensionsAttr>(attr)) {
      return vhlo::TypeExtensionsV1Attr::get(stablehlo_attr.getContext(),
                                             stablehlo_attr.getBounds());
    }

    // Was not VHLO encoding, or convertible.
    return {};
  }
};

class VhloToStablehloTypeConverter : public vhlo::VhloTypeConverter {
 public:
  VhloToStablehloTypeConverter() : vhlo::VhloTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](vhlo::TokenV1Type token) {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return stablehlo::TokenType::get(token.getContext());
    });
    addVhloToBuiltinConversions();

    addArgumentMaterialization(MaterializeIllegalCast);
    addSourceMaterialization(MaterializeIllegalCast);
    addTargetMaterialization(MaterializeIllegalCast);
  }

  Attribute convertEncoding(Attribute attr) const final {
    if (auto vhlo_attr =
            mlir::dyn_cast_or_null<vhlo::TypeExtensionsV1Attr>(attr)) {
      return stablehlo::TypeExtensionsAttr::get(vhlo_attr.getContext(),
                                                vhlo_attr.getBounds());
    }
    // All encodings supported in StableHLO.
    return attr;
  }
};

//===----------------------------------------------------------------------===//
// StableHLO+TFL --> VHLO+TFL Ops
//===----------------------------------------------------------------------===//

// Wrap op result uses in an unrealized cast to create a cast to buffer
// any type changes to result, and apply type converter to result:
//   result = op(V0)
//   V1     = op2(result)
//   ==>
//   result = op(V0)
//   V1     = unrealized_cast(result)
//   V2     = op2(V1)
void ConvertAndWrapUsesInUnrealizedCast(Value result, TypeConverter &converter,
                                        IRRewriter &rewriter) {
  auto type = result.getType();
  result.setType(converter.convertType(result.getType()));
  auto new_value = converter.materializeArgumentConversion(
      rewriter, result.getLoc(), type, {result});
  rewriter.replaceAllUsesExcept(result, new_value, new_value.getDefiningOp());
}

// Wrap operands in an an unrealized cast to create a cast to buffer any type
// changes to the operand, and apply type converter to operands:
//   V0 = op(operand)
//   ==>
//   V0 = unrealized_cast(operand)
//   V1 = op(V0)
void WrapOperandsInUnrealizedCastAndConvert(Operation *op,
                                            TypeConverter &converter,
                                            IRRewriter &rewriter) {
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto operand = op->getOperand(i);
    auto new_operand = converter.materializeArgumentConversion(
        rewriter, op->getLoc(), converter.convertType(operand.getType()),
        {operand});
    op->setOperand(i, new_operand);
  }
}

// vhlo.op %1 : vhlo.tensor<...>
//   ==>
// vhlo.op %1 : tensor<...>
//
// TODO: There's likely a way to make MLIR manage the unrealized cast
// conversions using a specific rewriter.
LogicalResult ApplyTypeConverter(ModuleOp op, TypeConverter &converter) {
  IRRewriter rewriter(op->getContext());

  op->walk([&](Operation *op) {
    if (op->getDialect()->getNamespace() != "vhlo") return;

    // Convert operands
    rewriter.modifyOpInPlace(op, [&]() {
      rewriter.setInsertionPoint(op);
      WrapOperandsInUnrealizedCastAndConvert(op, converter, rewriter);

      // Convert op types
      for (auto value : op->getResults()) {
        rewriter.setInsertionPointAfter(value.getDefiningOp());
        ConvertAndWrapUsesInUnrealizedCast(value, converter, rewriter);
      }

      // Convert block arguments
      for (auto &region : op->getRegions()) {
        for (auto &block : region.getBlocks()) {
          rewriter.setInsertionPointToStart(&block);
          for (auto arg : block.getArguments()) {
            ConvertAndWrapUsesInUnrealizedCast(arg, converter, rewriter);
          }
        }
      }
    });
  });
  return success();
}

// Legalize StableHLO portion of program to VHLO, leaves TFL untouched
LogicalResult ApplyStablehloToVhloPatterns(ModuleOp module,
                                           bool is_func_legal) {
  MLIRContext *context = module.getContext();
  ConversionTarget target(*context);
  target.addIllegalDialect<stablehlo::StablehloDialect>();
  target.addDynamicallyLegalDialect<func::FuncDialect>(
      [&](auto) { return is_func_legal; });
  target.addLegalDialect<TFL::TensorFlowLiteDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<vhlo::VhloDialect>();

  StablehloToOdmlTypeConverter converter;
  RewritePatternSet patterns(context);
  stablehlo::populateStablehloToVhloPatterns(&patterns, &converter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    return module->emitError("Failed partial conversion to VHLO");
  }
  return success();
}

LogicalResult ApplyVhloToVersionPatterns(ModuleOp module,
                                         const std::string &version) {
  PassManager pm(module.getContext());
  pm.addPass(stablehlo::createVhloToVersionPass({version}));
  if (failed(pm.run(module))) {
    return module->emitError("Failed VHLO to version") << version;
  }
  return success();
}

// Legalize VHLO portion of program to StableHLO, leaves TFL untouched.
LogicalResult ApplyVhloToStablehloPatterns(ModuleOp module) {
  MLIRContext *context = module.getContext();
  ConversionTarget target(*context);
  target.addIllegalDialect<vhlo::VhloDialect>();
  target.addLegalDialect<TFL::TensorFlowLiteDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<stablehlo::StablehloDialect>();

  VhloToStablehloTypeConverter converter;
  RewritePatternSet patterns(context);
  stablehlo::populateVhloToStablehloPatterns(&patterns, &converter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    return module->emitError("Failed partial conversion to StableHLO");
  }
  return success();
}

LogicalResult ApplyUnrealizedCastCanonicalization(ModuleOp module) {
  MLIRContext *context = module->getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  target.addIllegalOp<UnrealizedConversionCastOp>();
  populateReconcileUnrealizedCastsPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    return module->emitError("Failed to fold unrealized cast");
  }
  return success();
}

}  // namespace

struct LegalizeStablehloToVhloPass
    : public impl::LegalizeStablehloToVhloPassBase<
          LegalizeStablehloToVhloPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string target_version = tflite_supported_stablehlo_version;
    VhloToStablehloTypeConverter to_builtin_converter;

    // StableHLO --> VHLO (allow funcs)
    //   VHLO -> Downgrade to 0.19.0 / tflite_supported_stablehlo_version
    //     VHLO Tensor --> Builtin Tensor
    //       Remove cast(tensor->vhlo) -> cast(vhlo->tensor) pattern
    if (failed(ApplyStablehloToVhloPatterns(module,
                                            /*is_func_legal=*/true)) ||
        failed(ApplyVhloToVersionPatterns(module, target_version)) ||
        failed(ApplyTypeConverter(module, to_builtin_converter)) ||
        failed(ApplyUnrealizedCastCanonicalization(module)))
      return signalPassFailure();
  }
};

struct LegalizeVhloToStablehloPass
    : public impl::LegalizeVhloToStablehloPassBase<
          LegalizeVhloToStablehloPass> {
  void runOnOperation() override {
    // Revert the tensor types to VHLO
    auto module = getOperation();
    StablehloToOdmlTypeConverter to_vhlo_converter;

    // Builtin Tensor --> VHLO Tensor
    //   StableHLO --> VHLO
    //     VHLO --> Upgrade to current
    //       VHLO --> StableHLO
    //         Remove cast(tensor->vhlo) -> cast(vhlo->tensor) pattern
    if (failed(ApplyTypeConverter(module, to_vhlo_converter)) ||
        failed(ApplyVhloToVersionPatterns(module,
                                          stablehlo::getCurrentVersion())) ||
        failed(ApplyVhloToStablehloPatterns(module)) ||
        failed(ApplyUnrealizedCastCanonicalization(module)))
      return signalPassFailure();
  }
};

static PassRegistration<LegalizeStablehloToVhloPass> pass_s2v;
static PassRegistration<LegalizeVhloToStablehloPass> pass_v2s;

}  // namespace odml
}  // namespace mlir
