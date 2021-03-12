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

// The TF dialect uses some TF types that are illegal in the MHLO dialect and
// some generic types that are legal in MHLO. This pass legalizes TF types into
// types that are legal in MHLO. For example, TF::Qint8Type is converted to i8.
// Rewrites here should run before TF to MHLO op legalizations are run.
// TODO(b/180234029): The rewrite here should be part of the LegalizeTF pass
// rather than its own pass.

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"

#define DEBUG_TYPE "xla-legalize-tf-types"

namespace mlir {
namespace mhlo {
namespace {

bool IsIllegalElementType(Type type) {
  return type
      .isa<mlir::TF::Qint8Type, mlir::TF::Qint16Type, mlir::TF::Qint32Type,
           mlir::TF::Quint8Type, mlir::TF::Quint16Type>();
}

Type ToLegalElementType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<mlir::TF::Qint8Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 8);
      })
      .Case<mlir::TF::Qint16Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 16);
      })
      .Case<mlir::TF::Qint32Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 32);
      })
      .Case<mlir::TF::Quint8Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 8,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Case<mlir::TF::Quint16Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 16,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Default([&type](Type) { return type; });
}

// TODO(b/180234863): What's below this line is generic so convert it to a
// utility.

bool IsIllegalType(Type type) {
  return IsIllegalElementType(getElementTypeOrSelf(type));
}

Type ToLegalType(Type type) {
  if (IsIllegalElementType(type)) return ToLegalElementType(type);
  if (auto shaped = type.dyn_cast<ShapedType>()) {
    Type elem = shaped.getElementType();
    if (IsIllegalType(elem)) return shaped.clone(ToLegalType(elem));
  }
  return type;
}

class TfTypeConverter : public TypeConverter {
 public:
  TfTypeConverter() {
    addConversion([](Type type) -> Type {
      return IsIllegalType(type) ? ToLegalType(type) : type;
    });
  }
};

// An Op is illegal iff it contains an illegalType.
class TfTypeConversionTarget : public ConversionTarget {
 public:
  explicit TfTypeConversionTarget(MLIRContext &ctx, TfTypeConverter &converter)
      : ConversionTarget(ctx), converter_(converter) {
    markUnknownOpDynamicallyLegal();
  }

 protected:
  bool isDynamicallyLegal(Operation *op) const override {
    // The FuncOp type can contain types that the op's operand and result types
    // do not contain.
    if (auto func = dyn_cast<FuncOp>(op)) {
      if (!converter_.isSignatureLegal(func.getType())) return false;
    }
    return converter_.isLegal(op);
  }

 private:
  TfTypeConverter &converter_;
};

class TfTypePattern : public ConversionPattern {
 public:
  TfTypePattern(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(1, converter, MatchAnyOpTypeTag()) {}

  // The dialect conversion framework will call this matchAndRewrite on each
  // Operation in the IR tree. This call matchAndRewrite needs to update the
  // Operation's results and child regions.
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Update the results.
    llvm::SmallVector<Type, 4> new_results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                new_results)))
      return failure();

    // Update the regions. The dialect conversion framework wants new regions to
    // be created and updated, rather than updating the old op. Thus we use an
    // OperationState so we can add regions to the new up.
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      Region &new_region = *state.addRegion();
      rewriter.inlineRegionBefore(region, new_region, new_region.begin());
      if (failed(rewriter.convertRegionTypes(&new_region, *getTypeConverter())))
        return failure();
    }
    rewriter.replaceOp(op, rewriter.createOperation(state)->getResults());

    return success();
  }
};

struct LegalizeTfTypesPass
    : public LegalizeTfTypesPassBase<LegalizeTfTypesPass> {
  void runOnOperation() override;
};

void LegalizeTfTypesPass::runOnOperation() {
  TfTypeConverter converter;
  OwningRewritePatternList patterns;
  patterns.insert<TfTypePattern>(&getContext(), converter);
  populateFuncOpTypeConversionPattern(patterns, &getContext(), converter);
  TfTypeConversionTarget target(getContext(), converter);
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

static PassRegistration<LegalizeTfTypesPass> registration(
    "xla-legalize-tf-types",
    "Replace TensorFlow types with types that are legal in the MHLO dialect");

}  // namespace

std::unique_ptr<OperationPass<>> CreateLegalizeTfTypesPass() {
  return std::make_unique<LegalizeTfTypesPass>();
}

}  // namespace mhlo
}  // namespace mlir
