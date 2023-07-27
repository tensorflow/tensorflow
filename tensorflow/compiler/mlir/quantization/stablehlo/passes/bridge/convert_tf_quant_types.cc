/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/lib/monitoring/counter.h"

namespace mlir {
namespace stablehlo {
namespace {

#define GEN_PASS_DEF_CONVERTTFQUANTTYPES
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

// TODO: b/290366702 - Temporarily added metrics for debugging.
auto *mlir_tf_quant_op_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/tf_quant_op_count" /*metric_name*/,
    "Counts the number of ops that has qint types" /*metric description*/,
    "op_name" /*metric label*/);

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

// Check if the op is a quantization op that supports quantized types.
// TODO: b/289560952 - Narrow down the list of ops using prod metrics.
bool IsUnSupportedOp(Operation *op) {
  return llvm::isa<
      // clang-format off
      // go/keep-sorted start
      TF::UniformDequantizeOp,
      TF::UniformQuantizeOp,
      TF::UniformQuantizedAddOp,
      TF::UniformQuantizedClipByValueOp,
      TF::UniformQuantizedConvolutionHybridOp,
      TF::UniformQuantizedConvolutionOp,
      TF::UniformQuantizedDotHybridOp,
      TF::UniformQuantizedDotOp,
      TF::UniformRequantizeOp
      // go/keep-sorted end
      // clang-format on
      >(op);
}

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

class TFQuantTypeConverter : public TypeConverter {
 public:
  TFQuantTypeConverter() {
    addConversion([](Type type) -> Type {
      return IsIllegalType(type) ? ToLegalType(type) : type;
    });
  }
};

// An Op is illegal iff it is non-UQ op and it contains qint types.
class TFQuantTypeConversionTarget : public ConversionTarget {
 public:
  explicit TFQuantTypeConversionTarget(MLIRContext &ctx,
                                       TFQuantTypeConverter &converter)
      : ConversionTarget(ctx), converter_(converter) {
    markUnknownOpDynamicallyLegal([this](Operation *op) {
      // Do not convert UnifromQuantized ops.
      if (IsUnSupportedOp(op)) {
        return true;
      }
      // The FuncOp type can contain types that the op's operand and result
      // types do not contain.
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        if (!converter_.isSignatureLegal(func.getFunctionType())) return false;
      }
      return converter_.isLegal(op);
    });
  }

 private:
  TFQuantTypeConverter &converter_;
};

class TFQuantTypePattern : public ConversionPattern {
 public:
  TFQuantTypePattern(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

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
    rewriter.replaceOp(op, rewriter.create(state)->getResults());

    // TODO: b/290366702 - Temporarily added metrics for debugging.
    mlir_tf_quant_op_count->GetCell(std::string(op->getName().getStringRef()))
        ->IncrementBy(1);
    return success();
  }
};

struct ConvertTFQuantTypes
    : public impl::ConvertTFQuantTypesBase<ConvertTFQuantTypes> {
  void runOnOperation() override;
};

// TODO: b/289560952 - add qint <-> int casts around TF UQ ops.
void ConvertTFQuantTypes::runOnOperation() {
  TFQuantTypeConverter converter;
  RewritePatternSet patterns(&getContext());
  patterns.add<TFQuantTypePattern>(&getContext(), converter);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  TFQuantTypeConversionTarget target(getContext(), converter);
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTFQuantTypesPass() {
  return std::make_unique<ConvertTFQuantTypes>();
}

}  // namespace stablehlo
}  // namespace mlir
