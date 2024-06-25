/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/bfloat16_type.h"
#include "tensorflow/core/platform/bfloat16.h"

namespace mlir::quant::stablehlo {
namespace {

class BFloat16TypeConverter : public TypeConverter {
 public:
  BFloat16TypeConverter() {
    addConversion([](const Type type) -> Type {
      return IsLargeFloatType(type) ? ToBfloat16Type(type) : type;
    });
  }
};

// This helper function makes legality check easier. Both convert ops in the
// patterns below are considered legal:
//  - `BitcastConvertOp` (i32 -> f32) + `ConvertOp` (f32 -> bf16)
//  - `ConvertOp` (bf16 -> f32) -> `BitcastConvertOp` (f32 -> i32)
template <typename ConvertOp, typename OtherConvertOp>
bool IsConvertOpLegal(ConvertOp convert_op, BFloat16TypeConverter& converter) {
  if (!converter.isLegal(convert_op.getOperand().getType())) {
    auto other_convert_op = dyn_cast_or_null<OtherConvertOp>(
        convert_op.getOperand().getDefiningOp());
    return other_convert_op &&
           converter.isLegal(other_convert_op.getOperand().getType());
  } else if (!converter.isLegal(convert_op.getResult().getType())) {
    if (!convert_op.getResult().hasOneUse()) {
      return false;
    }
    auto other_convert_op = dyn_cast_or_null<OtherConvertOp>(
        *convert_op.getResult().getUsers().begin());
    return other_convert_op &&
           converter.isLegal(other_convert_op.getResult().getType());
  }
  return true;
}

class BFloat16TypeConversionTarget : public ConversionTarget {
 public:
  explicit BFloat16TypeConversionTarget(MLIRContext& ctx,
                                        BFloat16TypeConverter& converter)
      : ConversionTarget(ctx), converter_(converter) {
    markUnknownOpDynamicallyLegal([this](Operation* op) {
      // The FuncOp type can contain types that the op's operand and result
      // types do not contain.
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        if (!converter_.isSignatureLegal(func.getFunctionType())) return false;
      } else if (auto bitcast_convert_op =
                     dyn_cast<mlir::stablehlo::BitcastConvertOp>(op)) {
        return IsConvertOpLegal<mlir::stablehlo::BitcastConvertOp,
                                mlir::stablehlo::ConvertOp>(bitcast_convert_op,
                                                            converter_);
      } else if (auto convert_op = dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
        return IsConvertOpLegal<mlir::stablehlo::ConvertOp,
                                mlir::stablehlo::BitcastConvertOp>(convert_op,
                                                                   converter_);
      }
      return converter_.isLegal(op);
    });
  }

 private:
  BFloat16TypeConverter& converter_;
};

class BFloat16TypePattern : public ConversionPattern {
 public:
  BFloat16TypePattern(TypeConverter& converter, MLIRContext* ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, const ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (getTypeConverter()->isLegal(op)) {
      return failure();
    }
    if (isa<mlir::stablehlo::BitcastConvertOp>(op)) {
      // Skip `BitcastConvertOp`, which is handled by the other pattern.
      return failure();
    }

    // Update the results.
    SmallVector<Type, 4> new_results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                new_results)))
      return failure();

    // Update the regions. The dialect conversion framework wants new regions to
    // be created and updated, rather than updating the old op. Thus we use an
    // OperationState so we can add regions to the new op.
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    for (Region& region : op->getRegions()) {
      Region& new_region = *state.addRegion();
      rewriter.inlineRegionBefore(region, new_region, new_region.begin());
      if (failed(rewriter.convertRegionTypes(&new_region, *getTypeConverter())))
        return failure();
    }

    // Convert value of ConstantOp to bfloat16.
    if (auto const_op = dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
      const auto values = const_op.getValue().tryGetValues<float>();
      if (!values.has_value()) {
        return failure();
      }
      const SmallVector<tensorflow::bfloat16> bfloat16_values(values->begin(),
                                                              values->end());
      state.attributes.set(
          const_op.getValueAttrName(),
          DenseFPElementsAttr::get(
              mlir::dyn_cast<ShapedType>(const_op.getValue().getType())
                  .clone(rewriter.getBF16Type()),
              bfloat16_values));
    }

    rewriter.replaceOp(op, rewriter.create(state)->getResults());

    return success();
  }
};

class BitcastConvertOpPattern
    : public OpConversionPattern<mlir::stablehlo::BitcastConvertOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::BitcastConvertOp op,
      mlir::stablehlo::BitcastConvertOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    const bool is_input_legal =
        getTypeConverter()->isLegal(op.getOperand().getType());
    const bool is_output_legal =
        getTypeConverter()->isLegal(op.getResult().getType());
    if (is_input_legal && is_output_legal) {
      return failure();
    } else if (is_input_legal) {
      // output is f32, we bitcast_convert to f32 and then convert to bf16.
      const Value output = rewriter.create<mlir::stablehlo::BitcastConvertOp>(
          op->getLoc(), op.getResult().getType(), adaptor.getOperand());
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(
          op, getTypeConverter()->convertType(op.getResult().getType()),
          output);
    } else if (is_output_legal) {
      // input is f32, we convert from bf16 and then bitcast_convert.
      const Value output = rewriter.create<mlir::stablehlo::ConvertOp>(
          op->getLoc(), op.getOperand().getType(), adaptor.getOperand());
      rewriter.replaceOpWithNewOp<mlir::stablehlo::BitcastConvertOp>(
          op, op.getResult().getType(), output);
    } else {
      // Both input/output are f32. Convert to no-op.
      rewriter.replaceOp(op, adaptor.getOperand());
    }
    return success();
  }
};
}  // namespace

#define GEN_PASS_DEF_CONVERTFUNCTOBFLOAT16PASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"
namespace {
class ConvertFuncToBfloat16Pass
    : public impl::ConvertFuncToBfloat16PassBase<ConvertFuncToBfloat16Pass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFuncToBfloat16Pass)

  explicit ConvertFuncToBfloat16Pass() = default;

 private:
  void runOnOperation() override;
};

void ConvertFuncToBfloat16Pass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  MLIRContext* context = func_op.getContext();
  RewritePatternSet patterns(context);

  BFloat16TypeConverter converter;
  patterns.add<BFloat16TypePattern, BitcastConvertOpPattern>(converter,
                                                             context);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  BFloat16TypeConversionTarget target(*context, converter);
  if (failed(applyPartialConversion(func_op.getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo
