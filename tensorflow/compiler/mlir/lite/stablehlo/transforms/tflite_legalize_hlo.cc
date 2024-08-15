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

// The kept headers are provided for the included file `passes.h.inc`.
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/dot_general.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/gather.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/sort.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
namespace {

// Returns the shape of the given value in a Constant Op.
arith::ConstantOp ShapeToConst(PatternRewriter& rewriter, Value value) {
  ArrayRef<int64_t> shape = mlir::cast<ShapedType>(value.getType()).getShape();
  auto attr_type = RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                         rewriter.getIntegerType(64));
  auto attr = DenseElementsAttr::get(attr_type, shape);
  return rewriter.create<arith::ConstantOp>(value.getLoc(), attr_type, attr);
}

bool IsSign(APInt a, APInt sign) {
  if (a.isZero()) return a == sign;
  if (a.isNegative()) return sign == -1;
  return sign == 1;
}

bool IsSign(APFloat a, APFloat sign) {
  if (a.isNaN() || a.isZero()) return a == sign;
  if (a.isNegative()) return sign.isExactlyValue(-1.0);
  return sign.isExactlyValue(1.0);
}

bool IsDenseSplatIntAttr(ElementsAttr float_or_int) {
  return mlir::isa<SplatElementsAttr>(float_or_int) &&
         mlir::isa<DenseIntElementsAttr>(float_or_int);
}

bool IsDenseSplatFloatAttr(ElementsAttr float_or_int) {
  return mlir::isa<SplatElementsAttr>(float_or_int) &&
         mlir::isa<DenseFPElementsAttr>(float_or_int);
}

bool ValueEquals(ElementsAttr float_or_int, double rhs) {
  if (IsDenseSplatFloatAttr(float_or_int)) {
    return mlir::cast<SplatElementsAttr>(float_or_int)
        .getSplatValue<APFloat>()
        .isExactlyValue(rhs);
  } else if (IsDenseSplatIntAttr(float_or_int)) {
    return mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APInt>() ==
           static_cast<int>(rhs);
  }
  return false;
}

// Returns whether the splat constant is the sign of the int or float Tensor.
bool TensorIsSign(PatternRewriter& rewriter, ElementsAttr float_or_int,
                  ElementsAttr sgn_cst) {
  auto sgn_splat = llvm::dyn_cast<SplatElementsAttr>(sgn_cst);
  if (!sgn_splat) return false;

  auto splat = dyn_cast<SplatElementsAttr>(float_or_int);
  if (auto float_spl = llvm::dyn_cast_if_present<FloatAttr>(splat),
      sgn_cst_spl = llvm::dyn_cast_if_present<FloatAttr>(sgn_splat);
      float_spl && sgn_cst_spl) {
    return IsSign(float_spl.getValue(), sgn_cst_spl.getValue());
  }
  if (auto int_spl = llvm::dyn_cast_if_present<IntegerAttr>(splat),
      sgn_cst_spl = llvm::dyn_cast_if_present<IntegerAttr>(sgn_splat);
      int_spl && sgn_cst_spl) {
    return IsSign(int_spl.getValue(), sgn_cst_spl.getValue());
  }
  if (mlir::isa<DenseFPElementsAttr>(float_or_int)) {
    auto sgn_splat_value = sgn_splat.getSplatValue<APFloat>();
    return llvm::all_of(float_or_int.getValues<APFloat>(), [&](APFloat value) {
      return IsSign(value, sgn_splat_value);
    });
  }
  if (mlir::isa<DenseIntElementsAttr>(float_or_int)) {
    auto sgn_splat_value = sgn_splat.getSplatValue<APInt>();
    return llvm::all_of(float_or_int.getValues<APInt>(), [&](APInt value) {
      return IsSign(value, sgn_splat_value);
    });
  }
  return false;
}

bool SameTypeOrDefaultCompare(mhlo::ComparisonTypeAttr comparison_type_attr,
                              ElementsAttr cst) {
  if (!comparison_type_attr) return true;
  auto comparison_type_attr_value = comparison_type_attr.getValue();
  if (comparison_type_attr_value == mhlo::ComparisonType::FLOAT &&
      IsDenseSplatFloatAttr(cst)) {
    return true;
  }
  if ((comparison_type_attr_value == mhlo::ComparisonType::SIGNED ||
       comparison_type_attr_value == mhlo::ComparisonType::UNSIGNED) &&
      IsDenseSplatIntAttr(cst)) {
    return true;
  }
  return false;
}

bool ValueIsReciprocal(ElementsAttr float_or_int, ElementsAttr rhs) {
  if (IsDenseSplatFloatAttr(float_or_int) &&
      IsDenseSplatFloatAttr(float_or_int)) {
    return (mlir::cast<SplatElementsAttr>(float_or_int)
                .getSplatValue<APFloat>() *
            mlir::cast<SplatElementsAttr>(rhs).getSplatValue<APFloat>())
        .isExactlyValue(1.0);
  } else if (IsDenseSplatIntAttr(float_or_int) &&
             IsDenseSplatIntAttr(float_or_int)) {
    return (mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APInt>() *
            mlir::cast<SplatElementsAttr>(rhs).getSplatValue<APInt>()) == 1;
  }
  return false;
}

bool ValueGreaterThanZero(ElementsAttr float_or_int) {
  if (IsDenseSplatIntAttr(float_or_int)) {
    auto value =
        mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APInt>();
    return !value.isNegative() && !value.isZero();
  } else if (IsDenseSplatFloatAttr(float_or_int)) {
    auto value =
        mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APFloat>();
    return !value.isNaN() && !value.isNegative() && !value.isZero();
  }
  return false;
}

#define GEN_PASS_DEF_LEGALIZEHLOTOTFLITEPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

bool SupportedComparisonType(mhlo::ComparisonTypeAttr comp_type) {
  if (!comp_type) return true;
  auto c_ty = comp_type.getValue();
  return c_ty == mhlo::ComparisonType::FLOAT ||
         c_ty == mhlo::ComparisonType::SIGNED ||
         c_ty == mhlo::ComparisonType::UNSIGNED ||
         c_ty == mhlo::ComparisonType::NOTYPE;
}

class LegalizeHloToTfLitePass
    : public impl::LegalizeHloToTfLitePassBase<LegalizeHloToTfLitePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeHloToTfLitePass);

  void runOnOperation() override;
};

std::optional<bool> IsCbrtLegal(mhlo::CbrtOp op) {
  return !op.getType().getElementType().isF32();
}

bool IsNotOpLegal(mhlo::NotOp op) {
  return op.getType().getElementType().isInteger(64);
}

// Mark possible target ops from rounding patterns as having "unknown"
// legality. This is required to schedule patterns on these ops even
// though MhloDialect is explicitly marked legal (which cannot be changed
// easily).
void AddRoundingOpsAsUnknown(ConversionTarget& target) {
  target.addDynamicallyLegalOp<
      mhlo::FloorOp, mhlo::SubtractOp, mhlo::AndOp, mhlo::SelectOp, mhlo::RemOp,
      mhlo::AddOp, mhlo::SignOp, mhlo::MulOp, mhlo::DivOp, mhlo::OrOp,
      mhlo::BroadcastInDimOp, mhlo::ConstantOp, mhlo::RoundOp, mhlo::TupleOp>(
      [](Operation* op) { return std::nullopt; });
}

bool IsCompareLegal(mhlo::CompareOp op) {
  return !SupportedComparisonType(op.getCompareTypeAttr());
}

void SetUnaryOpLegal(ConversionTarget& target) {
  auto is_legal = [](Operation* op) {
    return !llvm::cast<ShapedType>(op->getOperand(0).getType())
                .getElementType()
                .isIntOrFloat();
  };
  target.addDynamicallyLegalOp<
      mhlo::AbsOp, mhlo::BitcastConvertOp, mhlo::CeilOp, mhlo::IsFiniteOp,
      mhlo::CosineOp, mhlo::ExpOp, mhlo::Expm1Op, mhlo::FloorOp, mhlo::ImagOp,
      mhlo::LogOp, mhlo::NegOp, mhlo::RealOp, mhlo::Log1pOp, mhlo::RsqrtOp,
      mhlo::SineOp, mhlo::LogisticOp, mhlo::SignOp, mhlo::SqrtOp, mhlo::TanhOp,
      mhlo::ConvertOp>(is_legal);
}

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_tflite_legalize_hlo.inc"
void LegalizeHloToTfLitePass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<odml::ConvertCustomCallOp, odml::LowerDotGeneralOp>(context);
  populateWithGenerated(patterns);

  ConversionTarget target(*context);
  target.addLegalDialect<TFL::TensorFlowLiteDialect, mhlo::MhloDialect>();
  target.addLegalOp<func::CallOp, func::ConstantOp, arith::ConstantOp>();

  target.addDynamicallyLegalOp<mhlo::CustomCallOp>(IsCustomCallLegal);
  target.addDynamicallyLegalOp<mhlo::CbrtOp>(IsCbrtLegal);
  target.addIllegalOp<mhlo::ClampOp, mhlo::DynamicReshapeOp, mhlo::RemOp,
                      mhlo::ReshapeOp, mhlo::ShiftRightArithmeticOp,
                      mhlo::ShiftRightLogicalOp, mhlo::DotGeneralOp,
                      mhlo::DotOp, mhlo::TransposeOp>();
  target.addDynamicallyLegalOp<mhlo::NotOp>(IsNotOpLegal);
  target.addDynamicallyLegalOp<mhlo::CompareOp>(IsCompareLegal);

  AddRoundingOpsAsUnknown(target);
  SetUnaryOpLegal(target);

  PopulatePadPatterns(context, patterns, target);
  PopulateReducePatterns(context, patterns, target);
  PopulateLegalizeReduceWindowPatterns(context, patterns, target);
  PopulateGatherPatterns(context, patterns, target);
  PopulateLegalizeConvPatterns(context, patterns, target);
  PopulateLegalizeSlicePatterns(context, patterns, target);
  PopulateSortPatterns(context, patterns, target);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("mhlo to TFLite legalization failed.");
    signalPassFailure();
  }
}

}  // namespace


// Creates an instance of the pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHloToTfLitePass() {
  return std::make_unique<LegalizeHloToTfLitePass>();
}

// Registers the pass implementation
static PassRegistration<LegalizeHloToTfLitePass> pass;

}  // namespace odml
}  // namespace mlir
