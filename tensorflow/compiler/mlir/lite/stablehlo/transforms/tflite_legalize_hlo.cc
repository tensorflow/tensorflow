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
#include "llvm/ADT/SmallSet.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/dot_general.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/gather.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/get_dimension_size.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/if.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/iota.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/sort.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/while.h"
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

// Returns true if broadcast_dimensions obey Tensorflow convention, as in new
// dimensions are added as prefix.
bool IsTFLStyleBroadcast(DenseIntElementsAttr broadcast_dimensions,
                         Value output) {
  // broadcast_dimensions is an increasing list by definition, thus it suffices
  // to check the first element.
  int64_t input_rank = broadcast_dimensions.getNumElements();
  int64_t output_rank = mlir::cast<ShapedType>(output.getType()).getRank();
  return input_rank == 0 ||
         (broadcast_dimensions.getValues<APInt>()[0].getSExtValue() ==
          output_rank - input_rank);
}

// Returns the intermediate shape that input tensor should be reshaped to during
// legalization of BroadcastInDimOp.
arith::ConstantOp ExpandedShape(OpBuilder& b, Value input,
                                DenseIntElementsAttr broadcast_dimensions,
                                Value output) {
  // Initialize expanded shape with output rank and dimensions of 1.
  llvm::SmallVector<Attribute> expanded_shape(
      llvm::cast<ShapedType>(output.getType()).getRank(),
      /*Value=*/b.getI32IntegerAttr(1));

  // Set dimension sizes specified by broadcast_dimensions.
  auto input_shape = llvm::cast<ShapedType>(input.getType()).getShape();

  for (auto x : llvm::enumerate(broadcast_dimensions)) {
    expanded_shape[x.value().getSExtValue()] =
        b.getI32IntegerAttr(static_cast<int32_t>(input_shape[x.index()]));
  }

  // Create the expanded type wrapped in a arith::ConstantOp.
  auto attr_type = RankedTensorType::get(
      {static_cast<int64_t>(expanded_shape.size())}, b.getIntegerType(32));
  auto attr = DenseElementsAttr::get(attr_type, expanded_shape);
  return b.create<arith::ConstantOp>(output.getLoc(), attr_type, attr);
}

Value ExpandedDynamicShape(OpBuilder& b, Value input,
                           DenseIntElementsAttr broadcast_dimensions,
                           Value output) {
  int64_t output_rank = mlir::cast<ShapedType>(output.getType()).getRank();
  llvm::SmallVector<int64_t, 4> expanded_dimensions;
  llvm::SmallSet<int64_t, 4> broadcast_dimensions_values;

  for (auto x : llvm::enumerate(broadcast_dimensions)) {
    broadcast_dimensions_values.insert(x.value().getSExtValue());
  }

  for (int64_t i = 0; i < output_rank; i++) {
    if (!broadcast_dimensions_values.contains(i)) {
      expanded_dimensions.push_back(i);
    }
  }

  Value expanded_input = input;

  for (int64_t i : expanded_dimensions) {
    auto index_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({}, b.getI64Type()), {i});
    Value index = b.create<arith::ConstantOp>(output.getLoc(), index_attr);

    auto cur_type = llvm::cast<ShapedType>(expanded_input.getType());
    auto cur_shape = cur_type.getShape();
    llvm::SmallVector<int64_t> new_shape;

    auto begin = cur_shape.begin();
    new_shape.append(begin, begin + i);
    new_shape.push_back(1);
    new_shape.append(begin + i, cur_shape.end());

    auto new_type = RankedTensorType::get(new_shape, cur_type.getElementType());

    expanded_input = b.create<TFL::ExpandDimsOp>(output.getLoc(), new_type,
                                                 expanded_input, index);
  }

  return expanded_input;
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
      // go/keep-sorted start
      // clang-format off
      mhlo::AbsOp,
      mhlo::BitcastConvertOp,
      mhlo::CeilOp,
      mhlo::ConvertOp,
      mhlo::CosineOp,
      mhlo::ExpOp,
      mhlo::Expm1Op,
      mhlo::FloorOp,
      mhlo::ImagOp,
      mhlo::IsFiniteOp,
      mhlo::Log1pOp,
      mhlo::LogOp,
      mhlo::LogisticOp,
      mhlo::NegOp,
      mhlo::RealOp,
      mhlo::RsqrtOp,
      mhlo::SignOp,
      mhlo::SineOp,
      mhlo::SqrtOp,
      mhlo::TanhOp
      // clang-format on
      // go/keep-sorted end
      >(is_legal);
}

// mhlo "bitwise ops" can be both bitwise (floats/ints) or logical (bools).
// TFL ops are only one of logical or bitwise.
void SetBinaryBitwiseLegal(ConversionTarget& target) {
  auto is_logical = [](Operation* op) {
    return llvm::cast<ShapedType>(op->getResultTypes()[0])
        .getElementType()
        .isInteger(1);
  };
  auto is_bitwise = [&](Operation* op) { return !is_logical(op); };
  target.addDynamicallyLegalOp<mhlo::OrOp, mhlo::AndOp>(is_bitwise);
  target.addDynamicallyLegalOp<mhlo::XorOp>(is_logical);
}

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_tflite_legalize_hlo.inc"
void LegalizeHloToTfLitePass::runOnOperation() {
  MLIRContext* context = &getContext();

  // Apply large rounding related patterns first without dialect conversion.
  // This unlocks cleaner match/fold behavior, making these patterns less
  // sensitive to broadcasted constants.
  {
    RewritePatternSet patterns(context);
    patterns.add<
        // clang-format off
        Phase1_Round,
        Phase1_FloorMod,
        Phase1_FloorMod2,
        Phase1_FloorDiv,
        Phase1_FloorDiv2,
        Phase1_FloorDiv3,
        Phase1_FloorDiv4,
        Phase1_FloorDiv5
        // clang-format on
        >(context);

    (void)applyPatternsAndFoldGreedily(getOperation().getOperation(),
                                       std::move(patterns));
  }

  {
    OpPassManager phase_2("builtin.module");
    phase_2.addPass(mlir::odml::CreateUnfoldSplatConstantPass());
    if (failed(runPipeline(phase_2, getOperation()))) {
      return signalPassFailure();
    }
  }

  ConversionTarget target(*context);
  target.addLegalDialect<TFL::TensorFlowLiteDialect, mhlo::MhloDialect>();
  target.addLegalOp<func::CallOp, func::ConstantOp, arith::ConstantOp>();

  target.addDynamicallyLegalOp<mhlo::CbrtOp>(IsCbrtLegal);
  target.addDynamicallyLegalOp<mhlo::NotOp>(IsNotOpLegal);
  target.addDynamicallyLegalOp<mhlo::CompareOp>(IsCompareLegal);
  target.addDynamicallyLegalOp<mhlo::TupleOp>(
      [](mhlo::TupleOp op) { return std::nullopt; });

  target.addIllegalOp<
      // go/keep-sorted start
      // clang-format off
      mhlo::AddOp,
      mhlo::Atan2Op,
      mhlo::BroadcastInDimOp,
      mhlo::ClampOp,
      mhlo::ConcatenateOp,
      mhlo::ConstantOp,
      mhlo::DivOp,
      mhlo::DotGeneralOp,
      mhlo::DotOp,
      mhlo::DynamicBroadcastInDimOp,
      mhlo::DynamicReshapeOp,
      mhlo::MaxOp,
      mhlo::MinOp,
      mhlo::MulOp,
      mhlo::PowOp,
      mhlo::RemOp,
      mhlo::ReshapeOp,
      mhlo::ReverseOp,
      mhlo::RoundNearestEvenOp,
      mhlo::RoundOp,
      mhlo::SelectOp,
      mhlo::ShiftRightArithmeticOp,
      mhlo::ShiftRightLogicalOp,
      mhlo::SubtractOp,
      mhlo::TransposeOp
      // clang-format on
      // go/keep-sorted end
      >();

  RewritePatternSet patterns(context);

  populateWithGenerated(patterns);

  SetUnaryOpLegal(target);
  SetBinaryBitwiseLegal(target);

  PopulatePadPatterns(context, patterns, target);
  PopulateReducePatterns(context, patterns, target);
  PopulateLegalizeReduceWindowPatterns(context, patterns, target);
  PopulateGatherPatterns(context, patterns, target);
  PopulateLegalizeConvPatterns(context, patterns, target);
  PopulateLegalizeSlicePatterns(context, patterns, target);
  PopulateSortPatterns(context, patterns, target);
  PopulateIotaPatterns(context, patterns, target);
  PopulateWhilePatterns(context, patterns, target);
  PopulateGetDimensionSizePatterns(context, patterns, target);
  PopulateIfPatterns(context, patterns, target);
  PopulateCustomCallPatterns(context, patterns, target);

  patterns.add<odml::LowerDotGeneralOp>(context);

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
