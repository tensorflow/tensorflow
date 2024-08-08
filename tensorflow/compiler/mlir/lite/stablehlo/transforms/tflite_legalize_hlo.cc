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
#include <memory>
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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

bool IsCompareLegal(mhlo::CompareOp op) {
  return !SupportedComparisonType(op.getCompareTypeAttr());
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
  target.addIllegalOp<mhlo::DotGeneralOp, mhlo::DotOp, mhlo::TransposeOp>();
  target.addDynamicallyLegalOp<mhlo::CompareOp>(IsCompareLegal);

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
