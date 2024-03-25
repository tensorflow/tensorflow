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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/dot_general.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
namespace {

// This file is generated from `passes.td` and provides the implementation base
// class.
#define GEN_PASS_DEF_LEGALIZEHLOTOTFLITEPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

class LegalizeHloToTfLitePass
    : public impl::LegalizeHloToTfLitePassBase<LegalizeHloToTfLitePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeHloToTfLitePass);

  void runOnOperation() override;
};

class ConvertReduceOpToTFLiteArgmax
    : public ConvertReduceOpToArgMinMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                                        TFL::ReduceAnyOp, true> {
 public:
  using ConvertReduceOpToArgMinMax::ConvertReduceOpToArgMinMax;

  bool IsValueInitValue(const DenseElementsAttr& attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (element_type.isa<FloatType>()) {
      auto value = *attr.value_begin<APFloat>();
      return value.isNegative() && value.isInfinity();
    } else if (element_type.isInteger(1)) {
      auto value = *attr.value_begin<APInt>();
      return value.isZero();
    } else {
      auto value = *attr.value_begin<APInt>();
      return element_type.isUnsignedInteger() ? value.isMinValue()
                                              : value.isMinSignedValue();
    }
  }
};

class ConvertReduceOpToTFLiteArgmin
    : public ConvertReduceOpToArgMinMax<TFL::ReduceMinOp, TFL::ArgMinOp,
                                        TFL::ReduceAllOp, false> {
 public:
  using ConvertReduceOpToArgMinMax::ConvertReduceOpToArgMinMax;

  bool IsValueInitValue(const DenseElementsAttr& attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (element_type.isa<FloatType>()) {
      auto value = *attr.value_begin<APFloat>();
      return !value.isNegative() && value.isInfinity();
    } else if (element_type.isInteger(1)) {
      auto value = *attr.value_begin<APInt>();
      return value.isZero();
    } else {
      auto value = *attr.value_begin<APInt>();
      return element_type.isUnsignedInteger() ? value.isMaxValue()
                                              : value.isMaxSignedValue();
    }
  }
};

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_tflite_legalize_hlo.inc"
void LegalizeHloToTfLitePass::runOnOperation() {
  MLIRContext& context = getContext();
  RewritePatternSet patterns(&getContext());
  // Add new conversion patterns here.
  PopulateLegalizeHloToTFLitePatterns(&patterns, &context);

  ConversionTarget target(context);
  target.addLegalDialect<TFL::TensorFlowLiteDialect, mhlo::MhloDialect>();
  target.addLegalOp<func::CallOp, func::ConstantOp, arith::ConstantOp>();
  target.addDynamicallyLegalOp<mhlo::CustomCallOp>(IsCustomCallLegal);
  target.addDynamicallyLegalOp<mhlo::ReduceOp>(IsReduceOpLegal);
  // Converted MHLO ops should be marked illegal here.
  // TODO: b/304003568 - Add TF_TransposeOp folding logic to tflite.
  target.addIllegalOp<mhlo::DotGeneralOp, mhlo::DotOp, mhlo::TransposeOp>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("mhlo to TFLite legalization failed.");
    signalPassFailure();
  }
}
}  // namespace

void PopulateLegalizeHloToTFLitePatterns(RewritePatternSet* patterns,
                                         MLIRContext* context) {
  patterns->add<odml::ConvertCustomCallOp>(context);
  populateWithGenerated(*patterns);

  patterns->add<ConvertReduceOpToTFLiteArgmin, ConvertReduceOpToTFLiteArgmax>(
      context);
}

// Creates an instance of the pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHloToTfLitePass() {
  return std::make_unique<LegalizeHloToTfLitePass>();
}

// Registers the pass implementation
static PassRegistration<LegalizeHloToTfLitePass> pass;

}  // namespace odml
}  // namespace mlir
