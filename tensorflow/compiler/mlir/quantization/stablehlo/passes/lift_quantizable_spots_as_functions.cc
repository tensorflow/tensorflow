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
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.h"
// TODO - b/303543789: Remove TF Quantizer util dependency.

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_LIFTQUANTIZABLESPOTSASFUNCTIONSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// TODO - b/303543789: Move the helper functions below to a separate util.
// Fetches the default or null attribute, used for pattern matching.
static Attribute DefaultOrNullAttr(OpBuilder& builder, Attribute& attr) {
  if (!attr) {
    return builder.getStringAttr(kNullAttributeValue);
  }
  return attr;
}

// Checks whether the value of a constant equals the given float, regardless
// of the tensor dimension.
static bool FloatValueEquals(const Attribute& attr, double value) {
  auto fp_attr = attr.dyn_cast_or_null<DenseFPElementsAttr>();
  if (!fp_attr) return false;

  if (fp_attr.isSplat()) {
    return fp_attr.getSplatValue<APFloat>().isExactlyValue(value);
  }
  return llvm::all_of(fp_attr.getValues<APFloat>(), [value](const APFloat& f) {
    return f.isExactlyValue(value);
  });
}

class LiftQuantizableSpotsAsFunctionsPass
    : public impl::LiftQuantizableSpotsAsFunctionsPassBase<
          LiftQuantizableSpotsAsFunctionsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LiftQuantizableSpotsAsFunctionsPass)

  explicit LiftQuantizableSpotsAsFunctionsPass() = default;

 private:
  void runOnOperation() override;
};

namespace simple_patterns {
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/lift_quantizable_spots_as_functions_simple.inc"
}

namespace fusion_patterns {
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/lift_quantizable_spots_as_functions_fusion.inc"
}

void LiftQuantizableSpotsAsFunctionsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module_op = getOperation();

  simple_patterns::populateWithGenerated(patterns);
  fusion_patterns::populateWithGenerated(patterns);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (failed(applyPatternsAndFoldGreedily(func, frozen_patterns))) {
      func.emitError()
          << "quant-stablehlo-lift-quantizable-spots-as-functions failed.";
      signalPassFailure();
    }
  }

  // Remove all attr_map attributes.
  module_op.walk([&](Operation* op) { op->removeAttr(kAttrMapAttribute); });
}

}  // namespace

}  // namespace mlir::quant::stablehlo
