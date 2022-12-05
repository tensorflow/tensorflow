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

#include <utility>

#include "llvm/ADT/None.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/variables_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZEVARIABLESPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Attribute name to identify whether variables should be legalized to TFLite or
// not.
const char kLegalizeTflVariables[] = "tfl._legalize_tfl_variables";

bool HasSupportedElementType(Operation* op) {
  return utils::IsSupportedVariableType(op);
}

bool IsSupportedElementType(ShapedType type) {
  return utils::IsSupportedVariableType(type);
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_legalize_variables.inc"

// Pass which legalizes TF variables which are already passed as bounded
// arguments to functions, to TFLite variables.
class LegalizeVariablesPass
    : public impl::LegalizeVariablesPassBase<LegalizeVariablesPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeVariablesPass)

  void runOnOperation() override {
    auto module = getOperation();
    // If TFLite variable legalization is not allowed, then we skip this pass.
    if (auto legalize_tfl_variables_attr =
            module->getAttr(kLegalizeTflVariables)) {
      if (!legalize_tfl_variables_attr.cast<BoolAttr>().getValue()) return;
    }

    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeVariablesPass() {
  return std::make_unique<LegalizeVariablesPass>();
}

}  // namespace TFL
}  // namespace mlir
