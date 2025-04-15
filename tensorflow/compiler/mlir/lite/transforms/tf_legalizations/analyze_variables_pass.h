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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_ANALYZE_VARIABLES_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_ANALYZE_VARIABLES_PASS_H_

#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"

namespace mlir {
namespace TFL {

// Pass which analyzes the variables in the graph and add an attribute whether
// variables should be legalized to TFLite native ones.
// This pass needs to run post TF->TFL legalization and before variable
// legalization.

class AnalyzeVariablesPass : public TFL::Pass<AnalyzeVariablesPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeVariablesPass)

  AnalyzeVariablesPass() = default;
  AnalyzeVariablesPass(const AnalyzeVariablesPass&) {};

  void runOnOperation() override;
  static llvm::StringRef GetName() { return "AnalyzeVariablesPass"; }
  static llvm::StringRef GetArgument() { return "tfl-analyze-variables-pass"; }
  static llvm::StringRef GetDescription() {
    return "Pass to analyze variables in the graph";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_ANALYZE_VARIABLES_PASS_H_
