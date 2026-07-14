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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_WHILE_LOOP_OUTLINE_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_WHILE_LOOP_OUTLINE_PASS_H_

#include <string>

#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

namespace mlir {
namespace TFL {

// Pass to hoist while op regions into functions.
// This pass outlines the cond/body region of the TFL WhileOp into functions and
// replaces the regions with calls to these outlined functions.
class WhileOutlinePass : public TFL::Pass<WhileOutlinePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WhileOutlinePass)

  WhileOutlinePass() = default;
  WhileOutlinePass(const WhileOutlinePass&) {};

  void runOnOperation() override;
  static llvm::StringRef GetName() { return "WhileOutlinePass"; }
  static llvm::StringRef GetArgument() { return "tfl-while-loop-outline"; }
  static llvm::StringRef GetDescription() {
    return "Pass to hoist while op regions into functions";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }

  // Outlines the regions of the WhileOp's cond and body and insert function
  // calls instead,
  void OutlineWhile(WhileOp while_op);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper_;
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_WHILE_LOOP_OUTLINE_PASS_H_
