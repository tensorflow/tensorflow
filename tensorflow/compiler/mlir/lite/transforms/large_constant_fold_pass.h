/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LARGE_CONSTANT_FOLD_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LARGE_CONSTANT_FOLD_PASS_H_

#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"

namespace mlir {
namespace TFL {

struct LargeConstantFoldPassOptions : public mlir::detail::PassOptions {
  mlir::detail::PassOptions::Option<bool> fold_fp16_resource_casts{
      *this, "fold-fp16-resource-casts",
      llvm::cl::desc("Fold fp16/bf16 resource casts"), llvm::cl::init(true)};
  mlir::detail::PassOptions::Option<bool> fold_elementwise_ops{
      *this, "fold-elementwise-ops",
      llvm::cl::desc(
          "Fold elementwise binary operations on resource constants"),
      llvm::cl::init(false)};
};

class LargeConstantFoldPass
    : public TFL::Pass<LargeConstantFoldPass, LargeConstantFoldPassOptions,
                       ModuleOp> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LargeConstantFoldPass)

  LargeConstantFoldPass() = default;
  explicit LargeConstantFoldPass(const mlir::detail::PassOptions& options)
      : TFL::Pass<LargeConstantFoldPass, LargeConstantFoldPassOptions,
                  ModuleOp>(options) {}
  explicit LargeConstantFoldPass(bool fold_fp16_resource_casts,
                                 bool fold_elementwise_ops = false) {
    GetOptions().fold_fp16_resource_casts = fold_fp16_resource_casts;
    GetOptions().fold_elementwise_ops = fold_elementwise_ops;
  }
  LargeConstantFoldPass(const LargeConstantFoldPass& other) = default;

  void runOnOperation() override;
  static llvm::StringRef GetName() { return "LargeConstantFoldPass"; }
  static llvm::StringRef GetArgument() { return "tfl-large-constant-fold"; }
  static llvm::StringRef GetDescription() {
    return "Fold operations on large constant resource attributes across "
           "functions.";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, mlir::TFL::TensorFlowLiteDialect,
                    mlir::arith::ArithDialect>();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> CreateLargeConstantFoldPass(
    bool fold_fp16_resource_casts = true, bool fold_elementwise_ops = false);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LARGE_CONSTANT_FOLD_PASS_H_
