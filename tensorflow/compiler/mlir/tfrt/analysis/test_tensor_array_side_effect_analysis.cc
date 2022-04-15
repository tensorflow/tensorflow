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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/analysis/tensor_array_side_effect_analysis.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

class TestTensorArraySideEffectAnalysis
    : public mlir::PassWrapper<TestTensorArraySideEffectAnalysis,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTensorArraySideEffectAnalysis)

 private:
  llvm::StringRef getArgument() const final {
    return "tfrt-test-tensor-array-effect";
  }
  llvm::StringRef getDescription() const final {
    return "Test TensorArraySideEffectAnalysis";
  }
  void runOnOperation() override {
    auto module = getOperation();
    TensorArraySideEffectAnalysis tensor_array_side_effect_analysis(module);

    for (auto func_op : module.getOps<mlir::func::FuncOp>()) {
      func_op.emitRemark() << "HasAtMostTensorArrayEffect: "
                           << tensor_array_side_effect_analysis
                                  .HasAtMostTensorArrayEffect(func_op);
    }
  }
};

mlir::PassRegistration<TestTensorArraySideEffectAnalysis> pass;

}  // namespace
}  // namespace tfrt_compiler
}  // namespace tensorflow
