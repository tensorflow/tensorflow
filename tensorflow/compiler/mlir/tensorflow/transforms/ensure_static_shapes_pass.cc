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

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {

namespace {

class TFEnsureStaticShapes
    : public TensorFlowEnsureStaticShapesPassBase<TFEnsureStaticShapes> {
 public:
  void runOnFunction() override {
    auto result = getFunction()->walk([&](mlir::Operation* op) {
      auto is_dynamic_shape_type = [](const mlir::Type type) {
        if (ShapedType shaped_type = type.dyn_cast<ShapedType>()) {
          return !shaped_type.hasStaticShape();
        }
        return false;
      };

      if (llvm::any_of(op->getResultTypes(), is_dynamic_shape_type) ||
          llvm::any_of(op->getOperandTypes(), is_dynamic_shape_type)) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTFEnsureStaticShapesPass() {
  return std::make_unique<TFEnsureStaticShapes>();
}

}  // namespace TF
}  // namespace mlir
