/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <initializer_list>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Block.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#define DEBUG_TYPE "tf-shape-inference"

namespace mlir {
namespace TF {

namespace {

// This transformation pass propagate shapes on the TensorFlow graph.
// It is a ModulePass in order to be able to change function types.
struct ShapeInference : public ModulePass<ShapeInference> {
  void runOnModule() override {
    auto module = getModule();
    auto versions = module.getAttrOfType<DictionaryAttr>("tf.versions");
    if (!versions) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "Missing 'tf.versions' attribute on the module, abort.\n";);
      return;
    }
    auto producer = versions.get("producer").dyn_cast<IntegerAttr>();
    if (!producer) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "Missing 'producer' attribute on the module, abort.\n";);
      return;
    }
    for (auto func : module.getOps<FuncOp>()) {
      InferShapeUntilFixPoint(&func.getBody(), producer.getInt());
    }

    if (auto main_func = module.lookupSymbol<mlir::FuncOp>("main")) {
      InferShapeForFunctionType(main_func);
    }
  }
};

PassRegistration<ShapeInference> pass(
    "tf-shape-inference", "Simple Shape Inference on TensorFlow Dialect");

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreateTFShapeInferencePass() {
  return std::make_unique<ShapeInference>();
}

}  // namespace TF
}  // namespace mlir
