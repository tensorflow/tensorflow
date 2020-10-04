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
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#define DEBUG_TYPE "tf-shape-inference"

namespace mlir {
namespace TF {

namespace {

// This transformation pass propagate shapes on the TensorFlow graph.
// It is a ModulePass in order to be able to change function types.
class ShapeInference
    : public PassWrapper<ShapeInference, OperationPass<ModuleOp>> {
 public:
  ShapeInference() = default;
  ShapeInference(const ShapeInference& that) {
    propagate_caller_callee_constants_ =
        that.propagate_caller_callee_constants_;
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto producer_or = tensorflow::GetTfGraphProducerVersion(module);
    if (!producer_or.ok()) {
      LLVM_DEBUG(llvm::dbgs() << producer_or.status().ToString(););
      return;
    }
    int64_t producer = producer_or.ValueOrDie();
    for (auto func : module.getOps<FuncOp>()) {
      if (failed(InferShapeForFunction(func, /*arg_shapes=*/{}, producer,
                                       propagate_caller_callee_constants_)))
        return signalPassFailure();
    }
  }

 private:
  Option<bool> propagate_caller_callee_constants_{
      *this, "propagate-caller-callee-constants",
      llvm::cl::desc("Propagate constants between callers and callees"),
      llvm::cl::init(true)};
};

PassRegistration<ShapeInference> pass(
    "tf-shape-inference", "Simple Shape Inference on TensorFlow Dialect");

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTFShapeInferencePass() {
  return std::make_unique<ShapeInference>();
}

}  // namespace TF
}  // namespace mlir
