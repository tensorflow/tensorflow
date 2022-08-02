/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for translating mixed IR to buffer form.

#include <memory>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/PassDetail.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct KernelgenFinalBufferizePass
    : public KernelgenFinalBufferizePassBase<KernelgenFinalBufferizePass> {
  // Default alignment_ specified in passes.td
  KernelgenFinalBufferizePass() = default;

  void runOnOperation() override {
    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::createFinalBufferizePass(/*alignment=*/64,
                                              populateExtraBufferizeDialects,
                                              populateExtraBufferizePatterns));
    (void)runPipeline(pm, getOperation());
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateKernelgenFinalBufferizePass() {
  return std::make_unique<KernelgenFinalBufferizePass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
