/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Legalize TensorFlow Lite to Tensor

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensor/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tensor/transforms/passes.h"


namespace mlir {
namespace tensor {

namespace {

#define GEN_PASS_DEF_TENSORLEGALIZETFLPASS
#include "tensorflow/compiler/mlir/tensor/transforms/passes.h.inc"

class LegalizeTFL : public impl::TensorLegalizeTFLPassBase<LegalizeTFL> {
public:
  LegalizeTFL() = default;
  void runOnOperation() override;
};

void LegalizeTFL::runOnOperation() {
}

}  // namespace

void populateLegalizeTFLPatterns(MLIRContext* ctx,
                                 RewritePatternSet& patterns) {
}

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFLPass() {
  return std::make_unique<LegalizeTFL>();
}

}  // namespace tensor
}  // namespace mlir

