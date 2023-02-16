/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_TPUVALIDATEINPUTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct TPUValidateInputsPass
    : public impl::TPUValidateInputsPassBase<TPUValidateInputsPass> {
  void runOnOperation() override;
};

void TPUValidateInputsPass::runOnOperation() {}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUValidateInputsPass() {
  return std::make_unique<TPUValidateInputsPass>();
}

}  // namespace TFTPU
}  // namespace mlir
