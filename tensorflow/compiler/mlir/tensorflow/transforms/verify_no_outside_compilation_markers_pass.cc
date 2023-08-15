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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project

namespace mlir {
namespace TFDevice {

namespace {

#define GEN_PASS_DEF_VERIFYNOOUTSIDECOMPILATIONMARKERSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

class VerifyNoOutsideCompilationMarkersPass
    : public impl::VerifyNoOutsideCompilationMarkersPassBase<
          VerifyNoOutsideCompilationMarkersPass> {
 public:
  void runOnOperation() override;
};

void VerifyNoOutsideCompilationMarkersPass::runOnOperation() {}

}  // namespace

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
CreateVerifyNoOutsideCompilationMarkersPass() {
  return std::make_unique<VerifyNoOutsideCompilationMarkersPass>();
}

}  // namespace TFDevice
}  // namespace mlir
