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

#include <memory>

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_VERIFYSUITABLEFOREXPORTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class VerifySuitableForExportPass
    : public impl::VerifySuitableForExportPassBase<
          VerifySuitableForExportPass> {
 public:
  void runOnOperation() override {
    if (failed(tensorflow::VerifyExportSuitable(getOperation())))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateVerifySuitableForExportPass() {
  return std::make_unique<VerifySuitableForExportPass>();
}

}  // namespace TF
}  // namespace mlir
