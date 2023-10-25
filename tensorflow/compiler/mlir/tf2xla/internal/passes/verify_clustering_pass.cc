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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

#define GEN_PASS_DEF_VERIFYCLUSTERINGPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

class VerifyClusteringPass
    : public impl::VerifyClusteringPassPassBase<VerifyClusteringPass> {
 public:
  void runOnOperation() override;
};

void VerifyClusteringPass::runOnOperation() {}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateVerifyClusteringPass() {
  return std::make_unique<VerifyClusteringPass>();
}
}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
