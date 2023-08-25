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

#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/test_passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace bisect {
namespace test {
namespace {

struct BreakLinalgTransposePass
    : public PassWrapper<BreakLinalgTransposePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BreakLinalgTransposePass)

  StringRef getArgument() const final { return "test-break-linalg-transpose"; }
  StringRef getDescription() const final { return "breaks linalg transpose"; }
  BreakLinalgTransposePass() = default;

  void runOnOperation() override {
    getOperation().walk([](linalg::TransposeOp op) {
      auto permutation = llvm::to_vector(op.getPermutation());
      std::swap(permutation[0], permutation[1]);
      op.setPermutation(permutation);
    });
  }
};
}  // namespace

void RegisterTestPasses() { PassRegistration<BreakLinalgTransposePass>(); }

}  // namespace test
}  // namespace bisect
}  // namespace mlir
