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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms/Transforms.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_DEF_FUSEINNERPARALLELLOOPSPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct FuseInnerParallelLoopsPass
    : impl::FuseInnerParallelLoopsPassBase<FuseInnerParallelLoopsPass> {
  void runOnOperation() override {
    getOperation().walk([](mlir::scf::ParallelOp op) {
      mlir::scf::naivelyFuseParallelOps(op.getRegion());
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFuseInnerParallelLoopsPass() {
  return std::make_unique<FuseInnerParallelLoopsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
