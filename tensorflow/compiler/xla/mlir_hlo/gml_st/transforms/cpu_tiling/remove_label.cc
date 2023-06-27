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

#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_REMOVELABELPASS
#include "gml_st/transforms/passes.h.inc"

struct RemoveLabelPass : public impl::RemoveLabelPassBase<RemoveLabelPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk(
        [](Operation *op) { removeLabel(op, kTransformedLabel); });
  }
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRemoveLabelPass() {
  return std::make_unique<mlir::gml_st::RemoveLabelPass>();
}

}  // namespace mlir::gml_st
