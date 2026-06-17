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

#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_STRIPNOINLINEATTRIBUTEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// This tranformation pass strips any "_noinline" attributes from the module.
struct StripNoinlineAttributePass
    : public impl::StripNoinlineAttributePassBase<StripNoinlineAttributePass> {
 public:
  // void runOnOperation() override;
  void runOnOperation() override {
    // Strip the "tf._noinline" attribute from top-level functions.
    for (auto func_op : getOperation().getOps<func::FuncOp>())
      func_op->removeAttr("tf._noinline");
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateStripNoinlineAttributePass() {
  return std::make_unique<StripNoinlineAttributePass>();
}

}  // namespace TF
}  // namespace mlir
