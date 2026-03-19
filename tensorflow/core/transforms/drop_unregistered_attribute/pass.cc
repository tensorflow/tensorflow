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

#include "tensorflow/core/transforms/drop_unregistered_attribute/pass.h"

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_DROPOUTPUTSHAPESATTR
#include "tensorflow/core/transforms/passes.h.inc"

struct DropOutputShapesAttrPass
    : impl::DropOutputShapesAttrBase<DropOutputShapesAttrPass> {
  LogicalResult initialize(MLIRContext* context) override {
    for (auto& str : skip_) {
      skip_id.insert(StringAttr::get(context, str));
    }
    return success();
  }
  void runOnOperation() override {
    Operation* op = getOperation();
    op->walk([this](Operation* op) {
      if (!skip_id.count(op->getName().getIdentifier()))
        op->removeAttr("_output_shapes");
    });
  }

  // Set of operation types to skip over.
  DenseSet<StringAttr> skip_id;
};

}  // namespace

std::unique_ptr<Pass> CreateDropOutputShapesAttrPass() {
  return std::make_unique<DropOutputShapesAttrPass>();
}

}  // namespace tfg
}  // namespace mlir
