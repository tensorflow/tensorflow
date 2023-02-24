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

#include <memory>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_STRIPTFATTRIBUTESPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct StripTfAttributesPass
    : public impl::StripTfAttributesPassBase<StripTfAttributesPass> {
  void runOnOperation() override;
};

bool ShouldStripAttr(NamedAttribute &namedAttr) {
  StringRef name = namedAttr.getName().strref();
  if (name.startswith("tf.") || name.startswith("tf_")) return true;
  StringRef value = namedAttr.getValue().getDialect().getNamespace();
  return value == "tf" || value.startswith("tf_");
}

void StripFunction(func::FuncOp func) {
  auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
      func->getAttrs(),
      [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
  for (auto namedAttr : stripAttrs) {
    func->removeAttr(namedAttr.getName());
  }

  for (int i = 0; i < func.getNumArguments(); ++i) {
    llvm::ArrayRef<mlir::NamedAttribute> attrs =
        mlir::function_interface_impl::getArgAttrs(func, i);
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        attrs,
        [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      func.removeArgAttr(i, namedAttr.getName());
    }
  }

  for (int i = 0; i < func.getNumResults(); ++i) {
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        func.getResultAttrs(i),
        [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      func.removeResultAttr(i, namedAttr.getName());
    }
  }
}

void StripTfAttributesPass::runOnOperation() {
  ModuleOp module = getOperation();

  // strip module itself
  auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
      module->getAttrs(),
      [](NamedAttribute namedAttr) { return ShouldStripAttr(namedAttr); }));
  for (auto namedAttr : stripAttrs) {
    module->removeAttr(namedAttr.getName());
  }

  // strip functions in module
  module.walk([&](func::FuncOp func) { StripFunction(func); });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateStripTfAttributesPass() {
  return std::make_unique<StripTfAttributesPass>();
}

}  // namespace TF
}  // namespace mlir
