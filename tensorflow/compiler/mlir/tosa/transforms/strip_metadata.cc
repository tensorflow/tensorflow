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

#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-strip-metadata"
#define DEBUG_TYPE PASS_NAME

namespace mlir::tosa {

#define GEN_PASS_DEF_STRIPM
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

namespace {

static bool isTFLAttr(NamedAttribute &namedAttr) {
  // TFLite uses both tf and tfl in attribute annotations.
  auto name = namedAttr.getName().strref();
  // Don't trim attributes from tf_saved_model---they carry ABI information.
  if (name.starts_with("tf_saved_model.")) return false;

  if (name.starts_with("tf.") || name.starts_with("tf_") ||
      name.starts_with("tfl.") || name.starts_with("tfl_")) {
    return true;
  }
  StringRef attrNamespace = namedAttr.getValue().getDialect().getNamespace();
  return attrNamespace == "tf" || attrNamespace == "tfl";
}

class StripModuleMetadataPass
    : public StripModuleMetadataBase<StripModuleMetadataPass> {
 public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        moduleOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      moduleOp->removeAttr(namedAttr.getName());
    }
  }
};

class StripFunctionMetadataPass
    : public StripFunctionMetadataBase<StripFunctionMetadataPass> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
        funcOp->getAttrs(),
        [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
    for (auto namedAttr : stripAttrs) {
      funcOp->removeAttr(namedAttr.getName());
    }

    for (int i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          mlir::function_interface_impl::getArgAttrs(funcOp, i),
          [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeArgAttr(i, namedAttr.getName());
      }
    }

    for (int i = 0, e = funcOp.getNumResults(); i < e; ++i) {
      auto stripAttrs = llvm::to_vector<4>(llvm::make_filter_range(
          mlir::function_interface_impl::getResultAttrs(funcOp, i),
          [](NamedAttribute namedAttr) { return isTFLAttr(namedAttr); }));
      for (auto namedAttr : stripAttrs) {
        funcOp.removeResultAttr(i, namedAttr.getName());
      }
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass() {
  return std::make_unique<StripModuleMetadataPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createStripFunctionMetadataPass() {
  return std::make_unique<StripFunctionMetadataPass>();
}

}  // namespace mlir::tosa
