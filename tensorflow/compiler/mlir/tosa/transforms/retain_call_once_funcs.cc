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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "retain-call-once-funcs"
#define DEBUG_TYPE PASS_NAME

namespace mlir::tosa {

#define GEN_PASS_DEF_RETAINCALLONCEFUNCS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

namespace {

class RetainCallOnceFuncsPass
    : public impl::RetainCallOnceFuncsBase<RetainCallOnceFuncsPass> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    llvm::DenseMap<StringRef, func::FuncOp> funcMap;
    for (auto func : moduleOp.getOps<mlir::func::FuncOp>()) {
      funcMap[func.getSymName()] = func;
    }

    for (auto func : moduleOp.getOps<mlir::func::FuncOp>()) {
      for (auto callOnce : func.getOps<mlir::TFL::CallOnceOp>()) {
        auto callFunc = funcMap[callOnce.getSessionInitFunction()];
        callOnce->setAttr("session_init_function_symbol",
                          SymbolRefAttr::get(callFunc));
      }
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createRetainCallOnceFuncsPass() {
  return std::make_unique<RetainCallOnceFuncsPass>();
}

}  // namespace mlir::tosa
