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

#include "tensorflow/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"

#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

PYBIND11_MODULE(mlir_wrapper, m) {
  m.def("preloadTensorFlowDialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::RegisterAllTensorFlowDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("verify", [](std::string input) {
    llvm::SourceMgr SM = llvm::SourceMgr();
    SM.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(input),
                          llvm::SMLoc());
    mlir::DialectRegistry registry;
    mlir::RegisterAllTensorFlowDialects(registry);
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(SM, &ctx);
    if (!module) {
      return false;
    }
    if (failed(mlir::verify(*module))) {
      module->emitError("Invalid MLIR module: failed verification.");
      return false;
    }
    return true;
  });

  init_basic_classes(m);
  init_types(m);
  init_builders(m);
  init_ops(m);
  init_attrs(m);
}
