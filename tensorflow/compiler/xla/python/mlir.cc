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

#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/status.h"

namespace py = pybind11;

namespace xla {
namespace {

// Converts an XlaComputation to an MHLO mlir::Module string. Exists for
// backwards compatibility.
// TODO(phawkins): port remaining users of XlaComputations to use mlir::Modules
// instead and delete this function.
StatusOr<std::string> PyXlaComputationToMlirModule(
    const XlaComputation& computation) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  TF_RETURN_IF_ERROR(ConvertHloToMlirHlo(*module, &computation.proto(),
                                         /*import_all_computations=*/true));
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(os, flags);
  return s;
}

StatusOr<XlaComputation> PyMlirModuleToXlaComputation(std::string mlir_module,
                                                      bool use_tuple_args,
                                                      bool return_tuple) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(mlir_module.data(), mlir_module.size()), &context);
  if (!module) {
    return diagnostic_handler.ConsumeStatus();
  }
  if (failed(module->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    module->dump();
    return diagnostic_handler.ConsumeStatus();
  }

  XlaComputation computation;
  TF_RETURN_IF_ERROR(
      MlirToXlaComputation(*module, computation, use_tuple_args, return_tuple));
  return computation;
}

}  // namespace

void BuildMlirSubmodule(py::module& m) {
  py::module mlir_module = m.def_submodule("mlir", "MLIR/XLA integration");

  mlir_module.def("xla_computation_to_mlir_module",
                  &PyXlaComputationToMlirModule);
  mlir_module.def("mlir_module_to_xla_computation",
                  &PyMlirModuleToXlaComputation, py::arg("mlir_module"),
                  py::arg("use_tuple_args") = false,
                  py::arg("return_tuple") = false);
}

}  // namespace xla
