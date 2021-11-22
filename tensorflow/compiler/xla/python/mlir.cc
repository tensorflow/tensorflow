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

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/status.h"

namespace py = pybind11;

namespace xla {

// Converts an XlaComputation to an MHLO mlir::Module string. Exists for
// backwards compatibility.
// TODO(phawkins): port remaining users of XlaComputations to use mlir::Modules
// instead and delete this function.
StatusOr<std::string> XlaComputationToMlirModule(
    const XlaComputation& computation) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  context.loadDialect<mlir::StandardOpsDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  TF_RETURN_IF_ERROR(ConvertHloToMlirHlo(*module, &computation.proto(),
                                         /*import_all_computations=*/true));
  std::string s;
  llvm::raw_string_ostream os(s);
  module->print(os);
  return s;
}

void BuildMlirSubmodule(py::module& m) {
  py::module mlir_module = m.def_submodule("mlir", "MLIR/XLA integration");

  mlir_module.def("xla_computation_to_mlir_module",
                  &XlaComputationToMlirModule);
}

}  // namespace xla
