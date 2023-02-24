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
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir/utils/error_util.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tensorflow/tsl/platform/errors.h"

namespace py = pybind11;

namespace xla {
namespace {

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseModule(
    mlir::MLIRContext* context, std::string str) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::mhlo::MhloDialect>();
  context->loadDialect<mlir::chlo::ChloDialect>();
  context->loadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  context->loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(str.data(), str.size()), context);
  if (!module) {
    return FromAbslStatus(diagnostic_handler.ConsumeStatus());
  }
  if (failed(module->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    module->dump();
    return FromAbslStatus(diagnostic_handler.ConsumeStatus());
  }
  return module;
}

std::string PrintModule(mlir::ModuleOp module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(os, flags);
  return s;
}

void EnablePrintBeforeAndAfter(mlir::PassManager& pm) {
  auto print_before = [](mlir::Pass*, mlir::Operation*) { return true; };
  auto print_after = [](mlir::Pass*, mlir::Operation*) { return true; };
  pm.enableIRPrinting(print_before, print_after);
}

// Converts an XlaComputation to an MHLO or StableHLO mlir::Module string.
// Exists for backwards compatibility.
// TODO(phawkins): port remaining users of XlaComputations to use mlir::Modules
// instead and delete this function.
StatusOr<std::string> PyXlaComputationToMlirModule(
    const XlaComputation& computation, bool emit_stable_hlo) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  TF_RETURN_IF_ERROR(ConvertHloToMlirHlo(*module, &computation.proto(),
                                         /*import_all_computations=*/true));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  if (emit_stable_hlo) {
    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  }
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("MHLO => StableHLO failed");
  }
  return PrintModule(*module);
}

StatusOr<XlaComputation> PyMlirModuleToXlaComputation(std::string mlir_module,
                                                      bool use_tuple_args,
                                                      bool return_tuple) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseModule(&context, mlir_module));
  XlaComputation computation;
  TF_RETURN_IF_ERROR(
      MlirToXlaComputation(*module, computation, use_tuple_args, return_tuple));
  return computation;
}

StatusOr<std::string> PyMhloToStablehlo(std::string mlir_module) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  // JAX can be customized in a way that involves operations from custom
  // dialects showing up in JAX IR.
  // `ParseModule` won't know about these dialects, but that's fine since we
  // just want to convert MHLO ops to StableHLO ops here and leave everything
  // else unchanged.
  // In order to achieve that, we're allowing unregistered dialects here.
  context.allowUnregisteredDialects(true);
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseModule(&context, mlir_module));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("MHLO => StableHLO failed");
  }
  return PrintModule(*module);
}

StatusOr<std::string> PyStablehloToMhlo(std::string mlir_module) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  // See PyMhloToStablehlo for an explanation of why we're allowing unregistered
  // dialects here.
  context.allowUnregisteredDialects(true);
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseModule(&context, mlir_module));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("StableHLO => MHLO failed");
  }
  return PrintModule(*module);
}

}  // namespace

void BuildMlirSubmodule(py::module& m) {
  py::module mlir_module = m.def_submodule("mlir", "MLIR/XLA integration");

  mlir_module.def("xla_computation_to_mlir_module",
                  &PyXlaComputationToMlirModule, py::arg("computation"),
                  py::arg("emit_stable_hlo") = true);
  mlir_module.def("mlir_module_to_xla_computation",
                  &PyMlirModuleToXlaComputation, py::arg("mlir_module"),
                  py::arg("use_tuple_args") = false,
                  py::arg("return_tuple") = false);
  mlir_module.def("mhlo_to_stablehlo", &PyMhloToStablehlo,
                  py::arg("mlir_module"));
  mlir_module.def("stablehlo_to_mhlo", &PyStablehloToMhlo,
                  py::arg("mlir_module"));
}

}  // namespace xla
