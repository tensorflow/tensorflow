/* Copyright 2021 The OpenXLA Authors.

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

#include "mhlo/transforms/passes.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "stablehlo/dialect/Serialization.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/refine_polymorphic_shapes.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {
namespace {

std::string PrintModule(mlir::ModuleOp module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(os, flags);
  return s;
}

absl::StatusOr<std::string> SerializeUsingBytecode(mlir::ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  if (mlir::failed(mlir::writeBytecodeToFile(module, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
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
absl::StatusOr<std::string> PyXlaComputationToMlirModule(
    const XlaComputation& computation, bool emit_stable_hlo) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&context));
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  context.appendDialectRegistry(registry);

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

absl::StatusOr<XlaComputation> PyMlirModuleToXlaComputation(
    absl::string_view mlir_module, bool use_tuple_args, bool return_tuple) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(mlir_module, context));
  XlaComputation computation;
  // SDY dialect may be part of the module which XLA doesn't know about.
  TF_RETURN_IF_ERROR(ExportShardyForHloRoundTrip(*module));
  TF_RETURN_IF_ERROR(MlirToXlaComputation(*module, computation, use_tuple_args,
                                          return_tuple,
                                          /*use_shardy=*/false));
  return computation;
}

absl::StatusOr<nb::bytes> PyMhloToStablehlo(absl::string_view mlir_module) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  // JAX can be customized in a way that involves operations from custom
  // dialects showing up in JAX IR.
  // `ParseMlirModuleString` won't know about these dialects, but that's fine
  // since we just want to convert MHLO ops to StableHLO ops here and leave
  // everything else unchanged.
  // In order to achieve that, we're allowing unregistered dialects here.
  context.allowUnregisteredDialects(true);
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(mlir_module, context));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("MHLO => StableHLO failed");
  }
  // Use bytecode, passing unregistered dialects with properties causes issues
  // when using textual assembly.
  TF_ASSIGN_OR_RETURN(std::string bytecode, SerializeUsingBytecode(*module));
  return nb::bytes(bytecode.data(), bytecode.size());
}

absl::StatusOr<nb::bytes> PyStablehloToMhlo(const nb::bytes& mlir_module) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  // See PyMhloToStablehlo for an explanation of why we're allowing unregistered
  // dialects here.
  context.allowUnregisteredDialects(true);
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      ParseMlirModuleString(
          absl::string_view(mlir_module.c_str(), mlir_module.size()), context));
  mlir::PassManager pm(&context);
  if (VLOG_IS_ON(3)) EnablePrintBeforeAndAfter(pm);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (!mlir::succeeded(pm.run(*module))) {
    return tsl::errors::InvalidArgument("StableHLO => MHLO failed");
  }

  // Use bytecode, passing unregistered dialects with properties causes issues
  // when using textual assembly.
  TF_ASSIGN_OR_RETURN(std::string bytecode, SerializeUsingBytecode(*module));
  return nb::bytes(bytecode.data(), bytecode.size());
}

absl::StatusOr<nb::bytes> PySerializePortableArtifact(
    absl::string_view mlir_module, absl::string_view target) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(mlir_module, context));

  // Serialize portable artifact
  TF_ASSIGN_OR_RETURN(
      std::string bytecode,
      SerializeUsingVersionedStablehlo(*module, target, /*inplace=*/true));
  return nb::bytes(bytecode.data(), bytecode.size());
}

absl::StatusOr<std::string> PyDeserializePortableArtifact(
    const nb::bytes& bytecode_str) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::stablehlo::deserializePortableArtifact(
          absl::string_view(bytecode_str.c_str(), bytecode_str.size()),
          &context);
  if (!module)
    return tsl::errors::InvalidArgument("Failed to deserialize StableHLO");
  return PrintModule(*module);
}

}  // namespace

void BuildMlirSubmodule(nb::module_& m) {
  nb::module_ mlir_module = m.def_submodule("mlir", "MLIR/XLA integration");

  mlir_module.def("xla_computation_to_mlir_module",
                  xla::ValueOrThrowWrapper(PyXlaComputationToMlirModule),
                  nb::arg("computation"), nb::arg("emit_stable_hlo") = true);
  mlir_module.def(
      "mlir_module_to_xla_computation",
      [](const nb::bytes& bytecode, bool use_tuple_args, bool return_tuple) {
        return xla::ValueOrThrow(PyMlirModuleToXlaComputation(
            absl::string_view(bytecode.c_str(), bytecode.size()),
            use_tuple_args, return_tuple));
      },
      nb::arg("mlir_module"), nb::arg("use_tuple_args") = false,
      nb::arg("return_tuple") = false);
  mlir_module.def("mlir_module_to_xla_computation",
                  xla::ValueOrThrowWrapper(PyMlirModuleToXlaComputation),
                  nb::arg("mlir_module"), nb::arg("use_tuple_args") = false,
                  nb::arg("return_tuple") = false);
  mlir_module.def(
      "mhlo_to_stablehlo",
      [](const nb::bytes& bytecode) {
        return xla::ValueOrThrow(PyMhloToStablehlo(
            absl::string_view(bytecode.c_str(), bytecode.size())));
      },
      nb::arg("mlir_module"));
  mlir_module.def("mhlo_to_stablehlo",
                  xla::ValueOrThrowWrapper(PyMhloToStablehlo),
                  nb::arg("mlir_module"));
  mlir_module.def("stablehlo_to_mhlo",
                  xla::ValueOrThrowWrapper(PyStablehloToMhlo),
                  nb::arg("mlir_module"));
  mlir_module.def(
      "serialize_portable_artifact",
      [](const nb::bytes& bytecode, absl::string_view target) {
        return xla::ValueOrThrow(PySerializePortableArtifact(
            absl::string_view(bytecode.c_str(), bytecode.size()), target));
      },
      nb::arg("mlir_module"), nb::arg("target"));
  mlir_module.def("serialize_portable_artifact",
                  xla::ValueOrThrowWrapper(PySerializePortableArtifact),
                  nb::arg("mlir_module"), nb::arg("target"));
  mlir_module.def("deserialize_portable_artifact",
                  xla::ValueOrThrowWrapper(PyDeserializePortableArtifact),
                  nb::arg("mlir_module"));
  mlir_module.def(
      "refine_polymorphic_shapes",
      [](nb::bytes bytecode, bool enable_shape_assertions,
         bool validate_static_shapes) -> nb::bytes {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        xla::ThrowIfError(RefinePolymorphicShapes(
            absl::string_view(bytecode.c_str(), bytecode.size()), os,
            enable_shape_assertions, validate_static_shapes));
        return nb::bytes(buffer.data(), buffer.size());
      },
      nb::arg("mlir_module"), nb::arg("enable_shape_assertions") = true,
      nb::arg("validate_static_shapes") = true,
      R"(Refines the dynamic shapes for a module.
        The "main" function must have static shapes and all the
        intermediate dynamic shapes depend only on the input static
        shapes. Optionally, also validates that the resulting module has
        only static shapes.
      )");
}

}  // namespace xla
