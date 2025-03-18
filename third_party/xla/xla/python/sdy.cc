/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/sdy.h"

#include <cassert>
#include <string>

#include "mhlo/transforms/passes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/status_casters.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"

namespace nb = nanobind;

namespace xla {

namespace {

absl::StatusOr<std::string> SerializeUsingBytecode(mlir::ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  if (mlir::failed(mlir::writeBytecodeToFile(module, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

}  // namespace

void BuildSdySubmodule(nb::module_& m) {
  nb::module_ mlir_module = m.def_submodule("sdy", "Shardy/XLA integration");

  mlir_module
      // TODO(b/707574930): define a C API for the XLA pipelines.
      .def(
          "sdy_round_trip_export_pipeline",
          [](const nb::bytes& bytecode) -> nb::bytes {
            mlir::MLIRContext context;
            mlir::OwningOpRef<mlir::ModuleOp> module =
                xla::ValueOrThrow(ParseMlirModuleString(
                    absl::string_view(bytecode.c_str(), bytecode.size()),
                    context));
            mlir::PassManager pm(&context);
            sdy::addSdyRoundTripExportPipeline(pm);
            tsl::StatusScopedDiagnosticHandler diagnosticHandler(&context);
            ThrowIfError(diagnosticHandler.consumeStatus(pm.run(module.get())));
            std::string module_str =
                xla::ValueOrThrow(SerializeUsingBytecode(module.get()));
            return nb::bytes(module_str.data(), module_str.size());
          },
          nb::arg("module"))
      .def(
          "sdy_round_trip_import_shardings",
          [](const nb::bytes& bytecode) -> nb::bytes {
            mlir::MLIRContext context;
            mlir::OwningOpRef<mlir::ModuleOp> module =
                xla::ValueOrThrow(ParseMlirModuleString(
                    absl::string_view(bytecode.c_str(), bytecode.size()),
                    context));
            mlir::PassManager pm(&context);
            pm.addPass(xla::sdy::createSdyRoundTripImportShardyAttrsPass());
            tsl::StatusScopedDiagnosticHandler diagnosticHandler(&context);
            ThrowIfError(diagnosticHandler.consumeStatus(pm.run(module.get())));
            std::string module_str =
                xla::ValueOrThrow(SerializeUsingBytecode(module.get()));
            return nb::bytes(module_str.data(), module_str.size());
          },
          nb::arg("module"))
      .def("lowered_with_shardy",
           [](const nb::bytes& bytecode) -> bool {
             mlir::MLIRContext context;
             mlir::OwningOpRef<mlir::ModuleOp> module =
                 xla::ValueOrThrow(ParseMlirModuleString(
                     absl::string_view(bytecode.c_str(), bytecode.size()),
                     context));
             return mlir::sdy::getMeshAttr(module.get(), "mesh") ||
                    sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(
                        module.get(), sdy::kMeshesRoundTripAttr)
                        .has_value();
           })
      // TODO(bartchr): delete this and all uses of it once I have JAX export
      // support multiple meshes.
      .def("get_mesh", [](const nb::bytes& bytecode) -> nb::list {
        mlir::MLIRContext context;
        mlir::OwningOpRef<mlir::ModuleOp> module =
            xla::ValueOrThrow(ParseMlirModuleString(
                absl::string_view(bytecode.c_str(), bytecode.size()), context));
        auto mesh_op =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::sdy::MeshOp>(
                module.get(), mlir::StringAttr::get(&context, "mesh"));
        if (!mesh_op) {
          return {};
        }
        nb::list mesh_shape;
        for (auto axis : mesh_op.getMeshAttr().getAxes()) {
          mesh_shape.append(
              nb::make_tuple(axis.getName().str(), axis.getSize()));
        }
        return mesh_shape;
      });
}

}  // namespace xla
