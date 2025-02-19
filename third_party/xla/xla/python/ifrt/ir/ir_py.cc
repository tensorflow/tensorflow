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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/absl_casters.h"
#include "xla/pjrt/status_casters.h"  // IWYU pragma: keep; Needed for ValueOrThrow
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

absl::StatusOr<py::bytes> SerializedVersionedProgram(
    MlirModule module, absl::string_view ifrt_ir_version,
    absl::string_view atom_program_version, bool version_in_place) {
  auto program = std::make_unique<IfrtIRProgram>(unwrap(module));
  TF_ASSIGN_OR_RETURN(
      auto serialized,
      Serialize(*program,
                std::make_unique<SerializeIfrtIRProgramOptions>(
                    std::string(ifrt_ir_version),
                    std::string(atom_program_version), version_in_place)));
  // Return just the data, to avoid the dependency on the Serialized proto.
  return py::bytes(serialized.data());
}

absl::StatusOr<py::bytes> SerializedVersionedProgram(
    absl::string_view module_str, absl::string_view ifrt_ir_version,
    absl::string_view atom_program_version, bool version_in_place) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto module,
                      support::ParseMlirModuleString(module_str, context));
  auto program = std::make_unique<IfrtIRProgram>(module.release());
  TF_ASSIGN_OR_RETURN(
      auto serialized,
      Serialize(*program,
                std::make_unique<SerializeIfrtIRProgramOptions>(
                    std::string(ifrt_ir_version),
                    std::string(atom_program_version), version_in_place)));
  // Return just the data, to avoid the dependency on the Serialized proto.
  return py::bytes(serialized.data());
}

absl::StatusOr<mlir::ModuleOp> DeserializeVersionedProgram(
    mlir::MLIRContext* context, absl::string_view serialized_program) {
  xla::ifrt::Serialized serialized;
  serialized.set_type_name(std::string(IfrtIRProgram::type_name()));
  serialized.set_data(std::string(serialized_program));
  TF_ASSIGN_OR_RETURN(
      auto program,
      Deserialize<IfrtIRProgram>(
          serialized,
          std::make_unique<DeserializeIfrtIRProgramOptions>(context)));
  return std::move(program->mlir_module);
}

absl::StatusOr<py::bytes> DeserializeVersionedProgram(
    absl::string_view serialized_program) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  TF_ASSIGN_OR_RETURN(
      auto module, DeserializeVersionedProgram(&context, serialized_program));
  return py::bytes(
      OperationToString(module, mlir::OpPrintingFlags().enableDebugInfo(true)));
}

}  // namespace

PYBIND11_MODULE(ir_py, m) {
  py::enum_<Version::CompatibilityRequirement>(m, "CompatibilityRequirement")
      .value("NONE", Version::CompatibilityRequirement::NONE)
      .value("WEEK_4", Version::CompatibilityRequirement::WEEK_4)
      .value("WEEK_12", Version::CompatibilityRequirement::WEEK_12)
      .value("MAX", Version::CompatibilityRequirement::MAX);

  m.def(
      "get_version_from_compatibility_requirement",
      [](Version::CompatibilityRequirement requirement) {
        return Version::fromCompatibilityRequirement(requirement).toString();
      },
      py::arg("requirement"));

  m.def("get_current_version",
        []() { return Version::getCurrentVersion().toString(); });

  m.def("get_minimum_version",
        []() { return Version::getMinimumVersion().toString(); });

  // Serializes the IFRT IR program to a stable versioned format. The function
  // expects the IFRT IR program to already have atom programs outlined to
  // modules.
  m.def(
      "serialize_versioned_program",
      [](MlirModule module, absl::string_view ifrt_ir_version,
         absl::string_view atom_program_version,
         bool version_in_place) -> py::bytes {
        return xla::ValueOrThrow(SerializedVersionedProgram(
            module, ifrt_ir_version, atom_program_version, version_in_place));
      },
      py::arg("module"), py::arg("ifrt_ir_version"),
      py::arg("atom_program_version"), py::arg("version_in_place"));
  m.def(
      "serialize_versioned_program_str",
      [](absl::string_view module_str, absl::string_view ifrt_ir_version,
         absl::string_view atom_program_version,
         bool version_in_place) -> py::bytes {
        return xla::ValueOrThrow(
            SerializedVersionedProgram(module_str, ifrt_ir_version,
                                       atom_program_version, version_in_place));
      },
      py::arg("module_str"), py::arg("ifrt_ir_version"),
      py::arg("atom_program_version"), py::arg("version_in_place"));

  // Deserializes a versioned IFRT IR program to IFRT IR.
  m.def(
      "deserialize_versioned_program",
      [](MlirContext context,
         absl::string_view serialized_program) -> MlirModule {
        return wrap(xla::ValueOrThrow(
            DeserializeVersionedProgram(unwrap(context), serialized_program)));
      },
      py::arg("context"), py::arg("serialized_program"));
  m.def(
      "deserialize_versioned_program_str",
      [](absl::string_view serialized_program) -> py::bytes {
        return xla::ValueOrThrow(
            DeserializeVersionedProgram(serialized_program));
      },
      py::arg("serialized_program"));
}

}  // namespace ifrt
}  // namespace xla
