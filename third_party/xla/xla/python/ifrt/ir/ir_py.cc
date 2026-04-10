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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "mlir/CAPI/IR.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {
namespace ifrt {

namespace {

absl::StatusOr<nb::bytes> SerializeVersionedProgram(
    MlirModule module, absl::string_view ifrt_ir_version,
    absl::string_view atom_program_version, bool version_in_place) {
  auto program = std::make_unique<IfrtIRProgram>(unwrap(module));
  TF_ASSIGN_OR_RETURN(
      Serialized serialized,
      Serialize(*program,
                std::make_unique<SerializeIfrtIRProgramOptions>(
                    std::string(ifrt_ir_version),
                    std::string(atom_program_version), version_in_place)));
  std::string serialized_str = serialized.SerializeAsString();
  return nb::bytes(serialized_str.data(), serialized_str.size());
}

absl::StatusOr<nb::bytes> SerializeVersionedProgram(
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
  std::string serialized_str = serialized.SerializeAsString();
  return nb::bytes(serialized_str.data(), serialized_str.size());
}

absl::StatusOr<std::string> DeserializeVersionedProgram(
    nb::bytes serialized_program, bool debug_info) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  Serialized serialized;
  absl::string_view serialized_program_view(serialized_program.c_str(),
                                            serialized_program.size());
  if (!serialized.ParseFromString(serialized_program_view)) {
    return absl::InvalidArgumentError(
        "Failed to parse serialized IFRT IR program.");
  }
  TF_ASSIGN_OR_RETURN(
      auto program,
      Deserialize<IfrtIRProgram>(
          serialized,
          std::make_unique<DeserializeIfrtIRProgramOptions>(&context)));
  return OperationToString(program->mlir_module,
                           mlir::OpPrintingFlags().enableDebugInfo(debug_info));
}

}  // namespace

NB_MODULE(ir_py, m) {
  nb::enum_<Version::CompatibilityRequirement>(m, "CompatibilityRequirement")
      .value("NONE", Version::CompatibilityRequirement::NONE)
      .value("WEEK_4", Version::CompatibilityRequirement::WEEK_4)
      .value("WEEK_12", Version::CompatibilityRequirement::WEEK_12)
      .value("MAX", Version::CompatibilityRequirement::MAX);

  m.def(
      "get_version_from_compatibility_requirement",
      [](Version::CompatibilityRequirement requirement) {
        return Version::fromCompatibilityRequirement(requirement).toString();
      },
      nb::arg("requirement"));

  m.def("get_current_version",
        []() { return Version::getCurrentVersion().toString(); });

  m.def("get_minimum_version",
        []() { return Version::getMinimumVersion().toString(); });

  // Serializes the IFRT IR program to a stable versioned format. The function
  // expects the IFRT IR program to already have atom programs outlined to
  // modules.
  m.def(
      "serialize_ifrt_ir_program",
      [](MlirModule module, absl::string_view ifrt_ir_version,
         absl::string_view atom_program_version,
         bool version_in_place) -> nb::bytes {
        return xla::ValueOrThrow(SerializeVersionedProgram(
            module, ifrt_ir_version, atom_program_version, version_in_place));
      },
      nb::arg("module"), nb::arg("ifrt_ir_version"),
      nb::arg("atom_program_version"), nb::arg("version_in_place"));
  m.def(
      "serialize_ifrt_ir_program",
      [](absl::string_view module_str, absl::string_view ifrt_ir_version,
         absl::string_view atom_program_version,
         bool version_in_place) -> nb::bytes {
        return xla::ValueOrThrow(
            SerializeVersionedProgram(module_str, ifrt_ir_version,
                                      atom_program_version, version_in_place));
      },
      nb::arg("module_str"), nb::arg("ifrt_ir_version"),
      nb::arg("atom_program_version"), nb::arg("version_in_place"));

  // Deserializes a versioned IFRT IR program to IFRT IR.
  m.def(
      "deserialize_ifrt_ir_program",
      [](nb::bytes serialized_program, bool debug_info) -> std::string {
        return xla::ValueOrThrow(
            DeserializeVersionedProgram(serialized_program, debug_info));
      },
      nb::arg("serialized_program"), nb::arg("debug_info"));
}

}  // namespace ifrt
}  // namespace xla
