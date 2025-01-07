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

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "xla/status_macros.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

// Serialization/deserialization for `IfrtIRProgram`.
class IfrtIRProgramSerDes
    : public llvm::RTTIExtends<IfrtIRProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return IfrtIRProgram::type_name();
  }

  // Serializes the `IfrtIRProgram`.
  //
  // If no `options` are provided, the program is serialize to a non-stable
  // representation. Otherwise, if `options` are provided the program is
  // serialized to a stable versioned IFRT IR representation, and the atom
  // program modules are serialized to VHLO.
  absl::StatusOr<std::string> Serialize(
      Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    auto& program = llvm::cast<IfrtIRProgram>(serializable);
    if (program.mlir_module == nullptr) {
      return absl::InvalidArgumentError("Unable to serialize null MLIR module");
    }

    IfrtIrProgramProto program_proto;
    llvm::raw_string_ostream ifrt_ir_program_stream(
        *program_proto.mutable_ifrt_program());
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(
        program.mlir_module->getContext());

    const auto* serialize_options =
        llvm::cast_or_null<SerializeIfrtIRProgramOptions>(options.get());
    if (serialize_options == nullptr) {
      // Serialize to bytecode the whole program if no options are provided.
      // This is a fast path for the case where the user does not care about
      // stable serialization.
      mlir::BytecodeWriterConfig writer_config;
      if (mlir::failed(mlir::writeBytecodeToFile(
              program.mlir_module, ifrt_ir_program_stream, writer_config))) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Failed to serialize IFRT IR module string: %s",
                            diagnostic_handler.ConsumeStatus().message()));
      }
    } else {
      program_proto.set_ifrt_version(serialize_options->ifrt_version);
      mlir::OwningOpRef<mlir::ModuleOp> cloned;
      mlir::ModuleOp mlir_module;
      if (serialize_options->version_in_place) {
        mlir_module = program.mlir_module;
      } else {
        cloned = program.mlir_module.clone();
        mlir_module = *cloned;
      }
      // Run the pipeline to convert IFRT IR program to a versioned artifact.
      mlir::PassManager pm(mlir_module->getContext());
      CreateIfrtToVersionedPipeline(pm, serialize_options->ifrt_version,
                                    serialize_options->atom_program_version,
                                    program_proto);
      if (mlir::failed(pm.run(mlir_module))) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Failed to version IFRT IR program: %s",
                            diagnostic_handler.ConsumeStatus().message()));
      }

      // Serialize the versioned IFRT IR program to bytecode.
      auto fail_or_bytecode_version =
          Version::fromString(serialize_options->ifrt_version)
              ->getBytecodeVersion();
      if (mlir::failed(fail_or_bytecode_version)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Failed to get IFRT IR bytecode version for IR version %s",
            serialize_options->ifrt_version));
      }
      std::string bytecode_version_string =
          absl::StrCat("IFRT_v", serialize_options->ifrt_version);
      mlir::BytecodeWriterConfig writer_config(bytecode_version_string);
      writer_config.setDesiredBytecodeVersion(*fail_or_bytecode_version);
      if (mlir::failed(mlir::writeBytecodeToFile(
              mlir_module, ifrt_ir_program_stream, writer_config))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Failed to serialize versioned IFRT IR module string: %s",
            diagnostic_handler.ConsumeStatus().message()));
      }
    }
    return program_proto.SerializeAsString();
  }

  // Deserializes an `IfrtIRProgram`.
  //
  // If the serialized program was versioned then this method will attempt to
  // deserialize IFRT IR and the VHLO atom program to the current version of
  // IFRT IR, respectively StableHLO. An error is returned if the serialized
  // IFRT IR versions or VHLO version are outside of the compatibility window.
  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    const auto* deserialize_options =
        llvm::dyn_cast_or_null<DeserializeIfrtIRProgramOptions>(options.get());
    bool use_existing_context = false;
    std::unique_ptr<mlir::MLIRContext> context;
    if (!deserialize_options || !deserialize_options->context) {
      context = std::make_unique<mlir::MLIRContext>();
    } else {
      use_existing_context = true;
      context =
          std::unique_ptr<mlir::MLIRContext>(deserialize_options->context);
    }
    absl::Cleanup release_context_pointer = [&]() {
      if (use_existing_context) {
        // Release the pointer s.t. the existing context is not freed.
        context.release();
      }
    };

    IfrtIrProgramProto program_proto;
    if (!program_proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError("Failed to parse IfrtIrProgramProto");
    }
    TF_ASSIGN_OR_RETURN(
        auto module,
        support::ParseMlirModuleString(program_proto.ifrt_program(), *context));

    if (program_proto.ifrt_version().empty()) {
      // The program was not versioned on serialization. The whole IFRT IR
      // program was serialized to bytecode.
      if (use_existing_context) {
        return std::make_unique<IfrtIRProgram>(module.release());
      } else {
        return std::make_unique<IfrtIRProgram>(std::move(context),
                                               std::move(module));
      }
    } else {
      // Run the pipeline to convert a versioned IFRT IR program artifact to
      // an IFRT IR program.
      mlir::BaseScopedDiagnosticHandler diagnostic_handler(context.get());
      mlir::PassManager pm(context.get());
      CreateIfrtFromVersionedPipeline(pm, program_proto);
      if (mlir::failed(pm.run(*module))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Failed to deserialize versioned IFRT IR program: %s",
            diagnostic_handler.ConsumeStatus().message()));
      }

      if (use_existing_context) {
        return std::make_unique<IfrtIRProgram>(module.release());
      } else {
        return std::make_unique<IfrtIRProgram>(std::move(context),
                                               std::move(module));
      }
    }
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `IfrtIRCompileOptions`.
class IfrtIRCompileOptionsSerDes
    : public llvm::RTTIExtends<IfrtIRCompileOptionsSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::IfrtIRCompileOptions";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable, std::unique_ptr<SerializeOptions>) override {
    const auto& options = llvm::cast<IfrtIRCompileOptions>(serializable);
    TF_ASSIGN_OR_RETURN(IfrtIrCompileOptionsProto options_proto,
                        options.ToProto());
    return options_proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    IfrtIrCompileOptionsProto options_proto;
    TF_RET_CHECK(options_proto.ParseFromString(serialized))
        << "Failed to parse IfrtIrCompileOptionsProto";
    return IfrtIRCompileOptions::FromProto(options_proto);
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char IfrtIRCompileOptionsSerDes::ID = 0;  // NOLINT
[[maybe_unused]] char IfrtIRProgramSerDes::ID = 0;         // NOLINT

// clang-format off
bool register_ifrt_ir_program_serdes = ([]() {
  RegisterSerDes<IfrtIRProgram>(std::make_unique<IfrtIRProgramSerDes>());
}(), true);

bool register_ifrt_ir_compile_options_serdes = ([]() {
  RegisterSerDes<IfrtIRCompileOptions>(
      std::make_unique<IfrtIRCompileOptionsSerDes>());
}(), true);
// clang-format on

}  // namespace

}  // namespace ifrt
}  // namespace xla
