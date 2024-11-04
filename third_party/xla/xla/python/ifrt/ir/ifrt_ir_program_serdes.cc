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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
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
    return "xla::ifrt::IfrtIRProgram";
  }

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const auto& program = llvm::cast<IfrtIRProgram>(serializable);
    if (program.mlir_module == nullptr) {
      return absl::InvalidArgumentError("Unable to serialize null MLIR module");
    }
    std::string serialized;
    llvm::raw_string_ostream out(serialized);
    mlir::BytecodeWriterConfig config;
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(
        program.mlir_module->getContext());
    if (mlir::failed(
            mlir::writeBytecodeToFile(program.mlir_module, out, config))) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to serialize IFRT IR module string: %s",
                          diagnostic_handler.ConsumeStatus().message()));
    }
    return serialized;
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    auto context = std::make_unique<mlir::MLIRContext>();
    TF_ASSIGN_OR_RETURN(auto module,
                        support::ParseMlirModuleString(serialized, *context));
    return std::make_unique<IfrtIRProgram>(std::move(context),
                                           std::move(module));
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

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
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
