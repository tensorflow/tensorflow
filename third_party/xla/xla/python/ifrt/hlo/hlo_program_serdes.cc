/* Copyright 2023 The OpenXLA Authors.

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
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Serialization.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/serdes_week_4_old_version_accessor.h"

namespace xla {
namespace ifrt {

namespace {

// Library that provides stable serialization and deserialization of
// `xla::ifrt::HloProgram`. Both serialization and deserialization require
// linking in this library.
//
// Serialization:
// ```
// ASSIGN_OR_RETURN(Serialized serialized, Serialize(xla_program));
// ```
//
// Deserialization:
// ```
// ASSIGN_OR_RETURN(auto deserialized, Deserialize(serialized));
// auto xla_program = llvm::dyn_cast<HloProgram>(deserialized);
// ```

class HloProgramSerDes : public llvm::RTTIExtends<HloProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    // TODO(phawkins): whenever we next break compatibility, change this to
    // "xla::ifrt::HloProgram".
    return "xla::ifrt::XlaProgram";
  }

  absl::StatusOr<absl::Cord> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    // All serialization of `HloProgram` is pinned to a at-least-4-week-old
    // version. An acceptable IFRT SerDes version is [4-week-old, current].
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() <
        SerDesWeek4OldVersionAccessor::Get().version_number()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for HloProgram serialization"));
    }

    // Currently, PjRT-IFRT accepts an `HloProgram` that contains C/MHLO. Since
    // these dialects don't provide version compatibility, the following
    // converts the module into StableHLO and use its portable serialization.

    const auto& program = llvm::cast<HloProgram>(serializable);
    if (program.mlir_module() == nullptr) {
      return absl::InvalidArgumentError("Unable to serialize null MLIR module");
    }

    mlir::OwningOpRef<mlir::ModuleOp> module(
        llvm::cast<mlir::ModuleOp>(program.mlir_module()->clone()));

    // Allow mixed serialization for stablehlo dialects.
    std::string serialized;
    if (version.version_number() >= SerDesVersionNumber(3)) {
      ASSIGN_OR_RETURN(serialized,
                       xla::SerializeUsingVersionedStablehlo(
                           *module, xla::GetDefaultStablehloVersion(),
                           xla::GetDefaultSdyVersion(),
                           /*inplace=*/false,
                           /*allow_mixed_serialization=*/true));
    } else {
      ASSIGN_OR_RETURN(serialized,
                       xla::SerializeUsingVersionedStablehlo(
                           *module, xla::GetDefaultStablehloVersion(),
                           xla::GetDefaultSdyVersion()));
    }
    return absl::Cord(std::move(serialized));
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const absl::Cord& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    // MLIR context is created with threading disabled; otherwise, deserializing
    // many programs may end up creating too many threads.
    auto context = std::make_unique<mlir::MLIRContext>(
        mlir::MLIRContext::Threading::DISABLED);
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(context.get());

    mlir::DialectRegistry registry;
    mlir::sdy::registerAllDialects(registry);
    context->appendDialectRegistry(registry);

    // The MLIR Lexer (used by the text/assembly parser) assumes the input
    // buffer is null-terminated. To avoid ASan stack-buffer-overflows when
    // parsing text-based StableHLO, we copy the serialized data to a
    // `std::string` to guarantee null-termination before deserialization.
    std::string flat_str(serialized);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::stablehlo::deserializePortableArtifact(flat_str, context.get());
    if (!module) {
      const absl::Status status = diagnostic_handler.ConsumeStatus();
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to deserialize StableHLO module;\n\nDetailed "
                       "error from MLIR: ",
                       status.message()));
    }

    return std::make_unique<HloProgram>(std::move(context), std::move(module));
  }

  static char ID;  // NOLINT
};

char HloProgramSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_xla_program_serdes = ([]() {
  RegisterSerDes<HloProgram>(std::make_unique<HloProgramSerDes>());
}(), true);
// clang-format on

}  // namespace

}  // namespace ifrt
}  // namespace xla
