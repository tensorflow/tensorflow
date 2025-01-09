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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Serialization.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/serdes.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

// Library that provides stable serialization and deserialization of
// `xla::ifrt::HloProgram`. Both serialization and deserialization require
// linking in this library.
//
// Serialization:
// ```
// TF_ASSIGN_OR_RETURN(Serialized serialized, Serialize(xla_program));
// ```
//
// Deserialization:
// ```
// TF_ASSIGN_OR_RETURN(auto deserialized, Deserialize(serialized));
// auto xla_program = llvm::dyn_cast<HloProgram>(deserialized);
// ```

class HloProgramSerDes : public llvm::RTTIExtends<HloProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    // TODO(phawkins): whenever we next break compatibility, change this to
    // "xla::ifrt::HloProgram".
    return "xla::ifrt::XlaProgram";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable, std::unique_ptr<SerializeOptions>) override {
    // Currently, PjRT-IFRT accepts an `HloProgram` that contains C/MHLO. Since
    // these dialects don't provide version compatibility, the following
    // converts the module into StableHLO and use its portable serialization.

    const auto& program = llvm::cast<HloProgram>(serializable);
    if (program.mlir_module == nullptr) {
      return absl::InvalidArgumentError("Unable to serialize null MLIR module");
    }

    mlir::OwningOpRef<mlir::ModuleOp> module(
        llvm::cast<mlir::ModuleOp>(program.mlir_module->clone()));

    // Serialize portable artifact.
    TF_ASSIGN_OR_RETURN(std::string serialized,
                        xla::SerializeUsingVersionedStablehlo(
                            *module, xla::GetDefaultStablehloVersion()));
    return serialized;
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    // MLIR context is created with threading disabled; otherwise, deserializing
    // many programs may end up creating too many threads.
    auto context = std::make_unique<mlir::MLIRContext>(
        mlir::MLIRContext::Threading::DISABLED);
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(context.get());

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::stablehlo::deserializePortableArtifact(serialized, context.get());
    if (!module) {
      const absl::Status status = diagnostic_handler.ConsumeStatus();
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to deserialize StableHLO module;\n\nDetailed "
                       "error from MLIR: ",
                       status.message()));
    }

    // Convert StableHLO back to MHLO to keep the contract the same before and
    // after a serialization/deserialization round trip.
    mlir::PassManager pm(context.get());
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    if (!mlir::succeeded(pm.run(*module))) {
      const absl::Status status = diagnostic_handler.ConsumeStatus();
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to legalize StableHLO to MHLO;\n\nDetailed error from MLIR: ",
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
