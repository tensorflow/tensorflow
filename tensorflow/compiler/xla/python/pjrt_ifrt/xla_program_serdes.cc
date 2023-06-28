/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/pjrt_ifrt/xla_program_serdes.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/api/PortableApi.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/python/ifrt/serdes.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/xla_compiler.h"

namespace xla {
namespace ifrt {

namespace {

class XlaProgramSerDes : public llvm::RTTIExtends<XlaProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::XlaProgram";
  }

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    // Currently, PjRT-IFRT accepts an `XlaProgram` that contains C/MHLO. Since
    // these dialects don't provide version compatibility, the following
    // converts the module into StableHLO and use its portable serialization.

    const auto& program = llvm::cast<XlaProgram>(serializable);
    mlir::OwningOpRef<mlir::ModuleOp> module(
        llvm::cast<mlir::ModuleOp>(program.mlir_module->clone()));

    mlir::PassManager pm(module->getContext());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createChloLegalizeToHloPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createShapeLegalizeToHloPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
    if (!mlir::succeeded(pm.run(*module))) {
      return absl::InvalidArgumentError(
          "CHLO => [MHLO+Shape] => StableHLO failed");
    }

    // Serialize portable artifact.
    std::string serialized;
    llvm::raw_string_ostream os(serialized);
    if (mlir::failed(mlir::stablehlo::serializePortableArtifact(
            *module, mlir::stablehlo::getCurrentVersion(), os))) {
      return absl::InvalidArgumentError("Failed to serialize StableHLO");
    }
    return serialized;
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    auto* xla_program_deserialize_options =
        llvm::dyn_cast_or_null<XlaDeserializeProgramOptions>(options.get());
    if (xla_program_deserialize_options == nullptr) {
      return absl::InvalidArgumentError(
          "Deserializing XlaProgram requires "
          "`xla::ifrt::XlaDeserializeProgramOptions` to be passed to "
          "`Deserialize()` as "
          "`xla::ifrt::DeserializeOptions`");
    }

    mlir::MLIRContext* context = xla_program_deserialize_options->mlir_context;
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::stablehlo::deserializePortableArtifact(serialized, context);

    // Convert StableHLO back to MHLO to keep the contract the same before and
    // after a serialization/deserialization round trip.
    mlir::PassManager pm(context);
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    if (!mlir::succeeded(pm.run(*module))) {
      return absl::InvalidArgumentError("StableHLO => MHLO failed");
    }

    return std::make_unique<XlaProgram>(std::move(module));
  }

  static char ID;  // NOLINT
};

char XlaProgramSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_xla_program_serdes = ([]() {
  RegisterSerDes<XlaProgram>(std::make_unique<XlaProgramSerDes>());
}(), true);
// clang-format on

}  // namespace

char XlaDeserializeProgramOptions::ID = 0;

}  // namespace ifrt
}  // namespace xla
