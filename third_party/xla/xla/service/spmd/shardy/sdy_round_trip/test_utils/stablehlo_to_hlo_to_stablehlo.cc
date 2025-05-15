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

#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/stablehlo_to_hlo_to_stablehlo.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::StringRef;

// Converts a StableHLO module to an HLO module.
absl::StatusOr<std::unique_ptr<HloModule>> toHlo(ModuleOp module) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hloModule,
                      xla::ConvertStablehloToHlo(module));
  hloModule->mutable_config().set_use_spmd_partitioning(true);
  return hloModule;
}

// Converts an HLO module to a StableHLO module.
absl::Status toStablehlo(std::unique_ptr<HloModule> hloModule,
                         ModuleOp& module) {
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> newModule,
      xla::ConvertHloToStablehlo(*module->getContext(), hloModule.get()));
  // Erase the old body region and replace it with the new one.
  module.getBodyRegion().takeBody(newModule.get().getBodyRegion());
  return absl::OkStatus();
}

class SdyRoundTripStablehloToHloToStablehloPass
    : public mlir::PassWrapper<SdyRoundTripStablehloToHloToStablehloPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripStablehloToHloToStablehloPass)

 private:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    // 1. StableHLO -> HLO
    absl::StatusOr<std::unique_ptr<HloModule>> hloModule = toHlo(module);
    if (!hloModule.ok()) {
      module.emitError(absl::StrCat("Failed to convert to HLO from StableHLO: ",
                                    hloModule.status().message()));
      return signalPassFailure();
    }

    // 2. HLO -> StableHLO
    if (absl::Status status = toStablehlo(std::move(*hloModule), module);
        !status.ok()) {
      module.emitError(absl::StrCat("Failed to convert to StableHLO from HLO: ",
                                    status.message()));
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-stablehlo-to-hlo-to-stablehlo";
  }

  StringRef getDescription() const override {
    return "Round trips from StableHLO -> HLO -> StableHLO.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    xla::RegisterMlirToHloDependentDialects(registry);
    // TODO(tomnatan): Cleanup once no longer needed.
    registry.insert<mlir::ub::UBDialect>();
  }
};

}  // namespace

void registerSdyRoundTripStablehloToHloToStablehloPass() {
  mlir::registerPass(createSdyRoundTripStablehloToHloToStablehloPass);
}

std::unique_ptr<mlir::Pass> createSdyRoundTripStablehloToHloToStablehloPass() {
  return std::make_unique<SdyRoundTripStablehloToHloToStablehloPass>();
}

}  // namespace sdy
}  // namespace xla
