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

#include "xla/service/spmd/shardy/sdy_round_trip/test_utils/mhlo_to_hlo_to_mhlo.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::StringRef;

// Converts an MHLO module to an HLO module.
absl::StatusOr<std::unique_ptr<HloModule>> toHlo(ModuleOp module) {
  absl::StatusOr<std::unique_ptr<HloModule>> hloModule;
  xla::HloProto hloProto;
  TF_RETURN_IF_ERROR(ConvertMlirHloToHlo(module, &hloProto,
                                         /*use_tuple_args=*/false,
                                         /*return_tuple=*/false));
  xla::HloModuleConfig moduleConfig;
  xla::ProgramShape expectedProgramShape(
      hloProto.hlo_module().host_program_shape());
  moduleConfig.SetDefaultComputationLayout(expectedProgramShape);
  moduleConfig.set_use_spmd_partitioning(true);
  return xla::HloModule::CreateFromProto(hloProto.hlo_module(), moduleConfig);
}

// Converts an HLO module to an MHLO module.
absl::Status toMhlo(std::unique_ptr<HloModule> hloModule, ModuleOp module) {
  // Delete the functions, which can be more than one due to preserving
  // the shmap_body functions.
  mlir::SymbolTableCollection symbolTableCollection;
  mlir::SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(module);
  for (mlir::Operation& op :
       llvm::make_early_inc_range(module.getBodyRegion().getOps())) {
    symbolTable.erase(&op);
  }
  TF_RETURN_IF_ERROR(
      xla::ConvertHloToMlirHlo(module, hloModule.get(),
                               /*import_all_computations=*/false,
                               /*flatten_computation_args_result=*/true));
  return absl::OkStatus();
}

class SdyRoundTripMhloToHloToMhloPass
    : public mlir::PassWrapper<SdyRoundTripMhloToHloToMhloPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripMhloToHloToMhloPass)

 private:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    // 1. MHLO -> HLO
    absl::StatusOr<std::unique_ptr<HloModule>> hloModule = toHlo(module);
    if (!hloModule.ok()) {
      module.emitError(absl::StrCat("Failed to convert to HLO from MHLO: ",
                                    hloModule.status().message()));
      return signalPassFailure();
    }

    // 2. HLO -> MHLO
    if (absl::Status status = toMhlo(std::move(*hloModule), module);
        !status.ok()) {
      module.emitError(absl::StrCat("Failed to convert to MHLO from HLO: ",
                                    status.message()));
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-mhlo-to-hlo-to-mhlo";
  }

  StringRef getDescription() const override {
    return "Round trips from MHLO -> HLO -> MHLO.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect, mlir::stablehlo::StablehloDialect,
                    mlir::mhlo::MhloDialect, mlir::quant::QuantDialect,
                    mlir::sparse_tensor::SparseTensorDialect>();
  }
};

}  // namespace

void registerSdyRoundTripMhloToHloToMhloPass() {
  mlir::registerPass(createSdyRoundTripMhloToHloToMhloPass);
}

std::unique_ptr<mlir::Pass> createSdyRoundTripMhloToHloToMhloPass() {
  return std::make_unique<SdyRoundTripMhloToHloToMhloPass>();
}

}  // namespace sdy
}  // namespace xla
