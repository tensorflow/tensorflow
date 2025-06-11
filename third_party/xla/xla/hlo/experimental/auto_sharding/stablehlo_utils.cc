/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/experimental/auto_sharding/stablehlo_utils.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"

namespace xla::spmd {

absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertShardyToHlo(
    mlir::ModuleOp shardy_stablehlo_module) {
  mlir::OwningOpRef<mlir::ModuleOp> shardy_stablehlo_module_copy(
      shardy_stablehlo_module.clone());
  mlir::PassManager pm(shardy_stablehlo_module_copy->getContext());
  // TODO(hanruobing): This export pipeline replaces any sdy.sharding_constraint
  // with an mhlo.copy rather than a stablehlo.custom_call @Sharding, we may
  // need to add an option to to convert to custom call @Sharding.
  xla::sdy::addStablehloExportPipeline(pm);
  if (mlir::failed(pm.run(shardy_stablehlo_module_copy.get()))) {
    return absl::InternalError("Failed to export to StableHLO.");
  }
  // TODO(hanruobing): Explore reinserting the original mesh and calling
  // xla::sdy::createDedupMeshesPass
  return xla::ConvertStablehloToHlo(shardy_stablehlo_module_copy.get());
}

}  // namespace xla::spmd
