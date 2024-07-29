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

#ifndef XLA_SERVICE_SPMD_SHARDY_MHLO_ROUND_TRIP_EXPORT_OPS_H_
#define XLA_SERVICE_SPMD_SHARDY_MHLO_ROUND_TRIP_EXPORT_OPS_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace xla {
namespace sdy {

// Creates a pass that converts Shardy ops to MHLO ops (except
// sdy::ManualComputationOp).
std::unique_ptr<mlir::Pass> createExportOpsPass();

// Register the xla-sdy-export-ops pass.
void registerExportOpsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_MHLO_ROUND_TRIP_EXPORT_OPS_H_
