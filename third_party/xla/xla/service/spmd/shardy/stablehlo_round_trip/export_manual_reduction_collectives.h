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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_MANUAL_REDUCTION_COLLECTIVES_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_MANUAL_REDUCTION_COLLECTIVES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Exports `sdy.all_reduce`, `sdy.reduce_scatter`, `sdy.sharded_to_unreduced`
// and `sdy.replicated_to_unreduced` that originate from user-defined shardings
// with unreduced axes. The exported ops are inside a full manual
// `sdy.manual_computation`.
std::unique_ptr<mlir::Pass>
createStablehloExportManualReductionCollectivesPass();

// Registers the xla-sdy-stablehlo-export-manual-reduction-collectives pass.
void registerStablehloExportManualReductionCollectivesPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_MANUAL_REDUCTION_COLLECTIVES_H_
