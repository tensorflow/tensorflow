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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_CLONE_MANUAL_COMPUTATION_CALLS_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_CLONE_MANUAL_COMPUTATION_CALLS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

std::unique_ptr<mlir::Pass> createSdyRoundTripCloneManualComputationCallsPass();

// Registers the xla-sdy-round-trip-clone-manual-computation-calls pass.
void registerSdyRoundTripCloneManualComputationCallsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_CLONE_MANUAL_COMPUTATION_CALLS_H_
