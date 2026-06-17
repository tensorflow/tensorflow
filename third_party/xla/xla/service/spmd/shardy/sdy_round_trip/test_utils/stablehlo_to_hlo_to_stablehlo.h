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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_TEST_UTILS_STABLEHLO_TO_HLO_TO_STABLEHLO_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_TEST_UTILS_STABLEHLO_TO_HLO_TO_STABLEHLO_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates the pass that round trips from StableHLO -> HLO -> StableHLO.
std::unique_ptr<mlir::Pass> createSdyRoundTripStablehloToHloToStablehloPass();

// Register the xla-sdy-round-trip-stablehlo-to-hlo-to-stablehlo pass.
void registerSdyRoundTripStablehloToHloToStablehloPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_TEST_UTILS_STABLEHLO_TO_HLO_TO_STABLEHLO_H_
