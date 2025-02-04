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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_CALLBACK_CUSTOM_CALLS_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_CALLBACK_CUSTOM_CALLS_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace xla {
namespace sdy {

// Creates a pass that converts the `CustomCallOp`s for host callbacks in XLA
// into the pattern that the XLA compiler recognizes.
//
// The rest of the XLA pipeline expects host callback custom calls to either be
// a tuple with a get_tuple_element or no results (which we changed due to
// shardy shardings expecting at least one result, and needing to attach a
// maximal sharding to the callbacks).
std::unique_ptr<mlir::Pass>
createStablehloRoundTripExportCallbackCustomCallsPass();

// Registers the xla-sdy-stablehlo-round-trip-export-callback-custom-calls pass.
void registerStablehloRoundTripExportCallbackCustomCallsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_CALLBACK_CUSTOM_CALLS_H_
