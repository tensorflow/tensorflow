/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_UNFLATTEN_CALL_GRAPH_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_UNFLATTEN_CALL_GRAPH_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates a pass that unflattens the given call graph by deduplicating
// functions with the same origin. In case `dedupFunctionsFully` is true, it
// deduplicates all functions with the same origin, and otherwise, it
// deduplicates for those with the same origin and the input/output shardings
// are the same.
std::unique_ptr<mlir::Pass> createUnflattenCallGraphPass(
    bool dedupFunctionsFully);

// Register the xla-sdy-unflatten-call-graph pass.
void registerUnflattenCallGraphPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_UNFLATTEN_CALL_GRAPH_H_
