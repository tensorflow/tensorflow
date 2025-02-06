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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_SHARD_MAP_EXPORT_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_SHARD_MAP_EXPORT_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates the pass that converts `ManualComputationOp`s to a separate function
// with a CallOp and a pair of `CustomCallOp`s that change the shape of the
// arguments/results. The CallOp saves the in/out shardings and manual axes as
// frontend attrs.
std::unique_ptr<mlir::Pass> createSdyRoundTripShardMapExportPass();

// Registers the xla-sdy-round-trip-shard-map-export pass.
void registerSdyRoundTripShardMapExportPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_SHARD_MAP_EXPORT_H_
