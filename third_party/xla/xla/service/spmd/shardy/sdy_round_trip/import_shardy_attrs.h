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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_IMPORT_SHARDY_ATTRS_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_IMPORT_SHARDY_ATTRS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates the pass to convert frontend attributes to SDY attributes:
//
// - Converts meshes from `kMeshesRoundTripAttr` to sdy.mesh symbols
// - Converts shardings from `HloSharding::kShardingFrontendAttrName` to
//   `kShardingAttr`
// - Converts sharding rules from `kShardingRuleRoundTripAttr` to
//   `kShardingRuleAttr`
// - Replaces `kFuncResultShardingTargetName` custom-calls with the
//   corresponding func result sharding.
std::unique_ptr<mlir::Pass> createSdyRoundTripImportShardyAttrsPass();

// Registers the xla-sdy-round-trip-import-shardy-attrs pass.
void registerSdyRoundTripImportShardyAttrsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_IMPORT_SHARDY_ATTRS_H_
