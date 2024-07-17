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

#ifndef XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_CONVERT_SHARDING_CUSTOM_CALLS_H_
#define XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_CONVERT_SHARDING_CUSTOM_CALLS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates a pass that converts a `CustomCall` with target name Sharding into a
// `ShardingConstraintOp`.
std::unique_ptr<mlir::Pass> createConvertShardingCustomCallsPass();

// Register the xla-sdy-convert-sharding-custom-calls pass.
void registerConvertShardingCustomCallsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_CONVERT_SHARDING_CUSTOM_CALLS_H_
