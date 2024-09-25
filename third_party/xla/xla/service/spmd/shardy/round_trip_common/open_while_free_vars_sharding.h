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

#ifndef XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_OPEN_WHILE_FREE_VARS_SHARDING_H_
#define XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_OPEN_WHILE_FREE_VARS_SHARDING_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates a pass that adds a fully open sharding constraint to free variables
// of while op that already have a user-defined sharding.
//
// This allows for their uses in the while op to be further sharded, which is
// important when converting to HLO as they will be lifted as passthrough while
// operands/results.
std::unique_ptr<mlir::Pass> createOpenWhileFreeVarsShardingPass();

// Registers the xla-sdy-open-while-free-vars-sharding pass.
void registerOpenWhileFreeVarsShardingPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_OPEN_WHILE_FREE_VARS_SHARDING_H_
