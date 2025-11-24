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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STABLEHLO_PASS_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STABLEHLO_PASS_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace xla {
namespace spmd {
void RegisterDialectDependencies(mlir::DialectRegistry& registry);
void AddAutoShardingToPipeline(mlir::OpPassManager& pm);
void RegisterAutoSharding();
// Register Alpa auto partitioner in case no other auto partitioner is already
// registered.
// TODO(b/431368844): Remove when there is a way for users to register Alpa.
void RegisterAutoShardingIfRegistryEmpty();
}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STABLEHLO_PASS_H_
