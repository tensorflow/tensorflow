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

#ifndef XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_EXPORT_NAMED_COMPUTATIONS_H_
#define XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_EXPORT_NAMED_COMPUTATIONS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates a pass that converts a `NamedComputationOp` to a `CallOp` with a new
// private function called the `NamedComputationOp`'s `name`. The new `FuncOp`
// and `CallOp` have the same shardings as the original `NamedComputationOp`s
// operands/results.
//
// If there is a function with the same name as the `NamedComputationOp` in the
// module, the MLIR symbol table will change it to `{name}_#`.
std::unique_ptr<mlir::Pass> createExportNamedComputationsPass();

// Register the xla-sdy-export-named-computations pass.
void registerExportNamedComputationsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_EXPORT_NAMED_COMPUTATIONS_H_
