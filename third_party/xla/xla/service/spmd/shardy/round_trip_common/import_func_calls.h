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

#ifndef XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_IMPORT_FUNC_CALLS_H_
#define XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_IMPORT_FUNC_CALLS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates a pass that converts a `CallOp` to a `NamedComputationOp` with the
// function body inlined and name of the callee.
//
// NOTE: In case there are multiple call ops for the same callee, we will clone
// the function body for each call op and emit a warning.
std::unique_ptr<mlir::Pass> createImportFuncCallsPass(
    bool enableNativeNonFlatSupport);

// Register the xla-sdy-import-calls pass.
void registerImportFuncCallsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_IMPORT_FUNC_CALLS_H_
