/* Copyright 2019 The OpenXLA Authors.

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

#ifndef MLIR_HLO_MHLO_TRANSFORMS_PASSES_H
#define MLIR_HLO_MHLO_TRANSFORMS_PASSES_H

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;
namespace func {
class FuncOp;
}  // namespace func

namespace mhlo {

#define GEN_PASS_DECL
#include "mhlo/transforms/mhlo_passes.h.inc"

/// Returns the default options for the ChloLegalizeToHighLevelMhloPass. These
/// options specify the ops that are supported by all XLA backends.
ChloLegalizeToHighLevelMhloPassOptions getDefaultChloToHighLevelMhloOptions();

/// Returns options for the ChloLegalizeToHighLevelMhloPass for the GPU backend.
ChloLegalizeToHighLevelMhloPassOptions getGpuChloToHighLevelMhloOptions();

// TODO(b/397167511): Remove legacy wrapper once callers are migrated.
inline std::unique_ptr<mlir::Pass>
createLegalizeTrigonometricToApproximationPass() {
  return createLegalizeTanhToApproximationPass();
}

// TODO(b/397167511): Remove legacy wrapper once callers are migrated.
inline std::unique_ptr<mlir::Pass> createExpandHloTuplesPass(
    const std::string& entryFunctionName) {
  ExpandHloTuplesPassOptions options;
  options.entry_function_name_ = entryFunctionName;
  return createExpandHloTuplesPass(options);
}

#define GEN_PASS_REGISTRATION
#include "mhlo/transforms/mhlo_passes.h.inc"

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_TRANSFORMS_PASSES_H
