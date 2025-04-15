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

/// Lowers from HLO dialect to Arithmetic dialect.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToArithmeticPass();

/// Lowers from HLO dialect to Linalg dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeHloToLinalgPass(
    bool enablePrimitiveOps = false);

// Sinks constants implicitly captured in control flow regions. This is
// necessary to export to XLA.
std::unique_ptr<OperationPass<func::FuncOp>>
createSinkConstantsToControlFlowPass();

/// Lowers trigonometric operations from the standard dialect to approximations
/// that do not use intrinsics.
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeTrigonometricToApproximationPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeDotToDotGeneralPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeEinsumToDotGeneralPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeTorchIndexSelectToGatherPass();
std::unique_ptr<OperationPass<func::FuncOp>> createFlattenTuplePass();

// Creates a pass for expanding mhlo.tuple ops.
std::unique_ptr<OperationPass<ModuleOp>> createExpandHloTuplesPass(
    const std::string& entryFunctionName = "main");

// Creates a pass for collapsing the mhlo.map if the map only has elementwise
// op.
std::unique_ptr<OperationPass<func::FuncOp>> createCollapseElementwiseMapPass();

// Pass to replace unsigned types with signless integers.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToSignlessPass();

// Test passes.
std::unique_ptr<Pass> createTestInferShapedTypeMethodsPass();
std::unique_ptr<Pass> createTestMaterializeBroadcastsPass();
std::unique_ptr<Pass> createTestUnfuseBatchNormPass();

#define GEN_PASS_REGISTRATION
#include "mhlo/transforms/mhlo_passes.h.inc"

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_TRANSFORMS_PASSES_H
