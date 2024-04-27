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
namespace lmhlo {
class FusionOp;
}  // namespace lmhlo

namespace mhlo {

#define GEN_PASS_DECL
#include "mhlo/transforms/mhlo_passes.h.inc"

/// Lowers HLO control flow ops to SCF.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeControlFlowPass();

/// Lowers sort to SCF & arith.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeSortPass();

/// Lowers from HLO dialect to Standard dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToStdPass();

// Lowers from sparse ops in CHLO dialect to Linalg dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeSparseOperationsPass(
    bool legalizeToCustomCalls = true);

// Canonicalize reduction ops to be suitable for codegen.
std::unique_ptr<OperationPass<func::FuncOp>>
createHloCanonicalizeReductionPass();

// Expand feature rich mhlo ops to simpler mhlo ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createMhloExpandOpsSimplifierPass();

// Rewrites scatter into transposes, reshapes and a simpler scatter.
std::unique_ptr<OperationPass<func::FuncOp>> createHloCanonicalizeScatterPass();

// Rewrites gather into transposes, reshapes and a simpler gather.
std::unique_ptr<OperationPass<func::FuncOp>> createHloCanonicalizeGatherPass();

// Rewrites dot operands that contain unit dimension.
std::unique_ptr<OperationPass<func::FuncOp>> createHloCanonicalizeDotPass();

/// Lowers from HLO dialect to LHLO dialect allocating/deallocating temporary
/// buffers if necessary.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToLhloPass();

/// Lowers from HLO dialect to Memref dialect allocating/deallocating temporary
/// buffers if necessary.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToMemrefPass();

/// Lowers from HLO dialect to Arithmetic dialect.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToArithmeticPass();

// Lowers shape operations from HLO dialect to Standard dialect.
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeHloShapeOpsToStandardPass();

/// Lowers from MHLO dialect to THLO dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeMHLOToTHLOPass(
    bool enableExperimentalOps = false);

/// Lowers from HLO dialect to Linalg dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeHloToLinalgPass(
    bool enablePrimitiveOps = false);

/// Lowers from HLO dialects dim operations.
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeShapeComputationsPass();

// Sinks constants implicitly captured in control flow regions. This is
// necessary to export to XLA.
std::unique_ptr<OperationPass<func::FuncOp>>
createSinkConstantsToControlFlowPass();

/// Lowers trigonometric operations from the standard dialect to approximations
/// that do not use intrinsics.
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeTrigonometricToApproximationPass();

// Move dynamic broadcasts up over element-wise operations and broadcast the
// operands rather than the result. This will eventually allow for larger
// fusions.
std::unique_ptr<OperationPass<func::FuncOp>> createBroadcastPropagationPass();

// Transformations that helps in restricting maximum rank among tensors in the
// pass.
std::unique_ptr<OperationPass<func::FuncOp>> createRestrictMaxRankPass();

// Prepare moving dynamic broadcasts up over element-wise operations and
// broadcast the operands rather than the result. This will eventually allow for
// larger fusions.
std::unique_ptr<OperationPass<func::FuncOp>> createMergeAssumingOpsPass();

// Iteratively reifies all shape computations in the function.
std::unique_ptr<OperationPass<func::FuncOp>> createShapeReificationPass();

/// Creates a pass to analyze shapes and to use that information for
/// shape-related optimizations.
std::unique_ptr<OperationPass<func::FuncOp>>
createSymbolicShapeOptimizationPass();

// Pass to simplify shape ops.
std::unique_ptr<OperationPass<func::FuncOp>> createShapeSimplification();

// Fuse shape constraints and merge all assuming regions.
std::unique_ptr<OperationPass<func::FuncOp>> createConstraintFusionPass();

// Group reduction and parallel dimensions of reduction operations and realize
// them through equivalent 1D or 2D reductions.
std::unique_ptr<OperationPass<func::FuncOp>> createGroupReductionDimensionsPass(
    bool preferColumnsReductions = true);

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeMhloPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLowerComplexPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeBroadcastToBroadcastInDimPass();
std::unique_ptr<::mlir::Pass> createLegalizeGeneralDotPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeCreateTokenToAfterAllPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeCrossReplicaSumToAllReducePass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeDotGeneralToDotPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeDotToDotGeneralPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeEinsumToDotGeneralPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeGatherToTorchIndexSelectPass();
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

/// Creates pass for rewriting sparse mhlo ops.
std::unique_ptr<OperationPass<func::FuncOp>> createSparseRewritingPass();

// Legalizes from the MHLO dialect to the StableHLO dialect.
std::unique_ptr<OperationPass<ModuleOp>> createHloLegalizeToStablehloPass();

// Legalizes from the StableHLO dialect to the MHLO dialect.
std::unique_ptr<OperationPass<ModuleOp>> createStablehloLegalizeToHloPass();

// Legalizes from the Shape dialect to the MHLO dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createShapeLegalizeToHloPass(
    bool legalizeConstraints = false);

// Legalizes from MHLO quantized ops with MHLO quant types to MHLO primitive ops
// like int ops.
std::unique_ptr<OperationPass<func::FuncOp>> createMhloQuantLegalizeToIntPass();

// Test passes.
std::unique_ptr<Pass> createTestInferShapedTypeMethodsPass();
std::unique_ptr<Pass> createTestMaterializeBroadcastsPass();
std::unique_ptr<Pass> createTestUnfuseBatchNormPass();

#define GEN_PASS_REGISTRATION
#include "mhlo/transforms/mhlo_passes.h.inc"

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_TRANSFORMS_PASSES_H
