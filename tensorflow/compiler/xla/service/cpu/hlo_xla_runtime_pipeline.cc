/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/hlo_xla_runtime_pipeline.h"

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"  // from @llvm-project
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"  // from @llvm-project
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"  // from @llvm-project
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/passes.h"

namespace xla {
namespace cpu {
namespace {

using mlir::OpPassManager;
using mlir::func::FuncOp;

// Adds Linalg passes to perform fusion, tiling, peeling and vectorization.
void AddLinalgTransformations(OpPassManager& pm,
                              const HloXlaRuntimePipelineOptions& options) {
  pm.addNestedPass<FuncOp>(tensorflow::CreateFusionPass());

  if (!options.vectorize) return;

  pm.addNestedPass<FuncOp>(tensorflow::CreateDetensorizeLinalgPass());

  // Unfortunately, at the moment there is no way to provide default values for
  // ListOption. That's why we have to provide them here. When
  // https://github.com/llvm/llvm-project/issues/52667 feature request is
  // accepted and implemented, this line will have to be removed.
  mlir::SmallVector<int64_t, 2> reduction_2d_tile_sizes = {4, 4};
  if (options.reduction_2d_tile_sizes.hasValue()) {
    reduction_2d_tile_sizes.assign(options.reduction_2d_tile_sizes.begin(),
                                   options.reduction_2d_tile_sizes.end());
  }
  pm.addNestedPass<FuncOp>(tensorflow::CreateTileReductionPass(
      options.vector_size, options.reduction_1d_tile_size,
      reduction_2d_tile_sizes));

  if (options.vectorize && options.codegen_transpose)
    pm.addNestedPass<FuncOp>(tensorflow::CreateTileTransposePass());
  pm.addNestedPass<FuncOp>(
      tensorflow::CreateTileCWisePass(options.vector_size));
  if (options.peel) {
    pm.addNestedPass<FuncOp>(tensorflow::CreatePeelTiledLoopsPass());
  }
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if (options.fuse_fill) {
    pm.addNestedPass<FuncOp>(
        tensorflow::CreateFuseFillIntoTiledReductionPass());
  }
  pm.addNestedPass<FuncOp>(tensorflow::CreateTileFillPass(options.vector_size));
  pm.addNestedPass<FuncOp>(tensorflow::CreateVectorizeTiledOpsPass());
}

void AddBufferizationPasses(OpPassManager& pm, bool one_shot_bufferize) {
  // Rewrite init_tensor ops to alloc_tensor ops.
  pm.addNestedPass<FuncOp>(mlir::createLinalgInitTensorToAllocTensorPass());
  // Run One-Shot Bufferize.
  if (one_shot_bufferize) {
    pm.addPass(mlir::hlo::createOneShotBufferizePass());
    return;
  }
  // Now bufferize all the compute operations (hlo + linalg) and func signature.
  pm.addPass(mlir::createComputeOpAndFuncBufferizePass());
  pm.addNestedPass<FuncOp>(mlir::gml_st::CreateTiledLoopBufferizePass());
  // Always run CSE and canonicalizer (which does dead code removal) before
  // bufferizing anything.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createFinalBufferizePass(/*alignment=*/64));
}

}  // namespace

// -------------------------------------------------------------------------- //
// Assemble a HLO XLA Runtime pipeline to lower from HLO to Linalg on buffers.
// -------------------------------------------------------------------------- //
void CreateHloXlaRuntimePipeline(OpPassManager& pm,
                                 const HloXlaRuntimePipelineOptions& options) {
  if (options.legalize_i1_tensors) {
    // Convert 'i1' tensors into 'i8' tensors.
    pm.addPass(tensorflow::CreateJitRtLegalizeI1TypesPass());
  }

  // Remove redundant shape operations left after legalizing to HLO.
  pm.addPass(mlir::createCSEPass());

  // Resolve all shape constraints (e.g. broadcast constraints that can be
  // proved statically and changed to const witness) early to allow more
  // efficient broadcast operations moving.
  pm.addNestedPass<FuncOp>(tensorflow::CreateSymbolicShapeOptimizationPass(
      /*constraints_only=*/true));

  // Analyze shapes and try to simplify the IR as early as possible.
  pm.addNestedPass<FuncOp>(mlir::createSymbolicShapeOptimizationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Move up broadcasting operations to allow for more fusion opportunities.
  // Add the broadcast propagation pass first, because it can help to avoid
  // exponential complexity from the EarlyBroadcastInDimOp pattern which is used
  // in the merge assuming ops pass further down.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createMergeAssumingOpsPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // After all shape constraints removed and broadcasts moved to the top, try
  // to resolve broadcasts that can be converted to linalg generic operations.
  pm.addNestedPass<FuncOp>(tensorflow::CreateSymbolicShapeOptimizationPass());

  // Group reduction and parallel dimensions of reduction operations and realize
  // them through equivalent 1D or 2D reductions, if possible.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createGroupReductionDimensionsPass());

  // Also, try to simplify reshape operations.
  pm.addNestedPass<FuncOp>(mlir::createSymbolicShapeOptimizationPass());

  // Transform HLO operations to Linalg and Standard.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeControlFlowPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeSortPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());
  pm.addPass(mlir::mhlo::createLegalizeToArithmeticPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createLegalizeHloShapeOpsToStandardPass());

  // Now that all compute operations are converted to standard (as a side effect
  // of bufferizing to memref dialect) we can remove the remaining references
  // to unsigned types.
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  pm.addNestedPass<FuncOp>(mlir::createShapeSimplification());
  pm.addNestedPass<FuncOp>(mlir::createShapeToShapeLowering());
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // Lower index cast on tensors to tensor.generate.
  pm.addNestedPass<FuncOp>(mlir::createLowerIndexCastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Convert complex types.
  pm.addPass(mlir::createConvertComplexToStandardPass());

  // Add linalg passes to perform fusion, tiling, peeling and vectorization.
  AddLinalgTransformations(pm, options);

  // Inline everything, bufferization doesn't model ownership across calls.
  pm.addPass(mlir::createInlinerPass());

  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());

  AddBufferizationPasses(pm, options.one_shot_bufferize || options.vectorize);

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Deallocate all temporary buffers.
  pm.addNestedPass<FuncOp>(mlir::bufferization::createBufferDeallocationPass());

  // Do trivial buffer forwarding across linalg.generic operations.
  pm.addNestedPass<FuncOp>(
      tensorflow::CreateLinalgTrivialBufferForwardingPass());

  // Remove trivial copy operations.
  pm.addNestedPass<FuncOp>(tensorflow::CreateLinalgTrivialCopyRemovalPass());

  if (options.vectorize)
    pm.addNestedPass<FuncOp>(mlir::gml_st::createGmlStToScfPass());

  pm.addPass(mlir::createBufferizationToMemRefPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (options.vectorize && options.codegen_transpose)
    pm.addNestedPass<FuncOp>(tensorflow::CreateLowerVectorTransposePass());

  mlir::VectorTransferToSCFOptions vec_to_scf_options;
  vec_to_scf_options.unroll = true;
  pm.addNestedPass<FuncOp>(
      mlir::createConvertVectorToSCFPass(vec_to_scf_options));
  pm.addNestedPass<FuncOp>(tensorflow::createRewriteVectorMultiReductionPass());

  pm.addNestedPass<FuncOp>(tensorflow::CreateMathApproximationPass({"all"}));
}

void CreateDefaultHloXlaRuntimePipeline(OpPassManager& pm) {
  HloXlaRuntimePipelineOptions options;
  options.one_shot_bufferize = true;
  options.vectorize = true;
  CreateHloXlaRuntimePipeline(pm, options);
}

static mlir::PassPipelineRegistration<HloXlaRuntimePipelineOptions>
    hlo_jitrt_pipeline("hlo-xla-runtime-pipeline",
                       "Convert HLO dialect to XLA Runtime compatible dialects",
                       CreateHloXlaRuntimePipeline);

}  // namespace cpu
}  // namespace xla
