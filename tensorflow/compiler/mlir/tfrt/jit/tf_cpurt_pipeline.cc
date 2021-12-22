/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/jit/tf_cpurt_pipeline.h"

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

// -------------------------------------------------------------------------- //
// Custom passes that are missing upstream.
// -------------------------------------------------------------------------- //

namespace tensorflow {
namespace {

using mlir::FuncOp;
using mlir::OpPassManager;

// Adds a Tensorflow producer version to the module to enable shape inference.
struct AddTensorflowProducerVersion
    : public mlir::PassWrapper<AddTensorflowProducerVersion,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Tensorflow producer version does not really impact anything during the
    // shape inference. Set it to `0` (any random number will do the work) to
    // bypass attribute checks.
    mlir::Builder builder(module);
    auto version =
        builder.getNamedAttr("producer", builder.getI32IntegerAttr(0));
    module->setAttr("tf.versions", builder.getDictionaryAttr({version}));
  }
};

// Adds Linalg passes to perform fusion, tiling, peeling and vectorization.
void AddLinalgTransformations(OpPassManager& pm,
                              const TfCpuRtPipelineOptions& options) {
  pm.addNestedPass<FuncOp>(CreateFusionPass());

  if (!options.vectorize) return;

  pm.addNestedPass<FuncOp>(CreateDetensorizeLinalgPass());

  // Unfortunately, at the moment there is no way to provide default values for
  // ListOption. That's why we have to provide them here. When
  // https://github.com/llvm/llvm-project/issues/52667 feature request is
  // accepted and implemented, this line will have to be removed.
  mlir::SmallVector<int64_t, 2> reduction_2d_tile_sizes = {4, 4};
  if (options.reduction_2d_tile_sizes.hasValue()) {
    reduction_2d_tile_sizes.assign(options.reduction_2d_tile_sizes.begin(),
                                   options.reduction_2d_tile_sizes.end());
  }
  pm.addNestedPass<FuncOp>(CreateCodegenStrategyForReductionPass(
      options.reduction_1d_tile_size, reduction_2d_tile_sizes));

  if (options.fuse_fill) {
    pm.addNestedPass<FuncOp>(CreateFuseFillIntoTiledReductionPass());
  }
  pm.addNestedPass<FuncOp>(
      CreateCodegenStrategyForCWisePass(options.cwise_tile_size));
  if (options.peel) {
    pm.addNestedPass<FuncOp>(CreatePeelTiledLoopsPass());
  }
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(CreateSinkUnusedOutputs());
  pm.addNestedPass<FuncOp>(CreateVectorizeTiledOpsPass());
}

}  // namespace

// -------------------------------------------------------------------------- //
// Assemble a TF-CPURT pipeline to lower from Tensorflow dialects to Linalg on
// buffers via progressive lowering to MHLO and Linalg.
// -------------------------------------------------------------------------- //
void CreateTfCpuRtPipeline(OpPassManager& pm,
                           const TfCpuRtPipelineOptions& options) {
  // Break Tensorflow fused operations into primitive operations before
  // lowering to HLO.
  pm.addNestedPass<FuncOp>(CreateFissionPass());

  // Run shape inference to propagate potentially specialized input shapes.
  pm.addPass(std::make_unique<AddTensorflowProducerVersion>());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Transform TF operation to HLO.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeTFPass());

  if (options.legalize_i1_tensors) {
    // Convert 'i1' tensors into 'i8' tensors.
    pm.addPass(CreateCpuRtLegalizeI1TypesPass());
  }

  // Resolve all shape constraints (e.g. broadcast constraints that can be
  // proved statically and changed to const witness) early to allow more
  // efficient broadcast operations moving.
  pm.addNestedPass<FuncOp>(
      CreateSymbolicShapeOptimizationPass(/*constraints_only=*/true));

  // Move up broadcasting operations to allow for more fusion opportunities.
  // Add the broadcast propagation pass first, because it can help to avoid
  // exponential complexity from the EarlyBroadcastInDimOp pattern which is used
  // in the merge assuming ops pass further down.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createMergeAssumingOpsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // After all shape constraints removed and broadcasts moved to the top, try
  // to resolve broadcasts that can be converted to linalg generic operations.
  pm.addNestedPass<FuncOp>(CreateSymbolicShapeOptimizationPass());

  // Transform HLO operations to Linalg and Standard.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createLegalizeHloShapeOpsToStandardPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateShapeSimplification());
  pm.addNestedPass<FuncOp>(mlir::createShapeToShapeLowering());
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // Lower index cast on tensors to tensor.generate.
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateLowerIndexCastPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Add linalg passes to perform fusion, tiling, peeling and vectorization.
  AddLinalgTransformations(pm, options);

  // Bufferize Linalg on tensors program.
  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());
  // Now bufferize all the compute operations (hlo + linalg) and func signature.
  pm.addPass(
      mlir::kernel_gen::transforms::CreateComputeOpAndFuncBufferizePass());
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateTiledLoopBufferizePass());
  // Now that all compute operations are converted to standard (as a side effect
  // of bufferizing to memref dialect) we can remove the remaining references
  // to unsigned types.
  pm.addPass(mlir::kernel_gen::transforms::CreateConvertToSignlessPass());
  // Turn tensor constants into global memrefs.
  // TODO(kramerb): Expose the patterns and add them to the bufferize passes.
  pm.addPass(mlir::createTensorConstantBufferizePass(/*alignment=*/64));
  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::kernel_gen::transforms::CreateFinalBufferizePass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Deallocate all temporary buffers.
  pm.addNestedPass<FuncOp>(mlir::bufferization::createBufferDeallocationPass());

  // Do trivial buffer forwarding across linalg.generic operations.
  pm.addNestedPass<FuncOp>(CreateLinalgTrivialBufferForwardingPass());

  // Remove trivial copy operations.
  pm.addNestedPass<FuncOp>(CreateLinalgTrivialCopyRemovalPass());

  if (options.vectorize) {
    pm.addNestedPass<FuncOp>(mlir::createConvertLinalgTiledLoopsToSCFPass());
  }
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  mlir::VectorTransferToSCFOptions vec_to_scf_options;
  vec_to_scf_options.unroll = true;
  pm.addNestedPass<FuncOp>(
      mlir::createConvertVectorToSCFPass(vec_to_scf_options));

  pm.addNestedPass<FuncOp>(CreateMathApproximationPass({"all"}));
}

void CreateDefaultTfCpuRtPipeline(OpPassManager& pm) {
  TfCpuRtPipelineOptions options;
  options.vectorize = tensorflow::GetCpuRtFlags().vectorize;
  CreateTfCpuRtPipeline(pm, options);
}

static mlir::PassPipelineRegistration<TfCpuRtPipelineOptions> tf_cpurt_pipeline(
    "tf-cpurt-pipeline",
    "Convert Tensorflow dialect to TFRT's CPURT compatible dialects",
    CreateTfCpuRtPipeline);

}  // namespace tensorflow
