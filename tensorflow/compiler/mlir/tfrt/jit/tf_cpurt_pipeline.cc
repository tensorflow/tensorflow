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
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
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

}  // namespace

// -------------------------------------------------------------------------- //
// Assemble a TF-CPURT pipeline to lower from Tensorflow dialects to Linalg on
// buffers via progressive lowering to MHLO and Linalg.
// -------------------------------------------------------------------------- //
void CreateTfCpuRtPipeline(mlir::OpPassManager& pm,
                           const TfCpuRtPipelineOptions& options) {
  // Break Tensorflow fused operations into primitive operations before
  // lowering to HLO.
  pm.addNestedPass<mlir::FuncOp>(CreateFissionPass());

  // Run shape inference to propagate potentially specialized input shapes.
  pm.addPass(std::make_unique<AddTensorflowProducerVersion>());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Transform TF operation to HLO.
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeTFPass());

  // Resolve all shape constraints (e.g. broadcast constraints that can be
  // proved statically and changed to const witness) early to allow more
  // efficient broadcast operations moving.
  pm.addNestedPass<mlir::FuncOp>(
      CreateSymbolicShapeOptimizationPass(/*constraints_only=*/true));

  // Move up broadcasting operations to allow for more fusion opportunities.
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // After all shape constraints removed and broadcasts moved to the top, try
  // to resolve broadcasts that can be converted to linalg generic operations.
  pm.addNestedPass<mlir::FuncOp>(CreateSymbolicShapeOptimizationPass());

  // Transform HLO operations to Linalg.
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreateShapeSimplification());
  pm.addNestedPass<mlir::FuncOp>(mlir::createShapeToShapeLowering());
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // Lower index cast on tensors to tensor.generate.
  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreateLowerIndexCastPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(CreateFusionPass());

  // Perform tiling-padding-vectorization if vectorization is enabled.
  if (options.vectorize) {
    pm.addNestedPass<mlir::FuncOp>(CreateDetensorizeLinalgPass());
    pm.addNestedPass<mlir::FuncOp>(CreateCodegenStrategyForReductionPass());
    pm.addNestedPass<mlir::FuncOp>(CreateCodegenStrategyForCWisePass());
    pm.addNestedPass<mlir::FuncOp>(CreatePeelTiledLoopsPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // TODO(b/205537489): Re-enable tiling after the padding function is fixed.
    // pm.addNestedPass<mlir::FuncOp>(CreatePadTiledOpsPass());
    pm.addNestedPass<mlir::FuncOp>(CreateSinkUnusedOutputs());
    pm.addNestedPass<mlir::FuncOp>(CreateVectorizeTiledOpsPass());
  }

  // Bufferize Linalg on tensors program.
  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());
  // Now bufferize all the compute operations (hlo + linalg) and func signature.
  pm.addPass(
      mlir::kernel_gen::transforms::CreateComputeOpAndFuncBufferizePass());
  pm.addNestedPass<mlir::FuncOp>(
      mlir::kernel_gen::transforms::CreateTiledLoopBufferizePass());
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
  pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());

  // Do trivial buffer forwarding across linalg.generic operations.
  pm.addNestedPass<mlir::FuncOp>(CreateLinalgTrivialBufferForwardingPass());

  // Remove trivial copy operations.
  pm.addNestedPass<mlir::FuncOp>(CreateLinalgTrivialCopyRemovalPass());

  // Specilize linalg.matmul to linalg.dot, linalg.matvec or linalg.vecmat, and
  // immediately canonicalize to clean up not taken branches.
  pm.addNestedPass<mlir::FuncOp>(CreateLinalgMatmulSpecializationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Tile and vectorize linalg operation using Linalg Codegen Strategy.
  pm.addNestedPass<mlir::FuncOp>(CreateCodegenStrategyForMatMulPass());

  if (options.vectorize) {
    pm.addNestedPass<mlir::FuncOp>(
        mlir::createConvertLinalgTiledLoopsToSCFPass());
  }
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  mlir::VectorTransferToSCFOptions vec_to_scf_options;
  vec_to_scf_options.unroll = true;
  pm.addNestedPass<mlir::FuncOp>(
      mlir::createConvertVectorToSCFPass(vec_to_scf_options));

  pm.addNestedPass<mlir::FuncOp>(CreateMathApproximationPass({"all"}));
}

void CreateDefaultTfCpuRtPipeline(mlir::OpPassManager& pm) {
  TfCpuRtPipelineOptions options;
  CreateTfCpuRtPipeline(pm, options);
}

static mlir::PassPipelineRegistration<TfCpuRtPipelineOptions> tf_cpurt_pipeline(
    "tf-cpurt-pipeline",
    "Convert Tensorflow dialect to TFRT's CPURT compatible dialects",
    CreateTfCpuRtPipeline);

}  // namespace tensorflow
