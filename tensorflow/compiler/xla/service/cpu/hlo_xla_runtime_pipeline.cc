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
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/runtime/compiler.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/gml_st/transforms/bufferizable_op_interface_impl.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/bufferizable_op_interface_impl.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/passes.h"

namespace xla {
namespace cpu {
namespace {

using mlir::func::FuncOp;

mlir::bufferization::OneShotBufferizationOptions GetBufferizationOptions() {
  using mlir::bufferization::BufferizationOptions;
  using mlir::bufferization::OneShotBufferizationOptions;

  OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.allowReturnAllocs = true;
  options.functionBoundaryTypeConversion =
      BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  options.unknownTypeConverterFn = [](mlir::Value value, unsigned memorySpace,
                                      const BufferizationOptions& options) {
    return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(
        value.getType().cast<mlir::TensorType>(), memorySpace);
  };
  return options;
}

void AddSparsificationPasses(xla::runtime::PassManager& passes) {
  passes->addNestedPass<FuncOp>(mlir::createLinalgGeneralizationPass());
  passes->addNestedPass<FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());
  passes->addPass(mlir::bufferization::createTensorCopyInsertionPass(
      GetBufferizationOptions()));
  passes->addPass(mlir::createSparseTensorRewritePass());
  passes->addPass(mlir::createSparsificationPass());
  passes->addPass(mlir::createSparseTensorConversionPass());
  passes->addPass(
      mlir::createDenseBufferizationPass(GetBufferizationOptions()));
  passes->addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
}

}  // namespace

// -------------------------------------------------------------------------- //
// Assemble a HLO XLA Runtime pipeline to lower from HLO to Linalg on buffers.
// -------------------------------------------------------------------------- //
void CreateDefaultHloXlaRuntimePipeline(xla::runtime::PassManager& passes) {
  passes->addPass(mlir::createInlinerPass());
  passes->addPass(mlir::mhlo::createExpandHloTuplesPass("main"));
  // TODO(b/233771980): Remove once custom_call doesn't use tuples.
  passes->addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createFlattenTuplePass());
  // Remove redundant shape operations left after legalizing to HLO.
  passes->addPass(mlir::createCSEPass());
  passes->addPass(mlir::createCanonicalizerPass());

  // Move up broadcasting operations to allow for more fusion opportunities.
  // Add the broadcast propagation pass first, because it can help to avoid
  // exponential complexity from the EarlyBroadcastInDimOp pattern which is used
  // in the merge assuming ops pass further down.
  passes->addNestedPass<FuncOp>(mlir::mhlo::createMergeAssumingOpsPass());
  passes->addNestedPass<FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  passes->addPass(mlir::createCSEPass());
  passes->addPass(mlir::createCanonicalizerPass());

  // Group reduction and parallel dimensions of reduction operations and realize
  // them through equivalent 1D or 2D reductions, if possible.
  passes->addNestedPass<FuncOp>(
      mlir::mhlo::createGroupReductionDimensionsPass());

  // Also, try to simplify reshape operations.
  passes->addNestedPass<FuncOp>(mlir::createSymbolicShapeOptimizationPass());

  passes->addNestedPass<FuncOp>(mlir::mhlo::createSparseRewritingPass());
  passes->addNestedPass<FuncOp>(mlir::mhlo::createLegalizeGeneralDotPass());

  // Transform HLO operations to Linalg and Standard.
  passes->addNestedPass<FuncOp>(mlir::mhlo::createLegalizeControlFlowPass());
  passes->addNestedPass<FuncOp>(mlir::mhlo::createLegalizeSortPass());
  passes->addNestedPass<FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());
  passes->addPass(mlir::mhlo::createLegalizeToArithmeticPass());
  passes->addNestedPass<FuncOp>(
      mlir::mhlo::createLegalizeHloShapeOpsToStandardPass());

  // Now that all compute operations are converted to standard (as a side effect
  // of bufferizing to memref dialect) we can remove the remaining references
  // to unsigned types.
  passes->addPass(mlir::mhlo::createConvertToSignlessPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  passes->addNestedPass<FuncOp>(mlir::createShapeSimplification());
  passes->addNestedPass<FuncOp>(mlir::createShapeToShapeLowering());
  passes->addPass(mlir::createConvertShapeToStandardPass());
  passes->addNestedPass<FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  passes->addPass(mlir::createCSEPass());
  passes->addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // Lower index cast on tensors to tensor.generate.
  passes->addNestedPass<FuncOp>(mlir::createLowerIndexCastPass());
  passes->addPass(mlir::createCSEPass());
  passes->addPass(mlir::createCanonicalizerPass());

  // Convert complex types.
  passes->addPass(mlir::createConvertComplexToStandardPass());

  // Add linalg passes to perform fusion, tiling, peeling and vectorization.
  //  AddLinalgTransformations(pm, options);

  // Inline everything, bufferization doesn't model ownership across calls.
  passes->addPass(mlir::createInlinerPass());

  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  passes->addPass(mlir::createCanonicalizerPass());

  // Convert sparse tensors.
  AddSparsificationPasses(passes);

  passes->addPass(mlir::createCSEPass());
  passes->addPass(mlir::createCanonicalizerPass());

  // Deallocate all temporary buffers.
  passes->addNestedPass<FuncOp>(
      mlir::bufferization::createBufferDeallocationPass());

  passes->addNestedPass<FuncOp>(mlir::gml_st::createGmlStToScfPass());

  passes->addPass(mlir::createBufferizationToMemRefPass());
  passes->addPass(mlir::createCSEPass());
  passes->addPass(mlir::createCanonicalizerPass());

  mlir::VectorTransferToSCFOptions vec_to_scf_options;
  vec_to_scf_options.unroll = true;
  passes->addNestedPass<FuncOp>(
      mlir::createConvertVectorToSCFPass(vec_to_scf_options));
}

void RegisterHloXlaRuntimePipelineDialects(
    xla::runtime::DialectRegistry& dialects) {
  mlir::arith::registerBufferizableOpInterfaceExternalModels(*dialects);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      *dialects);
  mlir::gml_st::registerBufferizableOpInterfaceExternalModels(*dialects);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(*dialects);
  mlir::mhlo::registerBufferizableOpInterfaceExternalModels(*dialects);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(*dialects);
  mlir::shape::registerBufferizableOpInterfaceExternalModels(*dialects);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(*dialects);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(*dialects);
}

static void CreateDefaultHloXlaPipeline(mlir::OpPassManager& pm) {
  xla::runtime::PassManager passes(&pm);
  CreateDefaultHloXlaRuntimePipeline(passes);
}

static mlir::PassPipelineRegistration<> hlo_xla_runtime_pipeline(
    "hlo-xla-runtime-pipeline",
    "Convert HLO dialect to XLA Runtime compatible dialects",
    CreateDefaultHloXlaPipeline);

}  // namespace cpu
}  // namespace xla
