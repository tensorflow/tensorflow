// Copyright 2021 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_tfrt_gpu.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime

namespace tensorflow {

void populateLmhloToTfrtGpuPasses(mlir::OpPassManager &pm) {
  pm.addPass(tensorflow::createConvertLmhloToGpuBranchPass());
  pm.addPass(
      tfrt::gpu::CreateStreamifyOpsPass<
          lmhlo::AllGatherOp, lmhlo::AllReduceOp, lmhlo::ReduceScatterOp,
          lmhlo::AllToAllOp, lmhlo::CollectivePermuteOp, lmhlo::CustomCallOp,
          lmhlo::TriangularSolveOp, lmhlo::ReplicaIdOp, lmhlo::PartitionIdOp,
          lmhlo::InfeedOp, lmhlo::OutfeedOp, lmhlo::FftOp,
          lmhlo_gpu::ConvForwardOp, lmhlo_gpu::ConvBackwardInputOp,
          lmhlo_gpu::ConvBackwardFilterOp, lmhlo_gpu::ConvForwardFusedOp,
          lmhlo_gpu::ConvForwardFusedSideInputOp, lmhlo_gpu::GEMMOp,
          lmhlo_gpu::CholeskyOp, lmhlo_gpu::AllReduceStartOp,
          lmhlo_gpu::AllReduceDoneOp, mlir::func::CallOp, mlir::memref::LoadOp,
          tfrt::compiler::CallOp, tfrt::compiler::WhileOp>());
  pm.addPass(tensorflow::createConvertLmhloToGpuPass());
  pm.addPass(mlir::createGpuAsyncRegionPass());
  tfrt::gpu::PopulateGpuToTfrtGpuPasses(pm);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void registerLmhloToTfrtGpuPass() {
  PassPipelineRegistration<>(
      "lmhlo-to-tfrt-gpu", "Pass pipeline to convert from LMHLO to TFRT.",
      [](OpPassManager &pm) { populateLmhloToTfrtGpuPasses(pm); });
}

}  // namespace tensorflow
