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

#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

namespace tensorflow {

void populateLmhloToTfrtGpuPasses(mlir::OpPassManager &pm) {
  pm.addPass(tensorflow::createConvertLmhloToGpuPass());
  pm.addPass(mlir::createGpuAsyncRegionPass());
  tfrt::gpu::populateGpuToTfrtGpuPasses(pm);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void registerLmhloToTfrtGpuPass() {
  PassPipelineRegistration<>(
      "lmhlo-to-tfrt-gpu", "Pass pipeline to convert from LMHLO to TFRT.",
      [](OpPassManager &pm) { populateLmhloToTfrtGpuPasses(pm); });
}

}  // namespace tensorflow
