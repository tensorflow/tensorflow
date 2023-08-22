/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h"

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project

namespace impl {
namespace {
#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h.inc"
}  // namespace
}  // namespace impl

namespace xla::gpu {

using namespace mlir;  // NOLINT

void registerGpu2Pases() { ::impl::registerPasses(); }

void populateGpu2RuntimePasses(mlir::OpPassManager& pm,
                               ThunkSequence* thunk_sequence,
                               RuntimeBackend backend,
                               const Gpu2PipelineOpts& opts) {
  // Use xla_gpu graph regions only if we are running with StreamExecutor
  // backend and graphs are enabled.
  bool use_graph_api =
      backend == RuntimeBackend::kStreamExecutor && opts.graph_level;

  if (use_graph_api) pm.addPass(createCreateGraphRegionsPass());
  pm.addPass(createConvertToGpu2RuntimePass(thunk_sequence, backend));
  if (use_graph_api) pm.addPass(createFinalizeGraphDispatchesPass());

  // Clean up IR before passing it to IREE compiler.
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

}  // namespace xla::gpu
