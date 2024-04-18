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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/passes.h"

#include "absl/log/check.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/assign_op_key.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/async_while.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/fuse_mlrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/parallelization.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/rewrite_ifrt_load_variable.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/tf_to_mlrt.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/while_to_map_fn.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"

namespace tensorflow {
namespace mlrt_compiler {

void RegisterMlrtPasses() {
  mlir::registerPass([]() { return CreateAssignOpKeyPass(); });
  mlir::registerPass([]() { return CreateAsyncWhilePass(); });
  mlir::registerPass([]() { return CreateParallelizationPass(); });
  mlir::registerPass([]() { return CreateWhileToMapFnPass(); });
  mlir::registerPass([]() { return CreateRewriteIfrtLoadVariablePass(); });
  mlir::registerPass(
      []() { return CreateTfToMlrtPreParallelizationConversionPass({}); });
  mlir::registerPass([]() { return CreateTfToMlrtConversionPass({}); });
  mlir::registerPass([]() { return CreateFuseMlrtOpPass(); });
}

void CreateTfToMlrtPipeline(mlir::OpPassManager &pm,
                            const TfrtPipelineOptions &options,
                            const tfrt_stub::FallbackState *fallback_state,
                            const tfrt_stub::CostRecorder *cost_recorder) {
  pm.addPass(
      mlrt_compiler::CreateTfToMlrtPreParallelizationConversionPass(options));

  pm.addPass(mlrt_compiler::CreateRewriteIfrtLoadVariablePass());

  if (options.enable_while_parallel_iterations) {
    pm.addPass(mlrt_compiler::CreateAsyncWhilePass());
  }

  pm.addPass(mlrt_compiler::CreateParallelizationPass(
      options.cost_threshold, options.merge_inter_dependent_streams,
      cost_recorder));

  DCHECK(fallback_state);
  pm.addPass(
      mlrt_compiler::CreateTfToMlrtConversionPass(options, fallback_state));

  // Perform optimizations in the lowered MLIR.
  pm.addNestedPass<mlir::func::FuncOp>(mlrt_compiler::CreateFuseMlrtOpPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
