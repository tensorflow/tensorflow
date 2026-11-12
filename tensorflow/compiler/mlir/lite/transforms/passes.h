/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_

// Declares all the TFLite converter passes: the ones in `tfl_passes.h` and the
// passes that operate on the TensorFlow dialect.

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/canonicalize_boundary_value_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/cast_bf16_ops_to_f32_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/cleanup_optimization_barrier_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/downcast_x64_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/large_constant_fold_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_batch_matmul_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/push_transpose_through_ewise_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tf_legalizations/analyze_variables_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tf_legalizations/legalize_tensorlist_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tf_legalizations/while_loop_outline_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tfl_passes.h"  // IWYU pragma: export
#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/split_merged_operands_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/unfold_large_splat_constants_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/unfreeze_global_constants.h"
#include "tensorflow/compiler/mlir/lite/transforms/utilities/elements_attr_roundtrip_pass.h"

namespace mlir {
namespace TFL {

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
inline std::unique_ptr<mlir::Pass> CreateWhileOutlinePass() {
  return Create<WhileOutlinePass>();
}

// Creates a pass which analyze the model whether it is safe to use
// native TFLite variables or not.
inline std::unique_ptr<mlir::Pass> CreateAnalyzeVariablesPass() {
  return Create<AnalyzeVariablesPass>();
}

// Creates a pass which is responsible for unfreezing mutable global tensors.
inline std::unique_ptr<mlir::Pass> CreateUnfreezeMutableGlobalTensorsPass() {
  return Create<UnfreezeMutableGlobalTensorsPass>();
}

// Legalize TensorList Ops iff all of them are supported.
inline std::unique_ptr<mlir::Pass> CreateLegalizeTensorListPass() {
  return Create<LegalizeTensorListPass>();
}

// Create a pass that canonicalize the boundary values.
inline std::unique_ptr<mlir::Pass> CreateCanonicalizeBoundaryValuePass() {
  return Create<CanonicalizeBoundaryValuePass>();
}

inline void registerTensorFlowLitePasses() {
  registerTensorFlowLiteTdPasses();
  // Register TFLite Converter Passes
  Register<UnfreezeMutableGlobalTensorsPass>();

  // TF Legalization Passes
  Register<AnalyzeVariablesPass>();
  Register<LegalizeTensorListPass>();
  Register<WhileOutlinePass>();

  // TFL Optimization Passes
  Register<OptimizePass, OptimizePassOptions>();
  Register<OptimizeBatchMatmulPass>();
  Register<UnfreezeMutableGlobalTensorsPass>();
  Register<OptimizeBroadcastLikePass, OptimizeBroadcastLikePassOptions>();
  Register<PushTransposeThroughEwisePass>();
  Register<CanonicalizeBoundaryValuePass>();

  // Other TFLite Passes
  Register<UnfoldLargeSplatConstantPass>();
  Register<SplitMergedOperandsPass>();
  Register<CleanupOptimizationBarrierPass>();
  Register<DowncastX64Pass>();
  Register<CastBf16OpsToF32Pass>();
  Register<LargeConstantFoldPass, LargeConstantFoldPassOptions>();

  // Utility Passes
  Register<DenseToDenseResourceElementsPass>();
  Register<DenseResourceToDenseElementsPass>();
}

}  // namespace TFL

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
