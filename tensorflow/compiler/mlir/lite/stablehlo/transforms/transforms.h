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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_TRANSFORMS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace mlir {
namespace odml {

// Adds all the necessary passes to lower a TF module to StableHLO.
// `skip_resize` enables or disables skipping conversion of tf.ResizeBilinear
// and tf.ResizeNearestNeighbor ops.
// `smuggle_disallowed_ops` enables or disables converting disallowed ops
// like tf.ResizeBilinear or tf.ResizeNearestNeighbor to mhlo.custom_call ops.
void AddTFToStablehloPasses(OpPassManager& pm, bool skip_resize,
                            bool smuggle_disallowed_ops);

// Adds all the backend-agonstic stableHLO optimization passes
// this function is a common entry point for all graph optimizations that are
// not specific to any hardware
void AddStablehloOptimizationPasses(OpPassManager& pm);

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_TRANSFORMS_H_
