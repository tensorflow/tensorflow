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

#ifndef TENSORFLOW_COMPILER_MLIR_STABLEHLO_TRANSFORMS_STABLEHLO_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_STABLEHLO_TRANSFORMS_STABLEHLO_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace odml {

// Unfuses MHLO batch norm inference op into arithmetic ops.
std::unique_ptr<Pass> createUnfuseBatchNormPass();

// Constant folds broadcast_in_dim op conditionally.
std::unique_ptr<Pass> createFoldBroadcastPass();

// Fuses MHLO binary element-wise ops and convolution op.
std::unique_ptr<Pass> createFuseConvolutionPass();

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_STABLEHLO_TRANSFORMS_STABLEHLO_PASSES_H_
