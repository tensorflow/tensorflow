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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_TRANSFORMS_LEGALIZE_TO_AFFINE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_TRANSFORMS_LEGALIZE_TO_AFFINE_H_

#include <memory>

#include "mlir/Pass/Pass.h"  // TF:local_config_mlir

namespace xla {
namespace mlir_gpu {

// Lowers from LHLO dialect to affine dialect.
std::unique_ptr<::mlir::FunctionPassBase> createLegalizeAffine();

// Adds patterns to convert LHLO binary ops to affine loops.
void AppendBinaryOpsPatterns(::mlir::MLIRContext* context,
                             ::mlir::OwningRewritePatternList* patterns);

}  // namespace mlir_gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_TRANSFORMS_LEGALIZE_TO_AFFINE_H_
