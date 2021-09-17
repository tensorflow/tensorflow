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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LHLO_GPU_TO_TFRT_CUDA_CCL_PATTERN_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LHLO_GPU_TO_TFRT_CUDA_CCL_PATTERN_H_

#include "mlir/IR/PatternMatch.h"

namespace tensorflow {

void populateCclConversionPattern(mlir::RewritePatternSet& patterns);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LHLO_GPU_TO_TFRT_CUDA_CCL_PATTERN_H_
