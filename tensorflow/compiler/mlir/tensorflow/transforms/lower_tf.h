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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TF_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TF_H_

#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project

namespace mlir {
namespace TF {

// Populates TensorFlow lowering patterns to lower some of the TensorFlow
// operations that can be represented using other TensorFlow operations.
void PopulateLoweringTFPatterns(MLIRContext *context,
                                OwningRewritePatternList *patterns);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_LOWER_TF_H_
