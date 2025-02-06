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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_TF_STABLEHLO_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_TF_STABLEHLO_PASS_H_

#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace mlir {
namespace odml {

// Adds passes which transform TF Ops to StableHLO Ops.
void AddLegalizeTFToStablehloPasses(OpPassManager& pm,
                                    bool skip_quantization_ops,
                                    bool skip_resize,
                                    bool skip_partitioned_calls);

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_TF_STABLEHLO_PASS_H_
