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

#ifndef TENSORFLOW_DTENSOR_MLIR_UTILS_DTENSOR_MLIR_PASSES_INTERNAL_H_
#define TENSORFLOW_DTENSOR_MLIR_UTILS_DTENSOR_MLIR_PASSES_INTERNAL_H_

#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace tensorflow {
namespace dtensor {

void AddDTensorAllReduceCombineOptimization(mlir::OpPassManager* pm);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_UTILS_DTENSOR_MLIR_PASSES_INTERNAL_H_
