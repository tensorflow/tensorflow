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

// LINT.IfChange
#include "tensorflow/dtensor/mlir/utils/dtensor_mlir_passes_internal.h"

#include <cstdlib>

#include "mlir/IR/BuiltinOps.h"
#include "tensorflow/dtensor/mlir/create_dtensor_mlir_passes.h"

namespace tensorflow {
namespace dtensor {

void AddDTensorAllReduceCombineOptimization(mlir::OpPassManager* pm){
  // Experimental feature. If zero, the optimization for combining all reduces
  // with same group assignment and reduction, will not be done.
  const char * env_str = (
      std::getenv("DTENSOR_ENABLE_COMBINE_ALL_REDUCES_OPTIMIZATION"));
  if (env_str && strcmp(env_str, "0") == 0) {
    return;
  }
  pm->addNestedPass<mlir::func::FuncOp>(
      CreateDTensorAllReduceCombineOptimization());
}

void AddDTensorEmbeddingPass(mlir::OpPassManager* pm){}

void AddDTensorEmbeddingPassV2(mlir::OpPassManager* pm){}

void AddDTensorEmbeddingCheckpointPass(mlir::OpPassManager* pm){}

}  // namespace dtensor
}  // namespace tensorflow

// LINT.ThenChange(dtensor_mlir_passes_internal.cc)
