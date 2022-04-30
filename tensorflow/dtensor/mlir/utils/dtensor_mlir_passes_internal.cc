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

#include "tensorflow/dtensor/mlir/utils/dtensor_mlir_passes_internal.h"

#include "mlir/IR/BuiltinOps.h"
#include "tensorflow/dtensor/mlir/create_dtensor_mlir_passes.h"

namespace tensorflow {
namespace dtensor {

void AddDTensorAllReduceCombineOptimization(mlir::OpPassManager* pm){}

void AddDTensorEmbeddingPass(mlir::OpPassManager* pm){}

void AddDTensorEmbeddingPassV2(mlir::OpPassManager* pm){}

void AddDTensorEmbeddingLoadPass(mlir::OpPassManager* pm){}

}  // namespace dtensor
}  // namespace tensorflow
