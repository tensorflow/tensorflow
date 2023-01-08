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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {

static const char* dynamic_dims = R"(
func.func @compute_dynamic(%arg0: tensor<?xf32>,
                      %arg1: tensor<1xf32>,
                      %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "tf.Mul"(%arg0, %arg1)
         : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
    %1 = "tf.AddV2"(%0, %arg2)
         : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    func.return %1 : tensor<?xf32>
}
)";

static const char* static_dims = R"(
func.func @compute_static(%arg0: tensor<1024xf32>,
                     %arg1: tensor<1xf32>,
                     %arg2: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = "tf.Mul"(%arg0, %arg1)
         : (tensor<1024xf32>, tensor<1xf32>) -> tensor<1024xf32>
    %1 = "tf.AddV2"(%0, %arg2)
         : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    func.return %1 : tensor<1024xf32>
}
)";

static llvm::SmallVector<InputTensorSpec> Inputs() {
  return {
      InputTensorSpec(DT_FLOAT, {1024}),  // %arg0
      InputTensorSpec(DT_FLOAT, {1}),     // %arg1
      InputTensorSpec(DT_FLOAT, {1024}),  // %arg2
  };
}

BM_Jitrt(ComputeDynamicDims, dynamic_dims, "compute_dynamic", Inputs())->Arg(0);
BM_Jitrt(ComputeStaticDims, static_dims, "compute_static", Inputs())->Arg(0);

}  // namespace tensorflow
