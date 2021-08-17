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

static const char* mlir_input = R"(
func @compute(%arg0: tensor<?x18xf32>,
              %arg1: tensor<18xf32>,
              %arg2: tensor<18xf32>,
              %arg3: tensor<18x300xf32>,
              %arg4: tensor<300xf32>,
              %arg5: tensor<300xf32>,
              %arg6: tensor<300xf32>,
              %arg7: tensor<300x300xf32>,
              %arg8: tensor<300xf32>,
              %arg9: tensor<300xf32>,
              %arg10: tensor<300xf32>,
              %arg11: tensor<300x1xf32>,
              %arg12: tensor<1xf32>) -> tensor<?x1xf32> {
    %0 = "tf.Log1p"(%arg0)
         : (tensor<?x18xf32>) -> tensor<?x18xf32>
    %1 = "tf.Sub"(%0, %arg1)
         : (tensor<?x18xf32>, tensor<18xf32>) -> tensor<?x18xf32>
    %2 = "tf.Mul"(%1, %arg2)
         : (tensor<?x18xf32>, tensor<18xf32>) -> tensor<?x18xf32>
    %3 = "tf.MatMul"(%2, %arg3) {transpose_a = false, transpose_b = false}
         : (tensor<?x18xf32>, tensor<18x300xf32>) -> tensor<?x300xf32>
    %4 = "tf.BiasAdd"(%3, %arg4) {data_format = "NHWC"}
         : (tensor<?x300xf32>, tensor<300xf32>) -> tensor<?x300xf32>
    %5 = "tf.Relu"(%4)
         : (tensor<?x300xf32>) -> tensor<?x300xf32>
    %6 = "tf.Mul"(%5, %arg5)
         : (tensor<?x300xf32>, tensor<300xf32>) -> tensor<?x300xf32>
    %7 = "tf.AddV2"(%6, %arg6)
         : (tensor<?x300xf32>, tensor<300xf32>) -> tensor<?x300xf32>
    %8 = "tf.MatMul"(%7, %arg7) {transpose_a = false, transpose_b = false}
         : (tensor<?x300xf32>, tensor<300x300xf32>) -> tensor<?x300xf32>
    %9 = "tf.BiasAdd"(%8, %arg8) {data_format = "NHWC"}
         : (tensor<?x300xf32>, tensor<300xf32>) -> tensor<?x300xf32>
    %10 = "tf.Relu"(%9)
         : (tensor<?x300xf32>) -> tensor<?x300xf32>
    %11 = "tf.Mul"(%10, %arg9)
         : (tensor<?x300xf32>, tensor<300xf32>) -> tensor<?x300xf32>
    %12 = "tf.AddV2"(%11, %arg10)
         : (tensor<?x300xf32>, tensor<300xf32>) -> tensor<?x300xf32>
    %13 = "tf.MatMul"(%12, %arg11) {transpose_a = false, transpose_b = false}
         : (tensor<?x300xf32>, tensor<300x1xf32>) -> tensor<?x1xf32>
    %14 = "tf.BiasAdd"(%13, %arg12) {data_format = "NHWC"}
         : (tensor<?x1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
    return %14 : tensor<?x1xf32>
  }
)";

static llvm::SmallVector<InputTensorSpec> Inputs() {
  return {
      InputTensorSpec(DT_FLOAT, {1, 18}),     // %arg0
      InputTensorSpec(DT_FLOAT, {18}),        // %arg1
      InputTensorSpec(DT_FLOAT, {18}),        // %arg2
      InputTensorSpec(DT_FLOAT, {18, 300}),   // %arg3
      InputTensorSpec(DT_FLOAT, {300}),       // %arg4
      InputTensorSpec(DT_FLOAT, {300}),       // %arg5
      InputTensorSpec(DT_FLOAT, {300}),       // %arg6
      InputTensorSpec(DT_FLOAT, {300, 300}),  // %arg7
      InputTensorSpec(DT_FLOAT, {300}),       // %arg8
      InputTensorSpec(DT_FLOAT, {300}),       // %arg9
      InputTensorSpec(DT_FLOAT, {300}),       // %arg10
      InputTensorSpec(DT_FLOAT, {300, 1}),    // %arg11
      InputTensorSpec(DT_FLOAT, {1}),         // %arg12
  };
}

BM_Mlir(Compute, mlir_input, "compute", Inputs())->Arg(0);

}  // namespace tensorflow
