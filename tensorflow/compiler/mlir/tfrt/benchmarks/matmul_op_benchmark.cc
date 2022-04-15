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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/matmul_op_benchmark.h"

namespace tensorflow {

static const char* mlir_input = R"(
func.func @matmul(%arg0: tensor<?x?xf32>,
             %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = "tf.MatMul"(%arg0, %arg1) {
           transpose_a = false,
           transpose_b = false
         } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %0 : tensor<?x?xf32>
  }
)";

// Use type aliases compatible with MLIR type names.
using f32 = float;

BM_TFMlir(MatMul, mlir_input, "matmul", f32)
    ->Args({10, 10, 10})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({1, 18, 300})
    ->Args({1, 300, 300})
    ->Args({1, 300, 1});

BM_Eigen(MatMul, f32)
    ->Args({10, 10, 10})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({1, 18, 300})
    ->Args({1, 300, 300})
    ->Args({1, 300, 1});

}  // namespace tensorflow
