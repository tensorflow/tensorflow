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

// Use type aliases compatible with MLIR type names.
using f32 = float;

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

BM_TFMlir(MatMul_64_64_64, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({64, 64, 64, 16, 16, 8})
    ->Args({64, 64, 64, 32, 32, 8})
    ->Args({64, 64, 64, 8, 8, 8})
    ->Args({64, 64, 64, 16, 16, 16})
    ->Args({64, 64, 64, 32, 32, 16})
    ->Args({64, 64, 64, 8, 8, 16})
    ->Args({64, 64, 64, 8, 8, 32});
BM_Eigen(MatMul_64_64_64, f32)->ArgNames({"m", "k", "n"})->Args({64, 64, 64});

BM_TFMlir(MatMul_128_128_128, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({128, 128, 128, 16, 16, 8})
    ->Args({128, 128, 128, 32, 32, 8})
    ->Args({128, 128, 128, 8, 8, 8})
    ->Args({128, 128, 128, 16, 16, 16})
    ->Args({128, 128, 128, 32, 32, 16})
    ->Args({128, 128, 128, 8, 8, 16})
    ->Args({128, 128, 128, 8, 8, 32});
BM_Eigen(MatMul_128_128_128, f32)
    ->ArgNames({"m", "k", "n"})
    ->Args({128, 128, 128});

BM_TFMlir(MatMul_256_256_256, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({256, 256, 256, 16, 16, 8})
    ->Args({256, 256, 256, 32, 32, 8})
    ->Args({256, 256, 256, 8, 8, 8})
    ->Args({256, 256, 256, 16, 16, 16})
    ->Args({256, 256, 256, 32, 32, 16})
    ->Args({256, 256, 256, 8, 8, 16})
    ->Args({256, 256, 256, 8, 8, 32});
BM_Eigen(MatMul_256_256_256, f32)
    ->ArgNames({"m", "k", "n"})
    ->Args({256, 256, 256});

BM_TFMlir(MatMul_100_100_100, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({100, 100, 100, 16, 16, 8})
    ->Args({100, 100, 100, 32, 32, 8})
    ->Args({100, 100, 100, 8, 8, 8})
    ->Args({100, 100, 100, 16, 16, 16})
    ->Args({100, 100, 100, 32, 32, 16})
    ->Args({100, 100, 100, 8, 8, 16})
    ->Args({100, 100, 100, 8, 8, 32});
BM_Eigen(MatMul_100_100_100, f32)
    ->ArgNames({"m", "k", "n"})
    ->Args({100, 100, 100});

BM_TFMlir(MatMul_1024_1024_1024, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({1024, 1024, 1024, 128, 128, 16})
    ->Args({1024, 1024, 1024, 256, 256, 8})
    ->Args({1024, 1024, 1024, 128, 128, 8})
    ->Args({1024, 1024, 1024, 64, 64, 8})
    ->Args({1024, 1024, 1024, 32, 32, 8})
    ->Args({1024, 1024, 1024, 16, 16, 8})
    ->Args({1024, 1024, 1024, 8, 8, 8})
    ->Args({1024, 1024, 1024, 16, 16, 16})
    ->Args({1024, 1024, 1024, 32, 32, 16})
    ->Args({1024, 1024, 1024, 8, 8, 16})
    ->Args({1024, 1024, 1024, 8, 8, 32});
BM_Eigen(MatMul_1024_1024_1024, f32)
    ->ArgNames({"m", "k", "n"})
    ->Args({1024, 1024, 1024});

BM_TFMlir(MatMul_1_18_300, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({1, 18, 300, 32, 32, 8})
    ->Args({1, 18, 300, 16, 16, 8})
    ->Args({1, 18, 300, 8, 8, 8});
BM_Eigen(MatMul_1_18_300, f32)->ArgNames({"m", "k", "n"})->Args({1, 18, 300});

BM_TFMlir(MatMul_1_300_300, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({1, 300, 300, 32, 32, 8})
    ->Args({1, 300, 300, 16, 16, 8})
    ->Args({1, 300, 300, 8, 8, 8});
BM_Eigen(MatMul_1_300_300, f32)->ArgNames({"m", "k", "n"})->Args({1, 300, 300});

BM_TFMlir(MatMul_1_300_1, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({1, 300, 1, 32, 32, 8})
    ->Args({1, 300, 1, 16, 16, 8})
    ->Args({1, 300, 1, 8, 8, 8});
BM_Eigen(MatMul_1_300_1, f32)->ArgNames({"m", "k", "n"})->Args({1, 300, 1});

BM_TFMlir(MatMul_10_10_10, mlir_input, "matmul", f32)
    ->ArgNames({"m", "k", "n", "tiled_m", "tiled_n", "tiled_k"})
    ->Args({10, 10, 10, 8, 8, 8})
    ->Args({10, 10, 10, 4, 4, 4});
BM_Eigen(MatMul_10_10_10, f32)->ArgNames({"m", "k", "n"})->Args({10, 10, 10});

}  // namespace tensorflow
