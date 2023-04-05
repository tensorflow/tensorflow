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

#include <array>
#include <string>

namespace tensorflow {

// Use type aliases compatible with MLIR type names.
using f32 = float;

static const char* matmul_ir_skeleton = R"(
func.func @matmul(%arg0: {0}, %arg1: {1}) -> {2} {
    %0 = "tf.MatMul"(%arg0, %arg1) {{
           transpose_a = false,
           transpose_b = false
         } : ({0}, {1}) -> {2}
    func.return %0 : {2}
  }
)";

std::string GetMatmulIR(std::array<int64_t, 2> lhs_shape,
                        std::array<int64_t, 2> rhs_shape,
                        std::array<int64_t, 2> out_shape,
                        llvm::StringRef element_type) {
  return llvm::formatv(
      matmul_ir_skeleton,
      PrintTensorType(lhs_shape, element_type),  // LHS type {0}
      PrintTensorType(rhs_shape, element_type),  // RHS type {1}
      PrintTensorType(out_shape, element_type)   // Out type {2}
  );
}

static void Shapes(benchmark::internal::Benchmark* b) {
  for (int64_t i = 16; i <= 2048; i *= 2) {
    b->Args({i, i, i, i >= 256});
  }

  b->Args({10, 10, 10, false});
  b->Args({100, 100, 100, false});

  b->Args({1, 300, 18, false});
  b->Args({1, 18, 300, false});
  b->Args({18, 1, 300, false});
  b->Args({18, 300, 1, false});
  b->Args({300, 1, 18, false});
  b->Args({300, 18, 1, false});

  for (int64_t i : {300, 256}) {
    b->Args({1, 1, i, false});
    b->Args({1, i, 1, false});
    b->Args({i, 1, 1, false});

    b->Args({1, i, i, false});
    b->Args({i, 1, i, false});
    b->Args({i, i, 1, false});
  }
}

BM_TFMlir(MatmulMlirStatic, false, "matmul", f32)->Apply(Shapes);
BM_TFMlir(MatmulMlirDynamic, true, "matmul", f32)->Apply(Shapes);
BM_Eigen(MatmulEigen, f32)->Apply(Shapes);

}  // namespace tensorflow
