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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/cwise_op_unary_benchmark.h"

namespace tensorflow {

static const char* mlir_input = R"(
#map0 = affine_map<(d0) -> (d0)>
func.func @log2_1d(%input: memref<?xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %input, %c0 : memref<?xf32>
  %output = memref.alloc(%0) : memref<?xf32>

  linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]
  }
    ins(%input: memref<?xf32>)
    outs(%output : memref<?xf32>)
  {
    ^bb0(%in: f32, %out: f32):
     %2 = math.log2 %in : f32
      linalg.yield %2 : f32
  }
  func.return %output : memref<?xf32>
}
)";

// Use type aliases compatible with MLIR type names.
using f32 = float;

#define EXPR_BUILDER [](auto& in) { return in.log2(); }

BM_Mlir(Log2, mlir_input, "log2_1d", 1, f32, 1.0, 1.0, /* num_threads */ 0)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1024)
    ->Arg(10 * 1024);

BM_EigenScalar(Log2, EXPR_BUILDER, 1, f32, 1.0, 1.0, /* num_threads */ 0)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1024)
    ->Arg(10 * 1024);

BM_EigenVectorized(Log2, EXPR_BUILDER, 1, f32, 1.0, 1.0, /* num_threads */ 0)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1024)
    ->Arg(10 * 1024);

}  // namespace tensorflow
