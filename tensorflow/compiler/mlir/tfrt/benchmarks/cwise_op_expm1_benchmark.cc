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
func.func @expm1_1d(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Expm1"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
)";

// Use type aliases compatible with MLIR type names.
using f32 = float;

#define EXPR_BUILDER [](auto& in) { return in.expm1(); }

BM_TFMlir(Expm1, mlir_input, "expm1_1d", 1, f32, 1.0, 0.0, /* num_threads */ 0)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1024)
    ->Arg(10 * 1024);

BM_EigenScalar(Expm1, EXPR_BUILDER, 1, f32, 1.0, 0.0, /* num_threads */ 0)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1024)
    ->Arg(10 * 1024);

BM_EigenVectorized(Expm1, EXPR_BUILDER, 1, f32, 1.0, 0.0, /* num_threads */ 0)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1024)
    ->Arg(10 * 1024);

}  // namespace tensorflow
