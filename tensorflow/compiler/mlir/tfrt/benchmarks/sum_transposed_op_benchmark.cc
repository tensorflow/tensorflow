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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h"

namespace tensorflow {
namespace {

#define BM_DYNAMIC_ALL(SUFFIX, M, N, K, N_DIMS_TO_REDUCE, DIMS_TO_REDUCE)  \
  BM_SUITE_SUM_F32(Sum3DTransposeDynamicAll_##SUFFIX##_##M##_##N##_##K, 3, \
                   INTS(M, N, K),                                          \
                   BOOLS(kDynamicDim, kDynamicDim, kDynamicDim),           \
                   N_DIMS_TO_REDUCE, INTS(DIMS_TO_REDUCE))

// Reduction dimensions = {1}
BM_DYNAMIC_ALL(Dim1, 2, 80, 2, 1, INTS(1));
BM_DYNAMIC_ALL(Dim1, 8, 6, 8, 1, INTS(1));
BM_DYNAMIC_ALL(Dim1, 80, 1, 80, 1, INTS(1));
BM_DYNAMIC_ALL(Dim1, 80, 60, 80, 1, INTS(1));
BM_DYNAMIC_ALL(Dim1, 81, 61, 81, 1, INTS(1));

// Reduction dimensions = {0, 2}
BM_DYNAMIC_ALL(Dims02, 2, 80, 2, 2, INTS(0, 2));
BM_DYNAMIC_ALL(Dims02, 8, 6, 8, 2, INTS(0, 2));
BM_DYNAMIC_ALL(Dims02, 80, 1, 80, 2, INTS(0, 2));
BM_DYNAMIC_ALL(Dims02, 80, 60, 80, 2, INTS(0, 2));
BM_DYNAMIC_ALL(Dims02, 81, 61, 81, 2, INTS(0, 2));

}  // namespace
}  // namespace tensorflow
