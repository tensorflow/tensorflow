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

#define BM_DYNAMIC_ALL(SUFFIX, M, N, K, N_DIMS_TO_REDUCE, DIMS_TO_REDUCE) \
  BM_SUITE_SUM_F32(Sum3DDynamicAll_##M##_##N##_##K##_##SUFFIX, 3,         \
                   INTS(M, N, K),                                         \
                   BOOLS(kDynamicDim, kDynamicDim, kDynamicDim),          \
                   N_DIMS_TO_REDUCE, INTS(DIMS_TO_REDUCE))

// Reduction dimensions = {0}
BM_DYNAMIC_ALL(ReductionDim0, 2, 80, 2, 1, INTS(0));
BM_DYNAMIC_ALL(ReductionDim0, 8, 6, 8, 1, INTS(0));
BM_DYNAMIC_ALL(ReductionDim0, 80, 1, 80, 1, INTS(0));
BM_DYNAMIC_ALL(ReductionDim0, 80, 60, 80, 1, INTS(0));
BM_DYNAMIC_ALL(ReductionDim0, 81, 61, 81, 1, INTS(0));

// Reduction dimensions = {1}
BM_DYNAMIC_ALL(ReductionDim1, 2, 80, 2, 1, INTS(1));
BM_DYNAMIC_ALL(ReductionDim1, 8, 6, 8, 1, INTS(1));
BM_DYNAMIC_ALL(ReductionDim1, 80, 1, 80, 1, INTS(1));
BM_DYNAMIC_ALL(ReductionDim1, 80, 60, 80, 1, INTS(1));
BM_DYNAMIC_ALL(ReductionDim1, 81, 61, 81, 1, INTS(1));

// Reduction dimensions = {2}
BM_DYNAMIC_ALL(ReductionDim2, 2, 80, 2, 1, INTS(2));
BM_DYNAMIC_ALL(ReductionDim2, 8, 6, 8, 1, INTS(2));
BM_DYNAMIC_ALL(ReductionDim2, 80, 1, 80, 1, INTS(2));
BM_DYNAMIC_ALL(ReductionDim2, 80, 60, 80, 1, INTS(2));
BM_DYNAMIC_ALL(ReductionDim2, 81, 61, 81, 1, INTS(2));

// Reduction dimensions = {0, 1}
BM_DYNAMIC_ALL(ReductionDims01, 2, 80, 2, 2, INTS(0, 1));
BM_DYNAMIC_ALL(ReductionDims01, 8, 6, 8, 2, INTS(0, 1));
BM_DYNAMIC_ALL(ReductionDims01, 80, 1, 80, 2, INTS(0, 1));
BM_DYNAMIC_ALL(ReductionDims01, 80, 60, 80, 2, INTS(0, 1));
BM_DYNAMIC_ALL(ReductionDims01, 81, 61, 81, 2, INTS(0, 1));

// Reduction dimensions = {0, 2}
BM_DYNAMIC_ALL(ReductionDims02, 2, 80, 2, 2, INTS(0, 2));
BM_DYNAMIC_ALL(ReductionDims02, 8, 6, 8, 2, INTS(0, 2));
BM_DYNAMIC_ALL(ReductionDims02, 80, 1, 80, 2, INTS(0, 2));
BM_DYNAMIC_ALL(ReductionDims02, 80, 60, 80, 2, INTS(0, 2));
BM_DYNAMIC_ALL(ReductionDims02, 81, 61, 81, 2, INTS(0, 2));

// Reduction dimensions = {1, 2}
BM_DYNAMIC_ALL(ReductionDims12, 2, 80, 2, 2, INTS(1, 2));
BM_DYNAMIC_ALL(ReductionDims12, 8, 6, 8, 2, INTS(1, 2));
BM_DYNAMIC_ALL(ReductionDims12, 80, 1, 80, 2, INTS(1, 2));
BM_DYNAMIC_ALL(ReductionDims12, 80, 60, 80, 2, INTS(1, 2));
BM_DYNAMIC_ALL(ReductionDims12, 81, 61, 81, 2, INTS(1, 2));

// Reduction dimensions = {0, 1, 2}
BM_DYNAMIC_ALL(ReductionDim012, 2, 80, 2, 3, INTS(0, 1, 2));
BM_DYNAMIC_ALL(ReductionDim012, 8, 6, 8, 3, INTS(0, 1, 2));
BM_DYNAMIC_ALL(ReductionDim012, 80, 1, 80, 3, INTS(0, 1, 2));
BM_DYNAMIC_ALL(ReductionDim012, 80, 60, 80, 3, INTS(0, 1, 2));
BM_DYNAMIC_ALL(ReductionDim012, 81, 61, 81, 3, INTS(0, 1, 2));

}  // namespace
}  // namespace tensorflow
