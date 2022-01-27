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

#include <string>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h"

namespace tensorflow {
namespace {

/// All dimensions are statically shaped.

#define BM_4D_REDUCE_STATIC_ALL(M, N, K, L)                                  \
  BM_SUITE_SUM_F32(Sum4DReduceStaticAll_##M##_##N##_##K##_##L, 4,            \
                   INTS(M, N, K, L),                                         \
                   BOOLS(kStaticDim, kStaticDim, kStaticDim, kStaticDim), 4, \
                   INTS(0, 1, 2, 3))

BM_4D_REDUCE_STATIC_ALL(2, 80, 2, 80);
BM_4D_REDUCE_STATIC_ALL(8, 6, 8, 6);
BM_4D_REDUCE_STATIC_ALL(80, 1, 80, 1);
BM_4D_REDUCE_STATIC_ALL(80, 60, 80, 60);
BM_4D_REDUCE_STATIC_ALL(81, 61, 81, 61);

#define BM_4D_COL_LIKE_REDUCE_STATIC_ALL(M, N, K, L)                          \
  BM_SUITE_SUM_F32(                                                           \
      Sum4DColLikeReduceStaticAll_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kStaticDim, kStaticDim, kStaticDim, kStaticDim), 2, INTS(0, 1))

BM_4D_COL_LIKE_REDUCE_STATIC_ALL(2, 80, 2, 80);
BM_4D_COL_LIKE_REDUCE_STATIC_ALL(8, 6, 8, 6);
BM_4D_COL_LIKE_REDUCE_STATIC_ALL(80, 1, 80, 1);
BM_4D_COL_LIKE_REDUCE_STATIC_ALL(80, 60, 80, 60);
BM_4D_COL_LIKE_REDUCE_STATIC_ALL(81, 61, 81, 61);

#define BM_4D_ROW_LIKE_REDUCE_STATIC_ALL(M, N, K, L)                          \
  BM_SUITE_SUM_F32(                                                           \
      Sum4DRowLikeReduceStaticAll_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kStaticDim, kStaticDim, kStaticDim, kStaticDim), 2, INTS(2, 3))

BM_4D_ROW_LIKE_REDUCE_STATIC_ALL(2, 80, 2, 80);
BM_4D_ROW_LIKE_REDUCE_STATIC_ALL(8, 6, 8, 6);
BM_4D_ROW_LIKE_REDUCE_STATIC_ALL(80, 1, 80, 1);
BM_4D_ROW_LIKE_REDUCE_STATIC_ALL(80, 60, 80, 60);
BM_4D_ROW_LIKE_REDUCE_STATIC_ALL(81, 61, 81, 61);

/// All parallel dimensions are statically shaped, all reduction dimensions are
/// dynamically shaped.

#define BM_4D_COL_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(M, N, K, L)       \
  BM_SUITE_SUM_F32(                                                    \
      Sum4DColLikeReduceDynamicReductionDims_##M##_##N##_##K##_##L, 4, \
      INTS(M, N, K, L),                                                \
      BOOLS(kDynamicDim, kDynamicDim, kStaticDim, kStaticDim), 2, INTS(0, 1))

BM_4D_COL_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(2, 80, 2, 80);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(8, 6, 8, 6);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(80, 1, 80, 1);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(80, 60, 80, 60);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(81, 61, 81, 61);

#define BM_4D_ROW_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(M, N, K, L)       \
  BM_SUITE_SUM_F32(                                                    \
      Sum4DRowLikeReduceDynamicReductionDims_##M##_##N##_##K##_##L, 4, \
      INTS(M, N, K, L),                                                \
      BOOLS(kStaticDim, kStaticDim, kDynamicDim, kDynamicDim), 2, INTS(2, 3))

BM_4D_ROW_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(2, 80, 2, 80);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(8, 6, 8, 6);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(80, 1, 80, 1);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(80, 60, 80, 60);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_REDUCTION_DIMS(81, 61, 81, 61);

/// Exactly one of the parallel dimensions is dynamically shaped, all other
/// dimensions are statically shaped.

#define BM_4D_COL_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(M, N, K, L)      \
  BM_SUITE_SUM_F32(                                                     \
      Sum4DColLikeReduceOneDynamicParallelDim_##M##_##N##_##K##_##L, 4, \
      INTS(M, N, K, L),                                                 \
      BOOLS(kStaticDim, kStaticDim, kStaticDim, kDynamicDim), 2, INTS(0, 1))

BM_4D_COL_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(2, 80, 2, 80);
BM_4D_COL_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(8, 6, 8, 6);
BM_4D_COL_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(80, 1, 80, 1);
BM_4D_COL_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(80, 60, 80, 60);
BM_4D_COL_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(81, 61, 81, 61);

#define BM_4D_ROW_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(M, N, K, L)      \
  BM_SUITE_SUM_F32(                                                     \
      Sum4DRowLikeReduceOneDynamicParallelDim_##M##_##N##_##K##_##L, 4, \
      INTS(M, N, K, L),                                                 \
      BOOLS(kStaticDim, kDynamicDim, kStaticDim, kStaticDim), 2, INTS(2, 3))

BM_4D_ROW_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(2, 80, 2, 80);
BM_4D_ROW_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(8, 6, 8, 6);
BM_4D_ROW_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(80, 1, 80, 1);
BM_4D_ROW_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(80, 60, 80, 60);
BM_4D_ROW_LIKE_REDUCE_ONE_DYNAMIC_PARALLEL_DIM(81, 61, 81, 61);

/// Exactly one of the parallel dimensions is dynamically shaped, the remaining
/// parallel dimension is statically shaped, all reduction dimensions are
/// dynamically shaped.

#define BM_4D_REDUCE_DYNAMIC_ALL(M, N, K, L)                                  \
  BM_SUITE_SUM_F32(Sum4DReduceDynamicAll_##M##_##N##_##K##_##L, 4,            \
                   INTS(M, N, K, L),                                          \
                   BOOLS(kDynamicDim, kDynamicDim, kDynamicDim, kDynamicDim), \
                   4, INTS(0, 1, 2, 3))

BM_4D_REDUCE_DYNAMIC_ALL(2, 80, 2, 80);
BM_4D_REDUCE_DYNAMIC_ALL(8, 6, 8, 6);
BM_4D_REDUCE_DYNAMIC_ALL(80, 1, 80, 1);
BM_4D_REDUCE_DYNAMIC_ALL(80, 60, 80, 60);
BM_4D_REDUCE_DYNAMIC_ALL(81, 61, 81, 61);

#define BM_4D_COL_LIKE_REDUCE_DYNAMIC_ALL(M, N, K, L)                          \
  BM_SUITE_SUM_F32(                                                            \
      Sum4DColLikeReduceDynamicAll_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kDynamicDim, kDynamicDim, kStaticDim, kDynamicDim), 2, INTS(0, 1))

BM_4D_COL_LIKE_REDUCE_DYNAMIC_ALL(2, 80, 2, 80);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_ALL(8, 6, 8, 6);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_ALL(80, 1, 80, 1);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_ALL(80, 60, 80, 60);
BM_4D_COL_LIKE_REDUCE_DYNAMIC_ALL(81, 61, 81, 61);

#define BM_4D_ROW_LIKE_REDUCE_DYNAMIC_ALL(M, N, K, L)                          \
  BM_SUITE_SUM_F32(                                                            \
      Sum4DRowLikeReduceDynamicAll_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kStaticDim, kDynamicDim, kDynamicDim, kDynamicDim), 2, INTS(2, 3))

BM_4D_ROW_LIKE_REDUCE_DYNAMIC_ALL(2, 80, 2, 80);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_ALL(8, 6, 8, 6);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_ALL(80, 1, 80, 1);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_ALL(80, 60, 80, 60);
BM_4D_ROW_LIKE_REDUCE_DYNAMIC_ALL(81, 61, 81, 61);

}  // namespace
}  // namespace tensorflow
