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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h"

namespace tensorflow {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Reduce tensor<Nxf32> -> tensor<f32>.
////////////////////////////////////////////////////////////////////////////////

#define BM_DYNAMIC_1D(SIZE)                                                    \
  BM_SUITE_SUM_F32(Sum1DFullDynamic_##SIZE, 1, INTS(SIZE), BOOLS(kDynamicDim), \
                   1, INTS(0))
BM_DYNAMIC_1D(3);
BM_DYNAMIC_1D(8);
BM_DYNAMIC_1D(80);
BM_DYNAMIC_1D(800);
BM_DYNAMIC_1D(8000);
BM_DYNAMIC_1D(8131);
BM_DYNAMIC_1D(1000000);
BM_DYNAMIC_1D(1010131);

#define BM_STATIC_1D(SIZE)                                                   \
  BM_SUITE_SUM_F32(Sum1DFullStatic_##SIZE, 1, INTS(SIZE), BOOLS(kStaticDim), \
                   1, INTS(0))
BM_STATIC_1D(3);
BM_STATIC_1D(8);
BM_STATIC_1D(80);
BM_STATIC_1D(800);
BM_STATIC_1D(8000);
BM_STATIC_1D(8131);
BM_STATIC_1D(1000000);
BM_STATIC_1D(1010131);

////////////////////////////////////////////////////////////////////////////////
// Reduce tensor<NxMxf32> -> tensor<f32>.
////////////////////////////////////////////////////////////////////////////////

#define BM_DYNAMIC_ALL_2D(ROWS, COLS)                                        \
  BM_SUITE_SUM_F32(Sum2DFullDynamicAll_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kDynamicDim, kDynamicDim), 2, INTS(0, 1))
BM_DYNAMIC_ALL_2D(2, 80);
BM_DYNAMIC_ALL_2D(8, 6);
BM_DYNAMIC_ALL_2D(80, 1);
BM_DYNAMIC_ALL_2D(80, 60);
BM_DYNAMIC_ALL_2D(81, 61);
BM_DYNAMIC_ALL_2D(800, 600);
BM_DYNAMIC_ALL_2D(802, 602);

#define BM_STATIC_ROW_2D(ROWS, COLS)                                        \
  BM_SUITE_SUM_F32(Sum2DFullStaticRow_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kStaticDim, kDynamicDim), 2, INTS(0, 1))
BM_STATIC_ROW_2D(2, 80);
BM_STATIC_ROW_2D(8, 6);
BM_STATIC_ROW_2D(80, 1);
BM_STATIC_ROW_2D(80, 60);
BM_STATIC_ROW_2D(81, 61);
BM_STATIC_ROW_2D(800, 600);
BM_STATIC_ROW_2D(802, 602);

#define BM_STATIC_COL_2D(ROWS, COLS)                                        \
  BM_SUITE_SUM_F32(Sum2DFullStaticCol_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kDynamicDim, kStaticDim), 2, INTS(0, 1))
BM_STATIC_COL_2D(2, 80);
BM_STATIC_COL_2D(8, 6);
BM_STATIC_COL_2D(80, 1);
BM_STATIC_COL_2D(80, 60);
BM_STATIC_COL_2D(81, 61);
BM_STATIC_COL_2D(800, 600);
BM_STATIC_COL_2D(802, 602);

#define BM_STATIC_ALL_2D(ROWS, COLS)                                        \
  BM_SUITE_SUM_F32(Sum2DFullStaticAll_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kStaticDim, kStaticDim), 2, INTS(0, 1))
BM_STATIC_ALL_2D(2, 80);
BM_STATIC_ALL_2D(8, 6);
BM_STATIC_ALL_2D(80, 1);
BM_STATIC_ALL_2D(80, 60);
BM_STATIC_ALL_2D(81, 61);
BM_STATIC_ALL_2D(800, 600);
BM_STATIC_ALL_2D(802, 602);

////////////////////////////////////////////////////////////////////////////////
// Reduce tensor<MxNxKxf32> -> tensor<f32>.
////////////////////////////////////////////////////////////////////////////////

#define BM_DYNAMIC_ALL_3D(M, N, K)                                        \
  BM_SUITE_SUM_F32(Sum3DFullDynamicAll_##M##_##N##_##K, 3, INTS(M, N, K), \
                   BOOLS(kDynamicDim, kDynamicDim, kDynamicDim), 3,       \
                   INTS(0, 1, 2))

BM_DYNAMIC_ALL_3D(2, 80, 2);
BM_DYNAMIC_ALL_3D(8, 6, 8);
BM_DYNAMIC_ALL_3D(80, 1, 80);
BM_DYNAMIC_ALL_3D(80, 60, 80);
BM_DYNAMIC_ALL_3D(81, 61, 81);

////////////////////////////////////////////////////////////////////////////////
// Reduce tensor<MxNxKxLxf32> -> tensor<f32>.
////////////////////////////////////////////////////////////////////////////////

#define BM_DYNAMIC_ALL_4D(M, N, K, L)                                         \
  BM_SUITE_SUM_F32(Sum4DFullDynamicAll_##M##_##N##_##K##_##L, 4,              \
                   INTS(M, N, K, L),                                          \
                   BOOLS(kDynamicDim, kDynamicDim, kDynamicDim, kDynamicDim), \
                   4, INTS(0, 1, 2, 3))

BM_DYNAMIC_ALL_4D(2, 80, 2, 80);
BM_DYNAMIC_ALL_4D(8, 6, 8, 6);
BM_DYNAMIC_ALL_4D(80, 1, 80, 1);
BM_DYNAMIC_ALL_4D(80, 60, 80, 60);
BM_DYNAMIC_ALL_4D(81, 61, 81, 61);

#define BM_STATIC_ALL_4D(M, N, K, L)                                         \
  BM_SUITE_SUM_F32(Sum4DFullStaticAll_##M##_##N##_##K##_##L, 4,              \
                   INTS(M, N, K, L),                                         \
                   BOOLS(kStaticDim, kStaticDim, kStaticDim, kStaticDim), 4, \
                   INTS(0, 1, 2, 3))

BM_STATIC_ALL_4D(2, 80, 2, 80);
BM_STATIC_ALL_4D(8, 6, 8, 6);
BM_STATIC_ALL_4D(80, 1, 80, 1);
BM_STATIC_ALL_4D(80, 60, 80, 60);
BM_STATIC_ALL_4D(81, 61, 81, 61);

}  // namespace
}  // namespace tensorflow
