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
// Reduce tensor<NxMxf32> -> tensor<Mxf32>.
////////////////////////////////////////////////////////////////////////////////

#define BM_DYNAMIC_ALL_2D(ROWS, COLS)                                       \
  BM_SUITE_SUM_F32(Sum2DColDynamicAll_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kDynamicDim, kDynamicDim), 1, INTS(0))
BM_DYNAMIC_ALL_2D(2, 80);
BM_DYNAMIC_ALL_2D(8, 6);
BM_DYNAMIC_ALL_2D(80, 1);
BM_DYNAMIC_ALL_2D(80, 60);
BM_DYNAMIC_ALL_2D(81, 61);
BM_DYNAMIC_ALL_2D(800, 600);
BM_DYNAMIC_ALL_2D(802, 602);

#define BM_STATIC_ROW_2D(ROWS, COLS)                                       \
  BM_SUITE_SUM_F32(Sum2DColStaticRow_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kStaticDim, kDynamicDim), 1, INTS(0))
BM_STATIC_ROW_2D(2, 80);
BM_STATIC_ROW_2D(8, 6);
BM_STATIC_ROW_2D(80, 1);
BM_STATIC_ROW_2D(80, 60);
BM_STATIC_ROW_2D(81, 61);
BM_STATIC_ROW_2D(800, 600);
BM_STATIC_ROW_2D(802, 602);

#define BM_STATIC_COL_2D(ROWS, COLS)                                       \
  BM_SUITE_SUM_F32(Sum2DColStaticCol_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kDynamicDim, kStaticDim), 1, INTS(0))
BM_STATIC_COL_2D(2, 80);
BM_STATIC_COL_2D(8, 6);
BM_STATIC_COL_2D(80, 1);
BM_STATIC_COL_2D(80, 60);
BM_STATIC_COL_2D(81, 61);
BM_STATIC_COL_2D(800, 600);
BM_STATIC_COL_2D(802, 602);

#define BM_STATIC_ALL_2D(ROWS, COLS)                                       \
  BM_SUITE_SUM_F32(Sum2DColStaticAll_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                   BOOLS(kStaticDim, kStaticDim), 1, INTS(0))
BM_STATIC_ALL_2D(2, 80);
BM_STATIC_ALL_2D(8, 6);
BM_STATIC_ALL_2D(80, 1);
BM_STATIC_ALL_2D(80, 60);
BM_STATIC_ALL_2D(81, 61);
BM_STATIC_ALL_2D(800, 600);
BM_STATIC_ALL_2D(802, 602);

////////////////////////////////////////////////////////////////////////////////
// Reduce tensor<MxNxKxf32>.
////////////////////////////////////////////////////////////////////////////////

#define BM_DYNAMIC_ALL_3D(SUFFIX, M, N, K, N_DIMS_TO_REDUCE, DIMS_TO_REDUCE) \
  BM_SUITE_SUM_F32(Sum3DColDynamicAll_##SUFFIX##_##M##_##N##_##K, 3,         \
                   INTS(M, N, K),                                            \
                   BOOLS(kDynamicDim, kDynamicDim, kDynamicDim),             \
                   N_DIMS_TO_REDUCE, INTS(DIMS_TO_REDUCE))

// Reduction dimensions = {0}
BM_DYNAMIC_ALL_3D(Dim0, 2, 80, 2, 1, INTS(0));
BM_DYNAMIC_ALL_3D(Dim0, 8, 6, 8, 1, INTS(0));
BM_DYNAMIC_ALL_3D(Dim0, 80, 1, 80, 1, INTS(0));
BM_DYNAMIC_ALL_3D(Dim0, 80, 60, 80, 1, INTS(0));
BM_DYNAMIC_ALL_3D(Dim0, 81, 61, 81, 1, INTS(0));

// Reduction dimensions = {0, 1}
BM_DYNAMIC_ALL_3D(Dims01, 2, 80, 2, 2, INTS(0, 1));
BM_DYNAMIC_ALL_3D(Dims01, 8, 6, 8, 2, INTS(0, 1));
BM_DYNAMIC_ALL_3D(Dims01, 80, 1, 80, 2, INTS(0, 1));
BM_DYNAMIC_ALL_3D(Dims01, 80, 60, 80, 2, INTS(0, 1));
BM_DYNAMIC_ALL_3D(Dims01, 81, 61, 81, 2, INTS(0, 1));

////////////////////////////////////////////////////////////////////////////////
// Reduce tensor<MxNxLxKxf32> -> tensor<LxKxf32>.
////////////////////////////////////////////////////////////////////////////////

#define BM_STATIC_ALL_4D(M, N, K, L)                                      \
  BM_SUITE_SUM_F32(                                                       \
      Sum4DColStaticAll_Dim01_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kStaticDim, kStaticDim, kStaticDim, kStaticDim), 2, INTS(0, 1))

BM_STATIC_ALL_4D(2, 80, 2, 80);
BM_STATIC_ALL_4D(8, 6, 8, 6);
BM_STATIC_ALL_4D(80, 1, 80, 1);
BM_STATIC_ALL_4D(80, 60, 80, 60);
BM_STATIC_ALL_4D(81, 61, 81, 61);

#define BM_STATIC_01_4D(M, N, K, L)                                       \
  BM_SUITE_SUM_F32(                                                       \
      Sum4DColStatic01_Dims01_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kDynamicDim, kDynamicDim, kStaticDim, kStaticDim), 2, INTS(0, 1))

BM_STATIC_01_4D(2, 80, 2, 80);
BM_STATIC_01_4D(8, 6, 8, 6);
BM_STATIC_01_4D(80, 1, 80, 1);
BM_STATIC_01_4D(80, 60, 80, 60);
BM_STATIC_01_4D(81, 61, 81, 61);

#define BM_STATIC_012_4D(M, N, K, L)                                       \
  BM_SUITE_SUM_F32(                                                        \
      Sum4DColStatic012_Dims01_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kStaticDim, kStaticDim, kStaticDim, kDynamicDim), 2, INTS(0, 1))

BM_STATIC_012_4D(2, 80, 2, 80);
BM_STATIC_012_4D(8, 6, 8, 6);
BM_STATIC_012_4D(80, 1, 80, 1);
BM_STATIC_012_4D(80, 60, 80, 60);
BM_STATIC_012_4D(81, 61, 81, 61);

#define BM_DYNAMIC_ALL_4D(M, N, K, L)                                      \
  BM_SUITE_SUM_F32(                                                        \
      Sum4DColDynamicAll_Dim01_##M##_##N##_##K##_##L, 4, INTS(M, N, K, L), \
      BOOLS(kDynamicDim, kDynamicDim, kStaticDim, kDynamicDim), 2, INTS(0, 1))

BM_DYNAMIC_ALL_4D(2, 80, 2, 80);
BM_DYNAMIC_ALL_4D(8, 6, 8, 6);
BM_DYNAMIC_ALL_4D(80, 1, 80, 1);
BM_DYNAMIC_ALL_4D(80, 60, 80, 60);
BM_DYNAMIC_ALL_4D(81, 61, 81, 61);

}  // namespace
}  // namespace tensorflow
