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

#define BM_DYNAMIC_ALL(ROWS, COLS)                                          \
  BM_SUITE_MEAN_F32(MeanRowDynamicAll_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                    BOOLS(kDynamicDim, kDynamicDim), 1, INTS(1))
BM_DYNAMIC_ALL(2, 80);
BM_DYNAMIC_ALL(8, 6);
BM_DYNAMIC_ALL(80, 1);
BM_DYNAMIC_ALL(80, 60);
BM_DYNAMIC_ALL(81, 61);
BM_DYNAMIC_ALL(800, 600);
BM_DYNAMIC_ALL(802, 602);

#define BM_STATIC_ROW(ROWS, COLS)                                         \
  BM_SUITE_MEAN_F32(MeanRowStaticRow##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                    BOOLS(kStaticDim, kDynamicDim), 1, INTS(1))
BM_STATIC_ROW(2, 80);
BM_STATIC_ROW(8, 6);
BM_STATIC_ROW(80, 1);
BM_STATIC_ROW(80, 60);
BM_STATIC_ROW(81, 61);
BM_STATIC_ROW(800, 600);
BM_STATIC_ROW(802, 602);

#define BM_STATIC_COL(ROWS, COLS)                                          \
  BM_SUITE_MEAN_F32(MeanRowStaticCol_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                    BOOLS(kDynamicDim, kStaticDim), 1, INTS(1))
BM_STATIC_COL(2, 80);
BM_STATIC_COL(8, 6);
BM_STATIC_COL(80, 1);
BM_STATIC_COL(80, 60);
BM_STATIC_COL(81, 61);
BM_STATIC_COL(800, 600);
BM_STATIC_COL(802, 602);

#define BM_STATIC_ALL(ROWS, COLS)                                          \
  BM_SUITE_MEAN_F32(MeanRowStaticAll_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
                    BOOLS(kStaticDim, kStaticDim), 1, INTS(1))
BM_STATIC_ALL(2, 80);
BM_STATIC_ALL(8, 6);
BM_STATIC_ALL(80, 1);
BM_STATIC_ALL(80, 60);
BM_STATIC_ALL(81, 61);
BM_STATIC_ALL(800, 600);
BM_STATIC_ALL(802, 602);

}  // namespace
}  // namespace tensorflow
