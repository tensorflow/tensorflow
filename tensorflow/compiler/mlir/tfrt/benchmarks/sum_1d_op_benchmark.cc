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

#define BM_DYNAMIC(SIZE)                                                    \
  BM_SUITE_SUM_F32(SumDynamic_##SIZE, 1, INTS(SIZE), BOOLS(kDynamicDim), 1, \
                   INTS(0))
BM_DYNAMIC(3);
BM_DYNAMIC(8);
BM_DYNAMIC(80);
BM_DYNAMIC(800);
BM_DYNAMIC(8000);
BM_DYNAMIC(8131);
BM_DYNAMIC(1000000);
BM_DYNAMIC(1010131);

#define BM_STATIC(SIZE)                                                   \
  BM_SUITE_SUM_F32(SumStatic_##SIZE, 1, INTS(SIZE), BOOLS(kStaticDim), 1, \
                   INTS(0))
BM_STATIC(3);
BM_STATIC(8);
BM_STATIC(80);
BM_STATIC(800);
BM_STATIC(8000);
BM_STATIC(8131);
BM_STATIC(1000000);
BM_STATIC(1010131);

}  // namespace
}  // namespace tensorflow
