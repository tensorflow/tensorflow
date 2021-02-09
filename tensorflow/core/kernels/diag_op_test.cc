/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <typename T>
static Graph* Diag(int n, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(type, TensorShape({n}));
  in.flat<T>().setRandom();
  Node* out = test::graph::Diag(g, test::graph::Constant(g, in), type);
  test::graph::DiagPart(g, out, type);
  return g;
}

#define BM_DiagDev(N, T, TFTYPE, DEVICE)                                      \
  static void BM_Diag##_##N##_##TFTYPE##_##DEVICE(                            \
      ::testing::benchmark::State& state) {                                   \
    test::Benchmark(#DEVICE, Diag<T>(N, TFTYPE), /*old_benchmark_api=*/false) \
        .Run(state);                                                          \
    state.SetItemsProcessed(static_cast<int64>(state.iterations()) * N * N);  \
  }                                                                           \
  BENCHMARK(BM_Diag##_##N##_##TFTYPE##_##DEVICE);

#define BM_Diag(N)                                       \
  BM_DiagDev(N, int, DT_INT32, cpu);                     \
  BM_DiagDev(N, float, DT_FLOAT, cpu);                   \
  BM_DiagDev(N, std::complex<float>, DT_COMPLEX64, cpu); \
  BM_DiagDev(N, int, DT_INT32, gpu);                     \
  BM_DiagDev(N, float, DT_FLOAT, gpu);                   \
  BM_DiagDev(N, std::complex<float>, DT_COMPLEX64, gpu);

BM_Diag(16);
BM_Diag(128);
BM_Diag(512);

}  // end namespace tensorflow
