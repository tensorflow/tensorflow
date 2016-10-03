/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
static Graph* BatchMatmul(int b, int m, int k, int n, bool adjoint_a,
                          bool adjoint_b, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, adjoint_a ? TensorShape({b, k, m}) : TensorShape({b, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, adjoint_b ? TensorShape({b, n, k}) : TensorShape({b, k, n}));
  in1.flat<T>().setRandom();
  test::graph::BatchMatmul(g, test::graph::Constant(g, in0),
                           test::graph::Constant(g, in1), adjoint_a, adjoint_b);
  return g;
}

#define BM_BatchMatmulDev(B, M, K, N, TA, TB, T, TFTYPE, DEVICE)                  \
  static void                                                                     \
      BM_BatchMatmul##_##B##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
          int iters) {                                                            \
    testing::UseRealTime();                                                       \
    testing::ItemsProcessed(static_cast<int64>(iters) * B * M * K * N * 2);       \
    test::Benchmark(#DEVICE, BatchMatmul<T>(B, M, K, N, TA, TB, TFTYPE))          \
        .Run(iters);                                                              \
  }                                                                               \
  BENCHMARK(                                                                      \
      BM_BatchMatmul##_##B##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE);

#define BM_BatchMatmul(B, M, K, N, TA, TB) \
  BM_BatchMatmulDev(B, M, K, N, TA, TB, float, DT_FLOAT, cpu);
// BM_BatchMatmulDev(B, M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64,
// cpu);
//  BM_BatchMatmulDev(B, M, K, N, TA, TB, float, DT_FLOAT, gpu);
/* Uncomment to enable benchmarks for double & complex types: */
// BM_BatchMatmulDev(B, M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64,
// gpu);
// BM_BatchMatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu);                    \
// BM_BatchMatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu);  \
// BM_BatchMatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu);                    \
// BM_BatchMatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

// Typical fully connected layers
BM_BatchMatmul(1, 1, 1024, 1024, false, false);
BM_BatchMatmul(1, 8, 1024, 1024, false, false);
BM_BatchMatmul(1, 16, 1024, 1024, false, false);
BM_BatchMatmul(1, 128, 1024, 1024, false, false);
BM_BatchMatmul(2, 1, 1024, 1024, false, false);
BM_BatchMatmul(2, 8, 1024, 1024, false, false);
BM_BatchMatmul(2, 16, 1024, 1024, false, false);
BM_BatchMatmul(2, 128, 1024, 1024, false, false);
BM_BatchMatmul(8, 1, 1024, 1024, false, false);
BM_BatchMatmul(8, 8, 1024, 1024, false, false);
BM_BatchMatmul(8, 16, 1024, 1024, false, false);
BM_BatchMatmul(8, 128, 1024, 1024, false, false);
BM_BatchMatmul(32, 1, 1024, 1024, false, false);
BM_BatchMatmul(32, 8, 1024, 1024, false, false);
BM_BatchMatmul(32, 16, 1024, 1024, false, false);
BM_BatchMatmul(32, 128, 1024, 1024, false, false);

// Square matmul.
BM_BatchMatmul(1, 32, 32, 32, false, false);
BM_BatchMatmul(1, 128, 128, 128, false, false);
BM_BatchMatmul(1, 256, 256, 256, false, false);
BM_BatchMatmul(1, 1024, 1024, 1024, false, false);
BM_BatchMatmul(1, 2048, 2048, 2048, false, false);
BM_BatchMatmul(2, 32, 32, 32, false, false);
BM_BatchMatmul(2, 128, 128, 128, false, false);
BM_BatchMatmul(2, 256, 256, 256, false, false);
BM_BatchMatmul(2, 1024, 1024, 1024, false, false);
BM_BatchMatmul(2, 2048, 2048, 2048, false, false);
BM_BatchMatmul(4, 32, 32, 32, false, false);
BM_BatchMatmul(4, 128, 128, 128, false, false);
BM_BatchMatmul(4, 256, 256, 256, false, false);
BM_BatchMatmul(4, 1024, 1024, 1024, false, false);
BM_BatchMatmul(4, 2048, 2048, 2048, false, false);
BM_BatchMatmul(8, 32, 32, 32, false, false);
BM_BatchMatmul(8, 128, 128, 128, false, false);
BM_BatchMatmul(8, 256, 256, 256, false, false);
BM_BatchMatmul(8, 1024, 1024, 1024, false, false);
BM_BatchMatmul(8, 2048, 2048, 2048, false, false);
BM_BatchMatmul(32, 32, 32, 32, false, false);
BM_BatchMatmul(32, 128, 128, 128, false, false);
BM_BatchMatmul(32, 256, 256, 256, false, false);
BM_BatchMatmul(32, 1024, 1024, 1024, false, false);
BM_BatchMatmul(32, 2048, 2048, 2048, false, false);

// Matrix-vector multiplies.
BM_BatchMatmul(1, 10000, 200, 1, false, false);
BM_BatchMatmul(8, 10000, 200, 1, false, false);
BM_BatchMatmul(32, 10000, 200, 1, false, false);
BM_BatchMatmul(1, 10000, 200, 1, true, false);
BM_BatchMatmul(8, 10000, 200, 1, true, false);
BM_BatchMatmul(32, 10000, 200, 1, true, false);
BM_BatchMatmul(1, 10000, 200, 1, false, true);
BM_BatchMatmul(8, 10000, 200, 1, false, true);
BM_BatchMatmul(32, 10000, 200, 1, false, true);
BM_BatchMatmul(1, 10000, 200, 1, true, true);
BM_BatchMatmul(8, 10000, 200, 1, true, true);
BM_BatchMatmul(32, 10000, 200, 1, true, true);

// Vector-matrix multiplies.
BM_BatchMatmul(1, 1, 200, 10000, false, false);
BM_BatchMatmul(8, 1, 200, 10000, false, false);
BM_BatchMatmul(32, 1, 200, 10000, false, false);
BM_BatchMatmul(1, 1, 200, 10000, true, false);
BM_BatchMatmul(8, 1, 200, 10000, true, false);
BM_BatchMatmul(32, 1, 200, 10000, true, false);
BM_BatchMatmul(1, 1, 200, 10000, false, true);
BM_BatchMatmul(8, 1, 200, 10000, false, true);
BM_BatchMatmul(32, 1, 200, 10000, false, true);
BM_BatchMatmul(1, 1, 200, 10000, true, true);
BM_BatchMatmul(8, 1, 200, 10000, true, true);
BM_BatchMatmul(32, 1, 200, 10000, true, true);

}  // end namespace tensorflow
