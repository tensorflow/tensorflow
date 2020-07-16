/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/bias_op.h"

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* BiasAdd(int d0, int d1, int d2, int d3) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, TensorShape({d0, d1, d2, d3}));
  Tensor bias(DT_FLOAT, TensorShape({d3}));
  input.flat<float>().setRandom();
  bias.flat<float>().setRandom();
  test::graph::Binary(g, "BiasAdd", test::graph::Constant(g, input),
                      test::graph::Constant(g, bias));
  return g;
}

static Graph* BiasAddGrad(int d0, int d1, int d2, int d3) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor out_backprop(DT_FLOAT, TensorShape({d0, d1, d2, d3}));
  out_backprop.flat<float>().setRandom();
  test::graph::Unary(g, "BiasAddGrad", test::graph::Constant(g, out_backprop));
  return g;
}

#define BM_BiasAddNHWC(N, W, H, C, DEVICE)                                   \
  static void BM_BiasAddNHWC##_##N##_##H##_##W##_##C##_##DEVICE(int iters) { \
    testing::UseRealTime();                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters) * N * H * W * C);      \
    test::Benchmark(#DEVICE, BiasAdd(N, H, W, C)).Run(iters);                \
  }                                                                          \
  BENCHMARK(BM_BiasAddNHWC##_##N##_##H##_##W##_##C##_##DEVICE);

#define BM_BiasAddGradNHWC(N, W, H, C, DEVICE)                          \
  static void BM_BiasAddGradNHWC##_##N##_##H##_##W##_##C##_##DEVICE(    \
      int iters) {                                                      \
    testing::UseRealTime();                                             \
    testing::ItemsProcessed(static_cast<int64>(iters) * N * H * W * C); \
    test::Benchmark(#DEVICE, BiasAddGrad(N, H, W, C)).Run(iters);       \
  }                                                                     \
  BENCHMARK(BM_BiasAddGradNHWC##_##N##_##H##_##W##_##C##_##DEVICE);

// CPU
BM_BiasAddNHWC(32, 32, 32, 128, cpu);
BM_BiasAddNHWC(32, 32, 32, 256, cpu);
BM_BiasAddNHWC(32, 32, 32, 512, cpu);
BM_BiasAddNHWC(32, 32, 32, 1024, cpu);

BM_BiasAddNHWC(32, 64, 64, 128, cpu);
BM_BiasAddNHWC(32, 64, 64, 256, cpu);
BM_BiasAddNHWC(32, 64, 64, 512, cpu);
BM_BiasAddNHWC(32, 64, 64, 1024, cpu);

BM_BiasAddGradNHWC(32, 32, 32, 128, cpu);
BM_BiasAddGradNHWC(32, 32, 32, 256, cpu);
BM_BiasAddGradNHWC(32, 32, 32, 512, cpu);
BM_BiasAddGradNHWC(32, 32, 32, 1024, cpu);

BM_BiasAddGradNHWC(32, 64, 64, 128, cpu);
BM_BiasAddGradNHWC(32, 64, 64, 256, cpu);
BM_BiasAddGradNHWC(32, 64, 64, 512, cpu);
BM_BiasAddGradNHWC(32, 64, 64, 1024, cpu);

#ifdef GOOGLE_CUDA
BM_BiasAddGradNHWC(32, 32, 32, 128, gpu);
BM_BiasAddGradNHWC(32, 32, 32, 256, gpu);
BM_BiasAddGradNHWC(32, 32, 32, 512, gpu);
BM_BiasAddGradNHWC(32, 32, 32, 1024, gpu);

BM_BiasAddGradNHWC(32, 64, 64, 128, gpu);
BM_BiasAddGradNHWC(32, 64, 64, 256, gpu);
BM_BiasAddGradNHWC(32, 64, 64, 512, gpu);
BM_BiasAddGradNHWC(32, 64, 64, 1024, gpu);
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
