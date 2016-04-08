/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

Tensor VecShape(int32 v) {
  Tensor shape(DT_INT32, TensorShape({1}));
  shape.vec<int32>()(0) = v;
  return shape;
}

Graph* RandomUniform(int64 n) {
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::RandomUniform(g, test::graph::Constant(g, VecShape(n)),
                             DT_FLOAT);
  return g;
}

Graph* RandomNormal(int64 n) {
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::RandomGaussian(g, test::graph::Constant(g, VecShape(n)),
                              DT_FLOAT);
  return g;
}

Graph* TruncatedNormal(int64 n) {
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::TruncatedNormal(g, test::graph::Constant(g, VecShape(n)),
                               DT_FLOAT);
  return g;
}

#define BM_RNG(DEVICE, RNG)                                   \
  static void BM_##DEVICE##_##RNG(int iters, int arg) {       \
    testing::ItemsProcessed(static_cast<int64>(iters) * arg); \
    test::Benchmark(#DEVICE, RNG(arg)).Run(iters);            \
  }                                                           \
  BENCHMARK(BM_##DEVICE##_##RNG)->Range(1 << 20, 8 << 20);

BM_RNG(cpu, RandomUniform);
BM_RNG(cpu, RandomNormal);
BM_RNG(cpu, TruncatedNormal);

BM_RNG(gpu, RandomUniform);
BM_RNG(gpu, RandomNormal);
BM_RNG(gpu, TruncatedNormal);

static void BM_PhiloxRandom(int iters) {
  // Fill 2M random numbers
  int count = 2 << 20;

  testing::ItemsProcessed(static_cast<int64>(iters) * count);

  random::PhiloxRandom gen(0x12345);

  int val = 1;
  for (int i = 0; i < iters; ++i) {
    for (int j = 0; j < count; j += 4) {
      /// each invocation of gen() returns 128-bit samples
      auto samples = gen();

      // use the result trivially so the compiler does not optimize it away
      val ^= samples[0] ^ samples[1] ^ samples[2] ^ samples[3];
    }
  }

  // A anchor point to make sure the compiler does not cut corners
  CHECK(val) << val;
}
BENCHMARK(BM_PhiloxRandom);

static void BM_StdMTRandom(int iters) {
  // Fill 2M random numbers
  int count = 2 << 20;

  testing::ItemsProcessed(static_cast<int64>(iters) * count);

  std::mt19937 gen(0x12345);

  int val = 1;
  for (int i = 0; i < iters; ++i) {
    for (int j = 0; j < count; ++j) {
      /// each invocation of gen() returns 32-bit sample
      uint32 sample = gen();

      // use the result trivially so the compiler does not optimize it away
      val ^= sample;
    }
  }

  // A anchor point to make sure the compiler does not cut corners
  CHECK(val) << val;
}
BENCHMARK(BM_StdMTRandom);

}  // end namespace tensorflow
