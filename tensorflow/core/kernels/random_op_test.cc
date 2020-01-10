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

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

Tensor VecShape(int64 v) {
  if (v >= std::numeric_limits<int32>::max()) {
    Tensor shape(DT_INT64, TensorShape({1}));
    shape.vec<int64>()(0) = v;
    return shape;
  } else {
    Tensor shape(DT_INT32, TensorShape({1}));
    shape.vec<int32>()(0) = v;
    return shape;
  }
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
  void BM_##DEVICE##_##RNG(int iters, int arg) {              \
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

Tensor VecAlphas(int64 n) {
  Tensor alphas(DT_DOUBLE, TensorShape({n}));
  for (int i = 0; i < n; i++) {
    // Alternate back and forth between small-and-growing (.25) and
    // large-and-shrinking (26.67) alpha.
    alphas.vec<double>()(i) =
        0.25 + MathUtil::IPow(1.1, i % 2 == 0 ? i : n - i);
  }
  return alphas;
}

void BM_cpu_RandomGamma(int iters, int nsamp, int nalpha) {
  testing::ItemsProcessed(static_cast<int64>(iters) * nsamp * nalpha);
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::RandomGamma(g, test::graph::Constant(g, VecShape(nsamp)),
                           test::graph::Constant(g, VecAlphas(nalpha)));
  test::Benchmark("cpu", g).Run(iters);
}
BENCHMARK(BM_cpu_RandomGamma)->RangePair(1 << 14, 4 << 15, 2, 50);

void BM_PhiloxRandom(int iters) {
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

void BM_StdMTRandom(int iters) {
  // Fill 2M random numbers
  int count = 2 << 20;

  testing::ItemsProcessed(static_cast<int64>(iters) * count);

  std::mt19937 gen(0x12345);

  uint_fast32_t val = 1;
  for (int i = 0; i < iters; ++i) {
    for (int j = 0; j < count; ++j) {
      /// each invocation of gen() returns 32-bit sample
      uint_fast32_t sample = gen();

      // use the result trivially so the compiler does not optimize it away
      val ^= sample;
    }
  }

  // A anchor point to make sure the compiler does not cut corners
  CHECK(val) << val;
}
BENCHMARK(BM_StdMTRandom);

}  // namespace
}  // namespace tensorflow
