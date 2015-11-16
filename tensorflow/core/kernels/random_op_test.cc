#include <random>

#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

Tensor Int32(int32 v) {
  Tensor t(DT_INT32, TensorShape({}));
  t.scalar<int32>()() = v;
  return t;
}

Graph* RandomUniform(int64 n) {
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::RandomUniform(g, test::graph::Constant(g, Int32(n)), DT_FLOAT);
  return g;
}

Graph* RandomNormal(int64 n) {
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::RandomGaussian(g, test::graph::Constant(g, Int32(n)), DT_FLOAT);
  return g;
}

Graph* RandomParameters(int64 n) {
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::RandomParameters(g, test::graph::Constant(g, Int32(n)),
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
BM_RNG(cpu, RandomParameters);

BM_RNG(gpu, RandomUniform);
BM_RNG(gpu, RandomNormal);
BM_RNG(gpu, RandomParameters);

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
