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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* RandomBinomialGraph(double count, double prob, int num_batches,
                                  int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32>().setValues({num_batches, samples_per_batch});

  Tensor counts_t(DT_FLOAT, TensorShape({num_batches}));
  counts_t.flat<float>().setConstant(count);
  Tensor probs_t(DT_FLOAT, TensorShape({num_batches}));
  probs_t.flat<float>().setConstant(prob);

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("randombinomial"), "RandomBinomial")
                  .Input(test::graph::Constant(g, shape_t))
                  .Input(test::graph::Constant(g, counts_t))
                  .Input(test::graph::Constant(g, probs_t))
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(g, &ret));
  return g;
}

static Graph* RandomBinomialInv(int num_batches, int samples_per_batch) {
  // Because counts * probs < 10, we are guaranteed to use inversion.
  return RandomBinomialGraph(10., 0.3, num_batches, samples_per_batch);
}

static Graph* RandomBinomialRej(int num_batches, int samples_per_batch) {
  // Because counts * probs > 10, we are guaranteed to use rejection.
  return RandomBinomialGraph(100., 0.3, num_batches, samples_per_batch);
}

static Graph* RandomBinomialInvComplement(int num_batches,
                                          int samples_per_batch) {
  // Because counts * (1 - probs) < 10, we are guaranteed to use inversion.
  return RandomBinomialGraph(10., 0.8, num_batches, samples_per_batch);
}

static Graph* RandomBinomialRejComplement(int num_batches,
                                          int samples_per_batch) {
  // Because counts * (1 - probs) > 10, we are guaranteed to use inversion.
  return RandomBinomialGraph(100., 0.2, num_batches, samples_per_batch);
}

#define BM_RandomBinomialInv(DEVICE, B, S)                                     \
  static void BM_RandomBinomialInv_##DEVICE##_##B##_##S(                       \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, RandomBinomialInv(B, S),                          \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_RandomBinomialInv_##DEVICE##_##B##_##S);

#define BM_RandomBinomialRej(DEVICE, B, S)                                     \
  static void BM_RandomBinomialRej_##DEVICE##_##B##_##S(                       \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, RandomBinomialRej(B, S),                          \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_RandomBinomialRej_##DEVICE##_##B##_##S);

#define BM_RandomBinomialInvComplement(DEVICE, B, S)                           \
  static void BM_RandomBinomialInvComplement_##DEVICE##_##B##_##S(             \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, RandomBinomialInvComplement(B, S),                \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_RandomBinomialInvComplement_##DEVICE##_##B##_##S);

#define BM_RandomBinomialRejComplement(DEVICE, B, S)                           \
  static void BM_RandomBinomialRejComplement_##DEVICE##_##B##_##S(             \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, RandomBinomialRejComplement(B, S),                \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_RandomBinomialRejComplement_##DEVICE##_##B##_##S);

BM_RandomBinomialInv(cpu, 1000, 1000);
BM_RandomBinomialRej(cpu, 1000, 1000);
BM_RandomBinomialInvComplement(cpu, 1000, 1000);
BM_RandomBinomialRejComplement(cpu, 1000, 1000);
BM_RandomBinomialInv(gpu, 1000, 1000);
BM_RandomBinomialRej(gpu, 1000, 1000);
BM_RandomBinomialInvComplement(gpu, 1000, 1000);
BM_RandomBinomialRejComplement(gpu, 1000, 1000);

}  // namespace tensorflow
