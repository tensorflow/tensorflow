/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

static Graph* PTruncatedNormal(int num_batches, int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32>().setValues({num_batches, samples_per_batch});

  // Use mean 0 and stdev 1
  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);

  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setRandom();
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(5.0);

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

static Graph* PTruncatedNormal2SD(int num_batches, int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(-2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(2.0);

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

static Graph* PTruncatedNormalOneTail(int num_batches, int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(std::numeric_limits<float>::infinity());

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

#define BM_PTruncatedNormalDev(DEVICE, B, S)                        \
  static void BM_PTruncatedNormal_##DEVICE##_##B##_##S(int iters) { \
    test::Benchmark(#DEVICE, PTruncatedNormal(B, S)).Run(iters);    \
    testing::ItemsProcessed(static_cast<int64>(B) * S * iters);     \
  }                                                                 \
  BENCHMARK(BM_PTruncatedNormal_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_2SD(DEVICE, B, S)                        \
  static void BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S(int iters) { \
    test::Benchmark(#DEVICE, PTruncatedNormal2SD(B, S)).Run(iters);     \
    testing::ItemsProcessed(static_cast<int64>(B) * S * iters);         \
  }                                                                     \
  BENCHMARK(BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_OneTail(DEVICE, B, S)                        \
  static void BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S(int iters) { \
    test::Benchmark(#DEVICE, PTruncatedNormalOneTail(B, S)).Run(iters);     \
    testing::ItemsProcessed(static_cast<int64>(B) * S * iters);             \
  }                                                                         \
  BENCHMARK(BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S);

BM_PTruncatedNormalDev(cpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(cpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(cpu, 10000, 100);
BM_PTruncatedNormalDev(gpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(gpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(gpu, 10000, 100);

}  // namespace tensorflow
