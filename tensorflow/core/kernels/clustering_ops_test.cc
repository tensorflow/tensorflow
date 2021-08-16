// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
// ==============================================================================

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

constexpr int k100Dim = 100;
// Number of points for tests.
constexpr int k10Points = 10;
constexpr int k100Points = 100;
constexpr int k1kPoints = 1000;
constexpr int k10kPoints = 10000;
constexpr int k1MPoints = 1000000;
// Number of centers for tests.
constexpr int k2Centers = 2;
constexpr int k5Centers = 5;
constexpr int k10Centers = 10;
constexpr int k20Centers = 20;
constexpr int k50Centers = 50;
constexpr int k100Centers = 100;
constexpr int k200Centers = 200;
constexpr int k500Centers = 500;
constexpr int k1kCenters = 1000;
constexpr int k10kCenters = 10000;
// Number of retries for tests.
constexpr int k0RetriesPerSample = 0;
constexpr int k3RetriesPerSample = 3;

Graph* SetUpKmeansPlusPlusInitialization(int num_dims, int num_points,
                                         int num_to_sample,
                                         int retries_per_sample) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor points(DT_FLOAT, TensorShape({num_points, num_dims}));
  Tensor sample_size(DT_INT64, TensorShape({}));
  Tensor seed(DT_INT64, TensorShape({}));
  Tensor num_retries_per_sample(DT_INT64, TensorShape({}));
  points.flat<float>().setRandom();
  sample_size.flat<int64_t>().setConstant(num_to_sample);
  seed.flat<int64_t>().setConstant(12345);
  num_retries_per_sample.flat<int64_t>().setConstant(retries_per_sample);

  TF_CHECK_OK(NodeBuilder("kmeans_plus_plus_initialization_op",
                          "KmeansPlusPlusInitialization")
                  .Input(test::graph::Constant(g, points))
                  .Input(test::graph::Constant(g, sample_size))
                  .Input(test::graph::Constant(g, seed))
                  .Input(test::graph::Constant(g, num_retries_per_sample))
                  .Finalize(g, nullptr /* node */));
  return g;
}

template <int num_points, int num_to_sample, int num_dims,
          int retries_per_sample>
void BM_KmeansPlusPlusInitialization(::testing::benchmark::State& state) {
  Graph* g = SetUpKmeansPlusPlusInitialization(
      num_dims, num_points, num_to_sample, retries_per_sample);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_points * num_dims * num_to_sample);
}

#define BENCHMARK_KMEANS_PLUS_PLUS(p, c, d, r)                     \
  void BM_KmeansPlusPlusInitialization_##p##_##c##_##d##_##r(      \
      ::testing::benchmark::State& state) {                        \
    BM_KmeansPlusPlusInitialization<p, c, d, r>(state);            \
  }                                                                \
  BENCHMARK(BM_KmeansPlusPlusInitialization_##p##_##c##_##d##_##r) \
      ->UseRealTime();

#define RUN_BM_KmeansPlusPlusInitialization(retries)                     \
  BENCHMARK_KMEANS_PLUS_PLUS(k10Points, k2Centers, k100Dim, retries);    \
  BENCHMARK_KMEANS_PLUS_PLUS(k10Points, k5Centers, k100Dim, retries);    \
  BENCHMARK_KMEANS_PLUS_PLUS(k10Points, k10Centers, k100Dim, retries);   \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k10Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k20Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k50Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k100Points, k100Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k100Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k200Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k500Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1kPoints, k1kCenters, k100Dim, retries);   \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k100Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k200Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k500Centers, k100Dim, retries); \
  BENCHMARK_KMEANS_PLUS_PLUS(k10kPoints, k1kCenters, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k100Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k200Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k500Centers, k100Dim, retries);  \
  BENCHMARK_KMEANS_PLUS_PLUS(k1MPoints, k1kCenters, k100Dim, retries)

RUN_BM_KmeansPlusPlusInitialization(k0RetriesPerSample);
RUN_BM_KmeansPlusPlusInitialization(k3RetriesPerSample);

#undef RUN_BM_KmeansPlusPlusInitialization
#undef BENCHMARK_KMEANS_PLUS_PLUS

Graph* SetUpKMC2Initialization(int num_points) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor distances(DT_FLOAT, TensorShape({num_points}));
  Tensor seed(DT_INT64, TensorShape({}));
  distances.flat<float>().setRandom();
  seed.flat<int64_t>().setConstant(12345);

  TF_CHECK_OK(
      NodeBuilder("KMC2ChainInitializationOp", "KMC2ChainInitialization")
          .Input(test::graph::Constant(g, distances))
          .Input(test::graph::Constant(g, seed))
          .Finalize(g, nullptr /* node */));
  return g;
}

template <int num_points, int num_to_sample, int num_dims>
void BM_KMC2Initialization(::testing::benchmark::State& state) {
  Graph* g = SetUpKMC2Initialization(num_points);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_points * num_dims * num_to_sample);
}
#define BENCHMARK_KMC2(p, c, d)               \
  void BM_KMC2Initialization_##p##_##c##_##d( \
      ::testing::benchmark::State& state) {   \
    BM_KMC2Initialization<p, c, d>(state);    \
  }                                           \
  BENCHMARK(BM_KMC2Initialization_##p##_##c##_##d)->UseRealTime();

#define RUN_BM_KMC2Initialization                   \
  BENCHMARK_KMC2(k10Points, k2Centers, k100Dim);    \
  BENCHMARK_KMC2(k10Points, k5Centers, k100Dim);    \
  BENCHMARK_KMC2(k10Points, k10Centers, k100Dim);   \
  BENCHMARK_KMC2(k100Points, k10Centers, k100Dim);  \
  BENCHMARK_KMC2(k100Points, k20Centers, k100Dim);  \
  BENCHMARK_KMC2(k100Points, k50Centers, k100Dim);  \
  BENCHMARK_KMC2(k100Points, k100Centers, k100Dim); \
  BENCHMARK_KMC2(k1kPoints, k100Centers, k100Dim);  \
  BENCHMARK_KMC2(k1kPoints, k200Centers, k100Dim);  \
  BENCHMARK_KMC2(k1kPoints, k500Centers, k100Dim);  \
  BENCHMARK_KMC2(k1kPoints, k1kCenters, k100Dim);   \
  BENCHMARK_KMC2(k10kPoints, k100Centers, k100Dim); \
  BENCHMARK_KMC2(k10kPoints, k200Centers, k100Dim); \
  BENCHMARK_KMC2(k10kPoints, k500Centers, k100Dim); \
  BENCHMARK_KMC2(k10kPoints, k1kCenters, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k100Centers, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k200Centers, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k500Centers, k100Dim);  \
  BENCHMARK_KMC2(k1MPoints, k1kCenters, k100Dim)

RUN_BM_KMC2Initialization;
#undef RUN_BM_KMC2Initialization
#undef BENCHMARK_KMC2

Graph* SetUpNearestNeighbors(int num_dims, int num_points, int num_centers,
                             int k) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor points(DT_FLOAT, TensorShape({num_points, num_dims}));
  Tensor centers(DT_FLOAT, TensorShape({num_centers, num_dims}));
  Tensor top(DT_INT64, TensorShape({}));
  points.flat<float>().setRandom();
  centers.flat<float>().setRandom();
  top.flat<int64_t>().setConstant(k);

  TF_CHECK_OK(NodeBuilder("nearest_centers_op", "NearestNeighbors")
                  .Input(test::graph::Constant(g, points))
                  .Input(test::graph::Constant(g, centers))
                  .Input(test::graph::Constant(g, top))
                  .Finalize(g, nullptr /* node */));
  return g;
}

template <int num_dims, int num_points, int num_centers, int k>
void BM_NearestNeighbors(::testing::benchmark::State& state) {
  Graph* g = SetUpNearestNeighbors(num_dims, num_points, num_centers, k);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_points * num_dims * num_centers);
}

constexpr int kTop1 = 1;
constexpr int kTop2 = 2;
constexpr int kTop5 = 5;
constexpr int kTop10 = 10;

#define BENCHMARK_NEAREST_NEIGHBORS(d, p, c, k)  \
  void BM_NearestNeighbors##d##_##p##_##c##_##k( \
      ::testing::benchmark::State& state) {      \
    BM_NearestNeighbors<d, p, c, k>(state);      \
  }                                              \
  BENCHMARK(BM_NearestNeighbors##d##_##p##_##c##_##k)->UseRealTime();

#define RUN_BM_NearestNeighbors(k)                                 \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1kPoints, k100Centers, k); \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1kPoints, k1kCenters, k);  \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1kPoints, k10kCenters, k); \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1MPoints, k100Centers, k); \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1MPoints, k1kCenters, k);  \
  BENCHMARK_NEAREST_NEIGHBORS(k100Dim, k1MPoints, k10kCenters, k)

RUN_BM_NearestNeighbors(kTop1);
// k > 1
RUN_BM_NearestNeighbors(kTop2);
RUN_BM_NearestNeighbors(kTop5);
RUN_BM_NearestNeighbors(kTop10);

#undef RUN_BM_NearestNeighbors
#undef BENCHMARK_NEAREST_NEIGHBORS
}  // namespace
}  // namespace tensorflow
