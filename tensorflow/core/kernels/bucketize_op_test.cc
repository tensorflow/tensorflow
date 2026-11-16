/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class BucketizeOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType dtype, const std::vector<float>& boundaries) {
    TF_ASSERT_OK(NodeDefBuilder("bucketize", "Bucketize")
                     .Input(FakeInput(dtype))
                     .Attr("boundaries", boundaries)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(BucketizeOpTest, Float) {
  MakeOp(DT_FLOAT, {0.0f, 10.0f, 100.0f});
  AddInputFromArray<float>(TensorShape({2, 4}),
                           {-5.f, 0.f, 0.5f, 10.f, 11.f, 99.f, 100.f, 1e9f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_INT32, TensorShape({2, 4}));
  test::FillValues<int32_t>(&expected, {0, 1, 1, 2, 2, 2, 3, 3});
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

TEST_F(BucketizeOpTest, Int32) {
  MakeOp(DT_INT32, {-1.5f, 2.0f, 7.25f});
  AddInputFromArray<int32_t>(TensorShape({6}), {-3, -1, 2, 3, 7, 8});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_INT32, TensorShape({6}));
  test::FillValues<int32_t>(&expected, {0, 1, 2, 2, 2, 3});
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

TEST_F(BucketizeOpTest, EmptyBoundaries) {
  MakeOp(DT_DOUBLE, {});
  AddInputFromArray<double>(TensorShape({3}), {-1.0, 0.0, 1.0});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_INT32, TensorShape({3}));
  test::FillValues<int32_t>(&expected, {0, 0, 0});
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

TEST_F(BucketizeOpTest, EmptyInput) {
  MakeOp(DT_FLOAT, {1.0f});
  AddInputFromArray<float>(TensorShape({0, 3}), {});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), TensorShape({0, 3}));
}

TEST_F(BucketizeOpTest, Int64AndParallelThreshold) {
  // Exactly at the parallelization threshold (4096) plus a few over.
  std::vector<float> boundaries = {0.0f, 10.0f, 100.0f};
  for (const int n : {4096, 4097, 8192}) {
    inputs_.clear();
    MakeOp(DT_INT64, boundaries);
    std::vector<int64_t> values(n);
    for (int i = 0; i < n; ++i) values[i] = static_cast<int64_t>((i * 17) % 150);
    AddInputFromArray<int64_t>(TensorShape({n}), values);
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected(DT_INT32, TensorShape({n}));
    auto e = expected.flat<int32_t>();
    for (int i = 0; i < n; ++i) {
      e(i) = std::upper_bound(boundaries.begin(), boundaries.end(),
                              static_cast<float>(values[i])) -
             boundaries.begin();
    }
    test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
  }
}

TEST_F(BucketizeOpTest, NanInputKeepsHistoricalPlacement) {
  // IEEE upper_bound: `nan < boundary` is always false, so no boundary is
  // "greater" than NaN and the insertion point is past the last boundary.
  MakeOp(DT_FLOAT, {0.0f, 1.0f, 2.0f});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  AddInputFromArray<float>(TensorShape({3}), {nan, 0.5f, 3.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_INT32, TensorShape({3}));
  test::FillValues<int32_t>(&expected, {3, 1, 3});
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

// Large enough to be split into many shards on a multi-threaded device.
TEST_F(BucketizeOpTest, LargeInputMatchesReference) {
  std::vector<float> boundaries;
  for (int i = 0; i < 257; ++i) boundaries.push_back(-100.0f + 0.75f * i);
  MakeOp(DT_FLOAT, boundaries);

  const int n = 1 << 20;
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-150.0f, 150.0f);
  std::vector<float> values(n);
  for (auto& v : values) v = dist(gen);
  // Exact boundary hits.
  for (int i = 0; i < 257; ++i) values[i * 13] = boundaries[i];
  AddInputFromArray<float>(TensorShape({n}), values);
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_INT32, TensorShape({n}));
  auto e = expected.flat<int32_t>();
  for (int i = 0; i < n; ++i) {
    e(i) = std::upper_bound(boundaries.begin(), boundaries.end(), values[i]) -
           boundaries.begin();
  }
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

static Graph* BucketizeGraph(int64_t n, int num_boundaries) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, TensorShape({n}));
  input.flat<float>().setRandom();
  std::vector<float> boundaries(num_boundaries);
  for (int i = 0; i < num_boundaries; ++i) {
    boundaries[i] = static_cast<float>(i) / num_boundaries;
  }
  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("bucketize"), "Bucketize")
                  .Input(test::graph::Constant(g, input))
                  .Attr("boundaries", boundaries)
                  .Finalize(g, &node));
  return g;
}

static void BM_Bucketize(::testing::benchmark::State& state) {
  const int64_t n = state.range(0);
  const int num_boundaries = state.range(1);
  test::Benchmark("cpu", BucketizeGraph(n, num_boundaries),
                  /*old_benchmark_api=*/false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}

BENCHMARK(BM_Bucketize)
    ->UseRealTime()
    ->ArgPair(1024, 100)
    ->ArgPair(8 * 1024, 100)
    ->ArgPair(64 * 1024, 100)
    ->ArgPair(1024 * 1024, 10)
    ->ArgPair(1024 * 1024, 100)
    ->ArgPair(1024 * 1024, 1000)
    ->ArgPair(16 * 1024 * 1024, 100);

}  // namespace
}  // namespace tensorflow
