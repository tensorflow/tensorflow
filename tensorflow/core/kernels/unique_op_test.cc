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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/log/check.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

const int kMaxStrLen = 40;

class UniqueOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType dtype, bool with_counts = false) {
    TF_ASSERT_OK(
        NodeDefBuilder("unique", with_counts ? "UniqueWithCounts" : "Unique")
            .Input(FakeInput(dtype))
            .Attr("out_idx", DT_INT32)
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(UniqueOpTest, Int64WithCounts) {
  MakeOp(DT_INT64, /*with_counts=*/true);
  AddInputFromArray<int64_t>(TensorShape({8}), {7, 3, 7, 7, -1, 3, 9, -1});
  TF_ASSERT_OK(RunOpKernel());
  Tensor y(DT_INT64, TensorShape({4}));
  test::FillValues<int64_t>(&y, {7, 3, -1, 9});
  test::ExpectTensorEqual<int64_t>(y, *GetOutput(0));
  Tensor idx(DT_INT32, TensorShape({8}));
  test::FillValues<int32_t>(&idx, {0, 1, 0, 0, 2, 1, 3, 2});
  test::ExpectTensorEqual<int32_t>(idx, *GetOutput(1));
  Tensor count(DT_INT32, TensorShape({4}));
  test::FillValues<int32_t>(&count, {3, 2, 2, 1});
  test::ExpectTensorEqual<int32_t>(count, *GetOutput(2));
}

// Every NaN is its own unique element (NaN != NaN), while -0.0 and 0.0 compare
// equal and collapse to the first one seen.
template <typename T>
void CheckFloatSpecialValues(UniqueOpTest* t) {
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();
  t->AddInputFromArray<T>(TensorShape({9}),
                          {T(1.5), nan, T(-0.0), T(1.5), nan, T(0.0), inf,
                           -inf, inf});
  TF_ASSERT_OK(t->RunOpKernel());
  const Tensor& y = *t->GetOutput(0);
  ASSERT_EQ(y.NumElements(), 6);
  auto yv = y.flat<T>();
  EXPECT_EQ(yv(0), T(1.5));
  EXPECT_TRUE(std::isnan(yv(1)));
  EXPECT_EQ(yv(2), T(0.0));
  EXPECT_TRUE(std::signbit(yv(2)));  // First-seen zero is -0.0.
  EXPECT_TRUE(std::isnan(yv(3)));
  EXPECT_EQ(yv(4), inf);
  EXPECT_EQ(yv(5), -inf);
  Tensor idx(DT_INT32, TensorShape({9}));
  test::FillValues<int32_t>(&idx, {0, 1, 2, 0, 3, 2, 4, 5, 4});
  test::ExpectTensorEqual<int32_t>(idx, *t->GetOutput(1));
  Tensor count(DT_INT32, TensorShape({6}));
  test::FillValues<int32_t>(&count, {2, 1, 2, 1, 2, 1});
  test::ExpectTensorEqual<int32_t>(count, *t->GetOutput(2));
}

TEST_F(UniqueOpTest, FloatSpecialValues) {
  MakeOp(DT_FLOAT, /*with_counts=*/true);
  CheckFloatSpecialValues<float>(this);
}

TEST_F(UniqueOpTest, DoubleSpecialValues) {
  MakeOp(DT_DOUBLE, /*with_counts=*/true);
  CheckFloatSpecialValues<double>(this);
}

TEST_F(UniqueOpTest, LargeFloatMatchesReference) {
  MakeOp(DT_FLOAT);
  const int n = 200000;
  std::mt19937 gen(7);
  std::uniform_int_distribution<int> dist(0, 5000);
  std::vector<float> values(n);
  for (auto& v : values) v = static_cast<float>(dist(gen)) * 0.25f;
  for (int i = 0; i < n; i += 1009) {
    values[i] = std::numeric_limits<float>::quiet_NaN();
  }
  AddInputFromArray<float>(TensorShape({n}), values);
  TF_ASSERT_OK(RunOpKernel());

  // Reference: first-occurrence order, NaNs always new.
  std::vector<float> ref_y;
  std::vector<int32_t> ref_idx(n);
  std::unordered_map<float, int32_t> seen;
  for (int i = 0; i < n; ++i) {
    if (std::isnan(values[i])) {
      ref_idx[i] = static_cast<int32_t>(ref_y.size());
      ref_y.push_back(values[i]);
      continue;
    }
    auto it = seen.emplace(values[i], static_cast<int32_t>(ref_y.size()));
    if (it.second) ref_y.push_back(values[i]);
    ref_idx[i] = it.first->second;
  }
  const Tensor& y = *GetOutput(0);
  ASSERT_EQ(y.NumElements(), ref_y.size());
  for (size_t i = 0; i < ref_y.size(); ++i) {
    if (std::isnan(ref_y[i])) {
      EXPECT_TRUE(std::isnan(y.flat<float>()(i))) << i;
    } else {
      EXPECT_EQ(y.flat<float>()(i), ref_y[i]) << i;
    }
  }
  Tensor idx(DT_INT32, TensorShape({n}));
  test::FillValues<int32_t>(&idx, ref_idx);
  test::ExpectTensorEqual<int32_t>(idx, *GetOutput(1));
}

TensorProto GetRandomInt32TensorProto(int dim, int max_int) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int int_val = std::rand() % max_int;
    tensor_proto.add_int_val(int_val);
  }
  return tensor_proto;
}

TensorProto GetRandomInt32TensorProtoWithRepeat(int dim, int repeat,
                                                int max_int) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int int_val = std::rand() % max_int;
    for (int j = 0; j < repeat; ++j) {
      tensor_proto.add_int_val(int_val);
    }
  }
  return tensor_proto;
}

void BM_Unique_INT32(::testing::benchmark::State& state) {
  const int dim = state.range(0);
  const int max_int = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({dim}));
  CHECK(input.FromProto(GetRandomInt32TensorProto(dim, max_int)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim *
                          sizeof(int32_t));
}

void BM_Unique_INT32_Repeat(::testing::benchmark::State& state) {
  const int dim = state.range(0);
  const int max_int = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({dim * 200}));
  CHECK(
      input.FromProto(GetRandomInt32TensorProtoWithRepeat(dim, 200, max_int)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim * 200 *
                          sizeof(int32_t));
}

void BM_Unique_FLOAT(::testing::benchmark::State& state) {
  const int dim = state.range(0);
  const int max_int = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_FLOAT, TensorShape({dim}));
  std::mt19937 gen(1);
  std::uniform_int_distribution<int> dist(0, max_int - 1);
  auto v = input.flat<float>();
  for (int i = 0; i < dim; ++i) v(i) = static_cast<float>(dist(gen)) * 0.5f;

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim *
                          sizeof(float));
}

TensorProto GetRandomStringsTensorProto(int dim, int max_str_len) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int len = std::rand() % max_str_len + 1;
    std::string rand_str;
    rand_str.resize(len);
    for (int j = 0; j < len; ++j) {
      rand_str[j] = static_cast<char>(j % 256);
    }
    tensor_proto.add_string_val(rand_str);
  }
  return tensor_proto;
}

void BM_Unique_STRING(::testing::benchmark::State& state) {
  const int dim = state.range(0);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_STRING, TensorShape({dim}));
  CHECK(input.FromProto(GetRandomStringsTensorProto(dim, kMaxStrLen)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_STRING)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim *
                          sizeof(tstring));
}

BENCHMARK(BM_Unique_INT32)
    ->UseRealTime()
    ->ArgPair(32, 1024 * 1024)
    ->ArgPair(256, 1024 * 1024)
    ->ArgPair(1024, 1024 * 1024)
    ->ArgPair(4 * 1024, 1024 * 1024)
    ->ArgPair(16 * 1024, 1024 * 1024)
    ->ArgPair(64 * 1024, 1024 * 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 1024 * 1024)
    ->ArgPair(32, 64 * 1024 * 1024)
    ->ArgPair(256, 64 * 1024 * 1024)
    ->ArgPair(1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024, 64 * 1024 * 1024)
    ->ArgPair(16 * 1024, 64 * 1024 * 1024)
    ->ArgPair(64 * 1024, 64 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 64 * 1024 * 1024);

BENCHMARK(BM_Unique_INT32_Repeat)
    ->UseRealTime()
    ->ArgPair(32, 1024 * 1024)
    ->ArgPair(256, 1024 * 1024)
    ->ArgPair(1024, 1024 * 1024)
    ->ArgPair(4 * 1024, 1024 * 1024)
    ->ArgPair(16 * 1024, 1024 * 1024)
    ->ArgPair(64 * 1024, 1024 * 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 1024 * 1024)
    ->ArgPair(32, 32 * 1024 * 1024)
    ->ArgPair(256, 32 * 1024 * 1024)
    ->ArgPair(1024, 32 * 1024 * 1024)
    ->ArgPair(4 * 1024, 32 * 1024 * 1024)
    ->ArgPair(16 * 1024, 32 * 1024 * 1024)
    ->ArgPair(64 * 1024, 32 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 32 * 1024 * 1024)
    ->ArgPair(32, 64 * 1024 * 1024)
    ->ArgPair(256, 64 * 1024 * 1024)
    ->ArgPair(1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024, 64 * 1024 * 1024)
    ->ArgPair(16 * 1024, 64 * 1024 * 1024)
    ->ArgPair(64 * 1024, 64 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 64 * 1024 * 1024);

BENCHMARK(BM_Unique_FLOAT)
    ->UseRealTime()
    ->ArgPair(4 * 1024, 1024)
    ->ArgPair(1024 * 1024, 1024)
    ->ArgPair(1024 * 1024, 64 * 1024)
    ->ArgPair(1024 * 1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 4 * 1024 * 1024);

BENCHMARK(BM_Unique_STRING)
    ->UseRealTime()
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4 * 1024)
    ->Arg(16 * 1024)
    ->Arg(64 * 1024)
    ->Arg(256 * 1024);

}  // namespace
}  // namespace tensorflow
