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

#include <functional>
#include <memory>
#include <vector>

#include "absl/strings/match.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace test {
namespace graph {

class Node* GatherNd(Graph* g, class Node* in0, class Node* in1) {
  class Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "GatherNd")
                  .Input(in0)
                  .Input(in1)
                  .Finalize(g, &ret));
  return ret;
}

}  // namespace graph
}  // namespace test

namespace {

class GatherNdOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType param_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherNd")
                     .Input(FakeInput(param_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GatherNdOpTest, Simple) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected, {8, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherNdOpTest, Error_OutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 5});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.message(), "indices[1] = [5] does not index into param shape [5]"))
      << s.message();
}

TEST_F(GatherNdOpTest, Quantized_UINT8) {
  MakeOp(DT_QUINT8, DT_INT32);

  // Feed and run
  AddInputFromArray<quint8>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_QUINT8, TensorShape({2}));
  test::FillValues<quint8>(&expected, {8, 4});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

TEST_F(GatherNdOpTest, Quantized_INT8) {
  MakeOp(DT_QINT8, DT_INT32);

  AddInputFromArray<qint8>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_QINT8, TensorShape({2}));
  test::FillValues<qint8>(&expected, {8, 4});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

class GatherNdOpIgnoreBadIndicesTest : public OpsTestBase {
 protected:
  void MakeOp(DataType param_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherNd")
                     .Input(FakeInput(param_type))
                     .Input(FakeInput(index_type))
                     .Attr("bad_indices_policy", "IGNORE")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GatherNdOpIgnoreBadIndicesTest, IgnoreOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {9, 1, 2, 8, 4});
  // Put the bad index in the middle to make sure others are still correctly
  // gathered.
  AddInputFromArray<int32>(TensorShape({3, 1}), {3, 5, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {8, 0, 1});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

class GatherNdOpConstructionTest : public OpsTestBase {};

TEST_F(GatherNdOpConstructionTest, Error_BadIndicesPolicyInvalid) {
  TF_ASSERT_OK(NodeDefBuilder("myop", "GatherNd")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("bad_indices_policy", "AN_UNRECOGNIZED_POLICY")
                   .Finalize(node_def()));
  EXPECT_NE(InitOp(), absl::OkStatus());
}

constexpr int kLookups = 2000;

template <typename Index>
static Graph* GatherNd(int dim) {
  Graph* g = new Graph(OpRegistry::Global());
  // Always use a 512MB buffer.
  // const int kRows = ((512 << 20) / sizeof(float)) / dim;
  Tensor params(DT_FLOAT, TensorShape({dim, 8, 16, 32}));
  params.flat<float>().setRandom();

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  Tensor indices(DataTypeToEnum<Index>::value, TensorShape({kLookups, 4}));
  auto indices_mat = indices.matrix<Index>();
  for (int i = 0; i < kLookups; i++) {
    indices_mat(i, 0) = rnd.Uniform(dim);
    indices_mat(i, 1) = rnd.Uniform(8);
    indices_mat(i, 2) = rnd.Uniform(16);
    indices_mat(i, 3) = rnd.Uniform(32);
  }

  test::graph::GatherNd(g, test::graph::Constant(g, params),
                        test::graph::Constant(g, indices));
  return g;
}

#define BM_GATHER_ND(DEVICE, INDEX)                              \
  static void BM_##DEVICE##_gather_nd_##INDEX(                   \
      ::testing::benchmark::State& state) {                      \
    const int dim = state.range(0);                              \
    test::Benchmark(#DEVICE, GatherNd<INDEX>(dim),               \
                    /*old_benchmark_api=*/false)                 \
        .Run(state);                                             \
    const int64_t tot =                                          \
        static_cast<int64_t>(state.iterations()) * kLookups * 4; \
    state.SetItemsProcessed(tot);                                \
    state.SetBytesProcessed(tot * sizeof(float));                \
  }                                                              \
  BENCHMARK(BM_##DEVICE##_gather_nd_##INDEX)                     \
      ->UseRealTime()                                            \
      ->Arg(10)                                                  \
      ->Arg(100)                                                 \
      ->Arg(1000)                                                \
      ->Arg(10000)

BM_GATHER_ND(cpu, int32);
BM_GATHER_ND(gpu, int32);
BM_GATHER_ND(cpu, int64_t);
BM_GATHER_ND(gpu, int64_t);

}  // namespace
}  // namespace tensorflow
