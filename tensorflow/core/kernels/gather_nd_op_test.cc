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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
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
  void MakeOp(DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherNd")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GatherNdOpTest, Simple) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 8, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected, {8, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
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

#define BM_GATHER_ND(DEVICE, INDEX)                                 \
  static void BM_##DEVICE##_gather_nd_##INDEX(int iters, int dim) { \
    const int64 tot = static_cast<int64>(iters) * kLookups * 4;     \
    testing::ItemsProcessed(tot);                                   \
    testing::BytesProcessed(tot * sizeof(float));                   \
    testing::UseRealTime();                                         \
    test::Benchmark(#DEVICE, GatherNd<INDEX>(dim)).Run(iters);      \
  }                                                                 \
  BENCHMARK(BM_##DEVICE##_gather_nd_##INDEX)                        \
      ->Arg(10)                                                     \
      ->Arg(100)                                                    \
      ->Arg(1000)                                                   \
      ->Arg(10000)

BM_GATHER_ND(cpu, int32);
BM_GATHER_ND(gpu, int32);
BM_GATHER_ND(cpu, int64);
BM_GATHER_ND(gpu, int64);

}  // namespace
}  // namespace tensorflow
