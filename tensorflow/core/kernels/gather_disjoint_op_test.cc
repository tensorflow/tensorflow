/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
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
namespace {

class GatherOpDisjointTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type, int n, DataType index_type,
              bool reverse_order) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherDisjoint")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(n, index_type))
                     .Attr("reverse_order", reverse_order)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GatherOpDisjointTest, ScalarIndices) {
  MakeOp(DT_FLOAT, 7, DT_INT32, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({7}), {0, 1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {4});
  AddInputFromArray<int32>(TensorShape({}), {4});
  AddInputFromArray<int32>(TensorShape({}), {5});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  const int expected_outputs[] = {3, 1, 0, 0, 4, 0, 5};
  for (int i = 0; i < 7; ++i) {
    Tensor expected(allocator(), DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected, {expected_outputs[i]});
    test::ExpectTensorEqual<float>(expected, *GetOutput(i));
  }
}

TEST_F(GatherOpDisjointTest, ScalarIndicesReverse) {
  MakeOp(DT_FLOAT, 7, DT_INT32, true);

  // Feed and run
  AddInputFromArray<float>(TensorShape({7}), {0, 1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {6});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {6});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  const int expected_outputs[] = {0, 0, 3, 2, 0, 1, 6};
  for (int i = 0; i < 7; ++i) {
    Tensor expected(allocator(), DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected, {expected_outputs[i]});
    test::ExpectTensorEqual<float>(expected, *GetOutput(i));
  }
}

TEST_F(GatherOpDisjointTest, ScalarIndices_Complex) {
  MakeOp(DT_COMPLEX64, 5, DT_INT32, false);

  // Feed and run
  AddInputFromArray<std::complex<float>>(
      TensorShape({5}), {std::complex<float>(0, 10), std::complex<float>(1, 11),
                         std::complex<float>(2, 12), std::complex<float>(3, 13),
                         std::complex<float>(4, 14)});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  const std::complex<float> expected_outputs[] = {
      std::complex<float>(3, 13), std::complex<float>(1, 11),
      std::complex<float>(0, 0), std::complex<float>(0, 0),
      std::complex<float>(4, 14)};
  for (int i = 0; i < 5; ++i) {
    Tensor expected(allocator(), DT_COMPLEX64, TensorShape({}));
    test::FillValues<std::complex<float>>(&expected, {expected_outputs[i]});
    test::ExpectTensorEqual<std::complex<float>>(expected, *GetOutput(i));
  }
}

TEST_F(GatherOpDisjointTest, Simple_TwoD32) {
  MakeOp(DT_DOUBLE, 2, DT_INT32, false);

  // Feed and run
  AddInputFromArray<double>(TensorShape({5, 3}),
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {1, 4, 1, 2});
  AddInputFromArray<int32>(TensorShape({4}), {2, 0, 3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_DOUBLE, TensorShape({4, 3}));
  test::FillValues<double>(&expected, {3, 4, 5, 12, 13, 14, 0, 0, 0, 6, 7, 8});
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
  test::FillValues<double>(&expected, {0, 0, 0, 0, 1, 2, 9, 10, 11, 0, 0, 0});
  test::ExpectTensorEqual<double>(expected, *GetOutput(1));
}

TEST_F(GatherOpDisjointTest, Simple_TwoD32_Reverse) {
  MakeOp(DT_DOUBLE, 2, DT_INT32, true);

  // Feed and run
  AddInputFromArray<double>(TensorShape({5, 3}),
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {1, 4, 1, 2});
  AddInputFromArray<int32>(TensorShape({4}), {2, 0, 3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_DOUBLE, TensorShape({4, 3}));
  test::FillValues<double>(&expected, {0, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0});
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
  test::FillValues<double>(&expected,
                           {6, 7, 8, 0, 1, 2, 9, 10, 11, 12, 13, 14});
  test::ExpectTensorEqual<double>(expected, *GetOutput(1));
}

TEST_F(GatherOpDisjointTest, ZeroSize_TwoD32) {
  MakeOp(DT_FLOAT, 1, DT_INT32, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 0}), {});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 0}));
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpDisjointTest, Simple_TwoD64) {
  MakeOp(DT_FLOAT, 2, DT_INT64, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int64>(TensorShape({4}), {1, 4, 1, 2});
  AddInputFromArray<int64>(TensorShape({4}), {2, 0, 3, 4});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {3, 4, 5, 12, 13, 14, 0, 0, 0, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  test::FillValues<float>(&expected, {0, 0, 0, 0, 1, 2, 9, 10, 11, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(1));
}

TEST_F(GatherOpDisjointTest, HighRank) {
  MakeOp(DT_FLOAT, 1, DT_INT32, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 0, 2, 3, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {1, 2, 0, 0, 3, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpDisjointTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT, 1, DT_INT32, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 99, 2});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      StringPiece(s.ToString()).contains("indices[2] = 99 is not in [0, 5)"))
      << s;
}

TEST_F(GatherOpDisjointTest, Error_ScalarParams) {
  MakeOp(DT_FLOAT, 1, DT_INT32, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({4}), {0, 0, 0, 0});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("params must be at least 1 dimensional"))
      << s;
}

constexpr int kLookups = 2000;

Node* GatherDisjointNode(Graph* g, Node* in0,
                         std::vector<NodeBuilder::NodeOut> in1,
                         bool reverse_order) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "GatherDisjoint")
                  .Input(in0)
                  .Input(in1)
                  .Attr("N", 1)
                  .Attr("reverse_order", reverse_order)
                  .Finalize(g, &ret));
  return ret;
}

template <typename Index>
static Graph* GatherDisjoint(int dim) {
  Graph* g = new Graph(OpRegistry::Global());
  // Always use a 512MB buffer.
  const int kRows = ((512 << 20) / sizeof(float)) / dim;
  Tensor params(DT_FLOAT, TensorShape({kRows, dim}));
  params.flat<float>().setRandom();

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<Index> indices_vec;
  indices_vec.reserve(kLookups);
  for (int i = 0; i < kLookups; i++) {
    indices_vec.push_back(rnd.Uniform(kRows));
  }
  Tensor indices(DataTypeToEnum<Index>::value, TensorShape({kLookups}));
  for (int i = 0; i < indices_vec.size(); i++) {
    indices.flat<Index>()(i) = indices_vec[i];
  }

  std::vector<NodeBuilder::NodeOut> inputs;
  // indices must be in host memory
  inputs.push_back(test::graph::HostConstant(g, indices));

  GatherDisjointNode(g, test::graph::Constant(g, params), inputs, false);
  return g;
}

#define BM_GATHER_DISJOINT(DEVICE, INDEX)                             \
  static void BM_##DEVICE##_gather_disj_##INDEX(int iters, int dim) { \
    const int64 tot = static_cast<int64>(iters) * kLookups * dim;     \
    testing::ItemsProcessed(tot);                                     \
    testing::BytesProcessed(tot * sizeof(float));                     \
    testing::StartTiming();                                           \
    test::Benchmark(#DEVICE, GatherDisjoint<INDEX>(dim)).Run(iters);  \
    testing::UseRealTime();                                           \
  }                                                                   \
  BENCHMARK(BM_##DEVICE##_gather_disj_##INDEX)                        \
      ->Arg(1)                                                        \
      ->Arg(10)                                                       \
      ->Arg(20)                                                       \
      ->Arg(64)                                                       \
      ->Arg(100)                                                      \
      ->Arg(200)                                                      \
      ->Arg(1000)

BM_GATHER_DISJOINT(cpu, int32);
BM_GATHER_DISJOINT(cpu, int64);
#if GOOGLE_CUDA
BM_GATHER_DISJOINT(gpu, int32);
BM_GATHER_DISJOINT(gpu, int64);
#endif

}  // namespace
}  // namespace tensorflow
