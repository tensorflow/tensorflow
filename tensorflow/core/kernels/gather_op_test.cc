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
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class GatherOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type, DataType index_type, int batch_dims = 0) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "GatherV2")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Attr("batch_dims", batch_dims)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GatherOpTest, ScalarIndices) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected, {3});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, ScalarIndices_Complex) {
  MakeOp(DT_COMPLEX64, DT_INT32);

  // Feed and run
  AddInputFromArray<std::complex<float>>(
      TensorShape({5}), {std::complex<float>(0, 10), std::complex<float>(1, 11),
                         std::complex<float>(2, 12), std::complex<float>(3, 13),
                         std::complex<float>(4, 14)});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_COMPLEX64, TensorShape({}));
  test::FillValues<std::complex<float>>(&expected,
                                        {std::complex<float>(3, 13)});
  test::ExpectTensorEqual<std::complex<float>>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD32_Axis0) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD32_Axis1) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 1, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 4}));
  test::FillValues<float>(&expected, {0, 1, 0, 2,  3, 4,  3,  5,  6,  7,
                                      6, 8, 9, 10, 9, 11, 12, 13, 12, 14});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, ZeroSize_TwoD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 0}), {});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 0}));
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int64>(TensorShape({4}), {0, 4, 0, 2});
  AddInputFromArray<int64>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, HighRank) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 0, 2, 3, 0});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {1, 2, 0, 2, 3, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 99, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "indices[2] = 99 is not in [0, 5)"))
      << s;
}

TEST_F(GatherOpTest, Error_BatchDimsOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32, 10);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 99, 2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "Expected batch_dims in the range [-1, 1], but got 10"))
      << s;
}

constexpr int kLookups = 2000;

template <typename Index>
static Graph* Gather(int dim) {
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

  Tensor axis(DataTypeToEnum<Index>::value, TensorShape({}));
  axis.scalar<Index>()() = 0;

  test::graph::Gather(g, test::graph::Constant(g, params),
                      test::graph::Constant(g, indices),
                      test::graph::HostConstant(g, axis));
  return g;
}

#define BM_GATHER(DEVICE, INDEX)                                               \
  static void BM_##DEVICE##_gather_##INDEX(                                    \
      ::testing::benchmark::State& state) {                                    \
    const int dim = state.range(0);                                            \
    test::Benchmark(#DEVICE, Gather<INDEX>(dim), /*old_benchmark_api=*/false)  \
        .Run(state);                                                           \
    const int64 tot = static_cast<int64>(state.iterations()) * kLookups * dim; \
    state.SetItemsProcessed(tot);                                              \
    state.SetBytesProcessed(tot * sizeof(float));                              \
  }                                                                            \
  BENCHMARK(BM_##DEVICE##_gather_##INDEX)                                      \
      ->UseRealTime()                                                          \
      ->Arg(1)                                                                 \
      ->Arg(10)                                                                \
      ->Arg(20)                                                                \
      ->Arg(64)                                                                \
      ->Arg(100)                                                               \
      ->Arg(200)                                                               \
      ->Arg(1000)

BM_GATHER(cpu, int32);
BM_GATHER(gpu, int32);
BM_GATHER(cpu, int64);
BM_GATHER(gpu, int64);

}  // namespace
}  // namespace tensorflow
