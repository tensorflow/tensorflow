/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class IsotonicRegressionOpTest : public OpsTestBase {
 public:
  void MakeOp(DataType type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "IsotonicRegression")
                     .Input(FakeInput(type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

class BenchmarkHelper : public IsotonicRegressionOpTest {
 public:
  void TestBody() override {}

  void AddIncreasingInput(int batch_size, int input_size) {
    std::vector<float> input_data(input_size * batch_size, 0);
    for (int i = 0; i < input_data.size(); i++) {
      input_data[i] = i;
    }
    AddInputFromArray<float>(TensorShape({batch_size, input_size}), input_data);
  }
};

TEST_F(IsotonicRegressionOpTest, Constant) {
  MakeOp(DT_FLOAT_REF);

  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected,
                          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  test::ExpectClose(expected, *GetOutput((0)));
}

TEST_F(IsotonicRegressionOpTest, IncreasingInput) {
  MakeOp(DT_FLOAT_REF);

  AddInputFromArray<float>(TensorShape({5, 3}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected,
                          {2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11, 14, 14, 14});
  test::ExpectClose(expected, *GetOutput((0)));

  Tensor expected_ord(allocator(), DT_INT32, TensorShape({5, 3}));
  test::FillValues<int>(&expected_ord,
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  test::ExpectTensorEqual<int>(expected_ord, *GetOutput((1)));
}

TEST_F(IsotonicRegressionOpTest, Decreasing) {
  MakeOp(DT_FLOAT_REF);

  AddInputFromArray<float>(TensorShape({5, 3}),
                           {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected,
                          {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  test::ExpectClose(expected, *GetOutput((0)));

  Tensor expected_ord(allocator(), DT_INT32, TensorShape({5, 3}));
  test::FillValues<int>(&expected_ord,
                        {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2});
  test::ExpectTensorEqual<int>(expected_ord, *GetOutput((1)));
}

#ifdef PLATFORM_GOOGLE

static void BM_IncreasingSequence(benchmark::State& state) {
  int batch_size = state.range(0);
  int input_size = state.range(1);

  for (auto _ : state) {
    state.PauseTiming();
    BenchmarkHelper helper;
    helper.MakeOp(DT_FLOAT_REF);
    helper.AddIncreasingInput(batch_size, input_size);
    state.ResumeTiming();
    absl::Status stat = helper.RunOpKernel();
  }
  state.SetItemsProcessed(
      static_cast<int64_t>(batch_size * input_size * state.iterations()));
}

BENCHMARK(BM_IncreasingSequence)
    ->Args({1, 1 << 0})
    ->Args({1, 1 << 5})
    ->Args({1, 1 << 8})
    ->Args({1, 1 << 10})
    ->Args({1, 1 << 20})
    ->Args({1, 2 << 20})
    ->Args({1 << 0, 1 << 10})
    ->Args({1 << 1, 1 << 10})
    ->Args({1 << 4, 1 << 10})
    ->Args({1 << 6, 1 << 10})
    ->Args({1 << 9, 1 << 10})
    ->Args({1 << 10, 1 << 10});

#endif  // PLATFORM_GOOGLE

}  // namespace
}  // namespace tensorflow
