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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class ScatterUpdateOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterUpdate")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};
class ScatterSubOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterSub")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterUpdateOpTest, Simple_StringType) {
  MakeOp(DT_STRING_REF, DT_INT32);
  AddInputFromArray<tstring>(TensorShape({1}), {"Brain"});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<tstring>(TensorShape({1}), {"TensorFlow"});
  TF_ASSERT_OK(RunOpKernel());
  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_STRING, TensorShape({1}));
  test::FillValues<tstring>(&expected, {"TensorFlow"});
  test::ExpectTensorEqual<tstring>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_BoolType) {
  MakeOp(DT_BOOL_REF, DT_INT32);
  AddInputFromArray<bool>(TensorShape({1}), {false});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<bool>(TensorShape({1}), {true});
  TF_ASSERT_OK(RunOpKernel());
  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_BOOL, TensorShape({1}));
  test::FillValues<bool>(&expected, {true});
  test::ExpectTensorEqual<bool>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_TwoD32) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_Two64) {
  MakeOp(DT_FLOAT_REF, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_ZeroD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {101});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {0, 0, 0, 101, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_OneD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3}), {100, 101, 102});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {100, 0, 102, 0, 101});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, HigherRank) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({8}), {0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({2, 3}), {0, 4, 2, 1, 3, 6});
  AddInputFromArray<float>(TensorShape({2, 3}), {10, 20, 30, 40, 50, 60});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {10, 40, 30, 50, 20, 0, 60, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "indices[2] = 99 is not in [0, 5)"))
      << s;
}

TEST_F(ScatterSubOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT_REF, DT_INT32);
  // Feed and run
  AddInputFromArray<float>(TensorShape({14}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 99});
  AddInputFromArray<float>(TensorShape({3}), {100, 101, 102});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "indices[2] = 99 is not in [0, 14)"))
      << s;
}

TEST_F(ScatterSubOpTest, StressIndexTest) {
  MakeOp(DT_INT32_REF, DT_INT32);
  // Feed and run
  const int kRows = 1;
  std::vector<int32> values(kRows, 0);
  const int kNumUpdates = 1000000;
  std::vector<int32> indices(kNumUpdates, 0);
  std::vector<int32> updates(kNumUpdates, 1);
  AddInputFromArray<int32>(TensorShape({kRows}), values);
  AddInputFromArray<int32>(TensorShape({kNumUpdates}), indices);
  AddInputFromArray<int32>(TensorShape({kNumUpdates}), updates);
  Status s = RunOpKernel();
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int32>(&expected, {-1000000});
  test::ExpectTensorEqual<int32>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Error_WrongDimsIndices) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1, 3}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Must have updates.shape = indices.shape + "
                                "params.shape[1:] or updates.shape = [], got "))
      << s;
}

TEST_F(ScatterUpdateOpTest, Error_MismatchedParamsAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(
      TensorShape({3, 4}),
      {100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Must have updates.shape = indices.shape + "
                                "params.shape[1:] or updates.shape = [], got "))

      << s;
}

TEST_F(ScatterUpdateOpTest, Error_MismatchedIndicesAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {100, 101, 102, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Must have updates.shape = indices.shape + "
                                "params.shape[1:] or updates.shape = [], got "))
      << s;
}

class ScatterUpdateBM : public ScatterUpdateOpTest {
 public:
  void TestBody() override {}
  void MakeBenchmarkOp(const char* op, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", op)
                     .Input(FakeInput(DT_FLOAT_REF))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_CHECK_OK(InitOp());
  }
};

template <typename Index>
void BM_ScatterHelper(::testing::benchmark::State& state, int embedding_size,
                      const char* op, bool big_num_updates = false) {
  const int kRows = 10000000 / embedding_size;
  std::vector<float> values;
  values.reserve(kRows);
  for (int i = 0; i < kRows * embedding_size; i++) {
    values.push_back(i);
  }
  const int kNumUpdates = big_num_updates ? 1000000 : 1000;
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<Index> indices;
  std::vector<float> updates;
  for (int i = 0; i < kNumUpdates; i++) {
    indices.push_back(rnd.Uniform(kRows));
    for (int j = 0; j < embedding_size; j++) {
      updates.push_back(i * 10 + j);
    }
  }

  ScatterUpdateBM bm;
  bm.MakeBenchmarkOp(op, DataTypeToEnum<Index>::v());
  bm.AddInputFromArray<float>(TensorShape({kRows, embedding_size}), values);
  bm.AddInputFromArray<Index>(TensorShape({kNumUpdates}), indices);
  bm.AddInputFromArray<float>(TensorShape({kNumUpdates, embedding_size}),
                              updates);
  for (auto i : state) {
    Status s = bm.RunOpKernel();
  }
  state.SetItemsProcessed((static_cast<int64>(kNumUpdates) * embedding_size) *
                          state.iterations());
}

void BM_ScatterUpdateInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterUpdate");
}
void BM_ScatterUpdateInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64>(state, embedding_size, "ScatterUpdate");
}

void BM_ScatterAddInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterAdd");
}

void BM_ScatterAddInt32Large(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterAdd", true);
}
void BM_ScatterAddInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64>(state, embedding_size, "ScatterAdd");
}

void BM_ScatterMulInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterMul");
}
void BM_ScatterMulInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64>(state, embedding_size, "ScatterMul");
}

void BM_ScatterDivInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterDiv");
}
void BM_ScatterDivInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64>(state, embedding_size, "ScatterDiv");
}

void BM_ScatterMinInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterMin");
}
void BM_ScatterMinInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64>(state, embedding_size, "ScatterMin");
}

void BM_ScatterMaxInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterMax");
}
void BM_ScatterMaxInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64>(state, embedding_size, "ScatterMax");
}

BENCHMARK(BM_ScatterUpdateInt32)
    ->Arg(1)
    ->Arg(10)
    ->Arg(32)
    ->Arg(50)
    ->Arg(64)
    ->Arg(80)
    ->Arg(96)
    ->Arg(112)
    ->Arg(192)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000);
BENCHMARK(BM_ScatterUpdateInt64)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(100000);

BENCHMARK(BM_ScatterAddInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterAddInt32Large)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);

BENCHMARK(BM_ScatterAddInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterMulInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterMulInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterDivInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterDivInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterMinInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterMinInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterMaxInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterMaxInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

}  // namespace
}  // namespace tensorflow
