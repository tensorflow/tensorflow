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

#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"  // NOLINT(misc-include-cleaner)

namespace tensorflow {

static Graph* PTruncatedNormal(int num_batches, int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32_t>().setValues({num_batches, samples_per_batch});

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
  TF_CHECK_OK(  // NOLINT(misc-include-cleaner)
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
  shape_t.flat<int32_t>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(-2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(2.0);

  Node* ret;
  TF_CHECK_OK(  // NOLINT(misc-include-cleaner)
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
  shape_t.flat<int32_t>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(std::numeric_limits<float>::infinity());

  Node* ret;
  TF_CHECK_OK(  // NOLINT(misc-include-cleaner)
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

// NOLINTBEGIN(misc-include-cleaner)
#define BM_PTruncatedNormalDev(DEVICE, B, S)                                   \
  static void BM_PTruncatedNormal_##DEVICE##_##B##_##S(                        \
      ::testing::benchmark::State& state) { /* NOLINT(misc-include-cleaner) */ \
    test::Benchmark(#DEVICE, PTruncatedNormal(B, S),                           \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_##DEVICE##_##B##_##S);  // NOLINT

#define BM_PTruncatedNormalDev_2SD(DEVICE, B, S)                               \
  static void BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S(                    \
      ::testing::benchmark::State& state) { /* NOLINT(misc-include-cleaner) */ \
    test::Benchmark(#DEVICE, PTruncatedNormal2SD(B, S),                        \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S);  // NOLINT

#define BM_PTruncatedNormalDev_OneTail(DEVICE, B, S)                           \
  static void BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S(                \
      ::testing::benchmark::State& state) { /* NOLINT(misc-include-cleaner) */ \
    test::Benchmark(#DEVICE, PTruncatedNormalOneTail(B, S),                    \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S);  // NOLINT

BM_PTruncatedNormalDev(cpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(cpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(cpu, 10000, 100);
BM_PTruncatedNormalDev(gpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(gpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(gpu, 10000, 100);
// NOLINTEND(misc-include-cleaner)

class ParameterizedTruncatedNormalOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ParameterizedTruncatedNormal")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ParameterizedTruncatedNormalOpTest, IntegerOverflow) {
  MakeOp();
  // Shape: [2, 1073741824] -> total elements 2147483648, which overflows
  // int32_t.
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 1073741824});
  AddInputFromArray<float>(TensorShape({2}), {0.0f, 0.0f});
  AddInputFromArray<float>(TensorShape({2}), {1.0f, 1.0f});
  AddInputFromArray<float>(TensorShape({2}), {2.0f, 2.0f});
  AddInputFromArray<float>(TensorShape({2}), {3.0f, 3.0f});

  absl::Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "ParameterizedTruncatedNormal does not support output shapes "
            "with more than 2**31 - 1 elements.");
}

TEST_F(ParameterizedTruncatedNormalOpTest, ParallelismAdjustment) {
  MakeOp();
  // Shape: 101 (which is kDesiredBatchSize + 1).
  // If the size is adjusted incorrectly (e.g. size - 1 mutant), the last
  // element (index 100) will not be generated/written to.
  AddInputFromArray<int32_t>(TensorShape({1}), {101});
  AddInputFromArray<float>(TensorShape({1}), {10.0f});
  AddInputFromArray<float>(TensorShape({1}), {1.0f});
  AddInputFromArray<float>(TensorShape({1}), {5.0f});
  AddInputFromArray<float>(TensorShape({1}), {15.0f});

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  auto flat_output = output->flat<float>();
  EXPECT_EQ(flat_output.size(), 101);
  for (int i = 0; i < 101; ++i) {
    float val = flat_output(i);
    EXPECT_GE(val, 5.0f);
    EXPECT_LE(val, 15.0f);
  }
}

class StatelessParameterizedTruncatedNormalOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("myop", "StatelessParameterizedTruncatedNormal")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("dtype", DT_FLOAT)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(StatelessParameterizedTruncatedNormalOpTest, IntegerOverflow) {
  MakeOp();
  // Shape: [2, 1073741824] -> total elements 2147483648, which overflows
  // int32_t.
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 1073741824});
  AddInputFromArray<int32_t>(TensorShape({2}), {123, 456});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  AddInputFromArray<float>(TensorShape({}), {2.0f});
  AddInputFromArray<float>(TensorShape({}), {3.0f});

  absl::Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "ParameterizedTruncatedNormal does not support output shapes "
            "with more than 2**31 - 1 elements.");
}

TEST_F(StatelessParameterizedTruncatedNormalOpTest, Success) {
  MakeOp();
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 3});
  AddInputFromArray<int32_t>(TensorShape({2}), {123, 456});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  AddInputFromArray<float>(TensorShape({}), {-10.0f});
  AddInputFromArray<float>(TensorShape({}), {10.0f});

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  auto flat_output = output->flat<float>();
  EXPECT_EQ(flat_output.size(), 6);
  for (int i = 0; i < 6; ++i) {
    float val = flat_output(i);
    EXPECT_GE(val, -10.0f);
    EXPECT_LE(val, 10.0f);
  }
}

}  // namespace tensorflow
