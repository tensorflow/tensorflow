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

#include <cmath>
#include <cstdint>
#include <limits>
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include <memory>
#endif
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include <utility>
#endif
#include <vector>

#include "absl/log/log.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/framework/device_factory.h"
#endif
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

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
  TF_CHECK_OK(
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
  TF_CHECK_OK(
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
  TF_CHECK_OK(
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

#define BM_PTruncatedNormalDev(DEVICE, B, S)                                   \
  static void BM_PTruncatedNormal_##DEVICE##_##B##_##S(                        \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormal(B, S),                           \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_2SD(DEVICE, B, S)                               \
  static void BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S(                    \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormal2SD(B, S),                        \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_OneTail(DEVICE, B, S)                           \
  static void BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S(                \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormalOneTail(B, S),                    \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S);

BM_PTruncatedNormalDev(cpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(cpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(cpu, 10000, 100);
BM_PTruncatedNormalDev(gpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(gpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(gpu, 10000, 100);

class ParameterizedTruncatedNormalOpTest : public OpsTestBase {
 protected:
  void Init(DataType dtype) {
    TF_CHECK_OK(NodeDefBuilder("op", "ParameterizedTruncatedNormal")
                    .Input(FakeInput(DT_INT32))  // shape
                    .Input(FakeInput(dtype))     // means
                    .Input(FakeInput(dtype))     // stddevs
                    .Input(FakeInput(dtype))     // minvals
                    .Input(FakeInput(dtype))     // maxvals
                    .Attr("dtype", dtype)
                    .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ParameterizedTruncatedNormalOpTest, TestNormal) {
  Init(DT_FLOAT);

  AddInputFromList<int32_t>(TensorShape({2}), {2, 50});
  AddInputFromList<float>(TensorShape({1}), {0.0f});
  AddInputFromList<float>(TensorShape({1}), {1.0f});
  AddInputFromList<float>(TensorShape({1}), {-2.0f});
  AddInputFromList<float>(TensorShape({1}), {2.0f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  ASSERT_NE(output, nullptr);
  auto flat_output = output->flat<float>();

  bool all_zeros = true;
  for (int i = 0; i < 100; ++i) {
    float val = flat_output(i);
    EXPECT_GE(val, -2.0f);
    EXPECT_LE(val, 2.0f);
    EXPECT_FALSE(std::isnan(val));
    if (val != 0.0f) {
      all_zeros = false;
    }
  }
  EXPECT_FALSE(all_zeros)
      << "Returned uninitialized/zeroed memory instead of generating samples.";
}

TEST_F(ParameterizedTruncatedNormalOpTest,
       TestBatchingAdjustmentDeterministic) {
  TF_CHECK_OK(NodeDefBuilder("op", "ParameterizedTruncatedNormal")
                  .Input(FakeInput(DT_INT32))  // shape
                  .Input(FakeInput(DT_FLOAT))  // means
                  .Input(FakeInput(DT_FLOAT))  // stddevs
                  .Input(FakeInput(DT_FLOAT))  // minvals
                  .Input(FakeInput(DT_FLOAT))  // maxvals
                  .Attr("dtype", DT_FLOAT)
                  .Attr("seed", 1)
                  .Attr("seed2", 2)
                  .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Input shape is [101]
  AddInputFromList<int32_t>(TensorShape({1}), {101});

  // Scalar inputs to trigger the batching adjustment optimization block
  AddInputFromList<float>(TensorShape({}), {0.0f});
  AddInputFromList<float>(TensorShape({}), {1.0f});
  AddInputFromList<float>(TensorShape({}), {-2.0f});
  AddInputFromList<float>(TensorShape({}), {2.0f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  ASSERT_NE(output, nullptr);
  auto flat_output = output->flat<float>();

  // Verify the size of the output tensor
  ASSERT_EQ(flat_output.size(), 101);

  // Print values to get them
  for (int i = 0; i < 101; ++i) {
    LOG(INFO) << "VAL_RUN1_" << i << ": " << flat_output(i);
  }

  // Store first run's outputs
  std::vector<float> first_run_vals;
  for (int i = 0; i < 101; ++i) {
    first_run_vals.push_back(flat_output(i));
  }

  // Re-initialize the op to reset the state of the random number generator
  TF_ASSERT_OK(InitOp());

  // Run the kernel a second time
  TF_ASSERT_OK(RunOpKernel());
  Tensor* output2 = GetOutput(0);
  ASSERT_NE(output2, nullptr);
  auto flat_output2 = output2->flat<float>();
  ASSERT_EQ(flat_output2.size(), 101);

  for (int i = 0; i < 101; ++i) {
    EXPECT_EQ(first_run_vals[i], flat_output2(i))
        << "Outputs are not deterministic at index " << i;
  }
}

TEST_F(ParameterizedTruncatedNormalOpTest, TestBatchingAdjustmentCorrectness) {
  Init(DT_FLOAT);

  // Input shape is [101] to trigger batching adjustment
  AddInputFromList<int32_t>(TensorShape({1}), {101});

  // Scalar inputs to trigger the batching adjustment optimization block
  AddInputFromList<float>(TensorShape({}), {0.0f});  // mean
  AddInputFromList<float>(TensorShape({}), {1.0f});  // stddev
  AddInputFromList<float>(TensorShape({}), {2.0f});  // minval
  AddInputFromList<float>(TensorShape({}), {3.0f});  // maxval

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  ASSERT_NE(output, nullptr);
  auto flat_output = output->flat<float>();
  ASSERT_EQ(flat_output.size(), 101);

  for (int i = 0; i < 101; ++i) {
    float val = flat_output(i);
    EXPECT_GE(val, 2.0f) << "Value at index " << i
                         << " is not within bounds [2.0, 3.0]";
    EXPECT_LE(val, 3.0f) << "Value at index " << i
                         << " is not within bounds [2.0, 3.0]";
  }
}

class StatelessParameterizedTruncatedNormalOpTest : public OpsTestBase {
 protected:
  void Init(DataType dtype) {
    TF_CHECK_OK(NodeDefBuilder("op", "StatelessParameterizedTruncatedNormal")
                    .Input(FakeInput(DT_INT32))  // shape
                    .Input(FakeInput(DT_INT32))  // seed
                    .Input(FakeInput(dtype))     // means
                    .Input(FakeInput(dtype))     // stddevs
                    .Input(FakeInput(dtype))     // minvals
                    .Input(FakeInput(dtype))     // maxvals
                    .Attr("dtype", dtype)
                    .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(StatelessParameterizedTruncatedNormalOpTest, TestNormal) {
  Init(DT_FLOAT);

  // Normal non-overflowing shape: [100]
  AddInputFromList<int32_t>(TensorShape({1}), {100});
  // seed: shape [2]
  AddInputFromList<int32_t>(TensorShape({2}), {1, 2});

  AddInputFromList<float>(TensorShape({}), {0.0f});
  AddInputFromList<float>(TensorShape({}), {1.0f});
  AddInputFromList<float>(TensorShape({}), {-2.0f});
  AddInputFromList<float>(TensorShape({}), {2.0f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  ASSERT_NE(output, nullptr);
  auto flat_output = output->flat<float>();
  ASSERT_EQ(flat_output.size(), 100);

  bool all_zeros = true;
  for (int i = 0; i < 100; ++i) {
    float val = flat_output(i);
    EXPECT_GE(val, -2.0f);
    EXPECT_LE(val, 2.0f);
    EXPECT_FALSE(std::isnan(val));
    if (val != 0.0f) {
      all_zeros = false;
    }
  }
  EXPECT_FALSE(all_zeros)
      << "Returned uninitialized/zeroed memory instead of generating samples.";
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class ParameterizedTruncatedNormalOpGPUTest : public OpsTestBase {
 protected:
  void Init(DataType dtype) {
    std::unique_ptr<Device> device_gpu(
        DeviceFactory::NewDevice("GPU", {}, "/job:a/replica:0/task:0"));
    if (device_gpu == nullptr) {
      GTEST_SKIP() << "No GPU device registered.";
    }
    SetDevice(DEVICE_GPU, std::move(device_gpu));

    TF_CHECK_OK(NodeDefBuilder("op", "ParameterizedTruncatedNormal")
                    .Input(FakeInput(DT_INT32))  // shape
                    .Input(FakeInput(dtype))     // means
                    .Input(FakeInput(dtype))     // stddevs
                    .Input(FakeInput(dtype))     // minvals
                    .Input(FakeInput(dtype))     // maxvals
                    .Attr("dtype", dtype)
                    .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ParameterizedTruncatedNormalOpGPUTest, TestIntegerOverflowGPU) {
  Init(DT_FLOAT);
  if (device_type_ != DEVICE_GPU) return;

  // We pass shape that overflows 32-bit: [2, 1073741824]
  AddInputFromList<int32_t>(TensorShape({2}), {2, 1073741824});
  AddInputFromList<float>(TensorShape({1}), {0.0f});
  AddInputFromList<float>(TensorShape({1}), {1.0f});
  AddInputFromList<float>(TensorShape({1}), {-2.0f});
  AddInputFromList<float>(TensorShape({1}), {2.0f});

  absl::Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(s.message(),
            "Number of elements exceeds std::numeric_limits<int>::max()");
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
