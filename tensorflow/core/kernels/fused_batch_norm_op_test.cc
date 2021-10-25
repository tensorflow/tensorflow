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

#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
class FusedBatchNormOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormOpTest, Training) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("exponential_avg_factor", 1.0)
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

  Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_mean, {10, 10});
  test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);

  Tensor expected_variance(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_variance, {14.00, 14.00});
  test::ExpectTensorNear<float>(expected_variance, *GetOutput(2), 0.01);
}

TEST_F(FusedBatchNormOpTest, TrainingRunningMean) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("exponential_avg_factor", 0.5)
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2}), {6.0, 6.0});
  AddInputFromArray<float>(TensorShape({2}), {16.0, 16.0});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

  Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_mean, {8, 8});
  test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);

  Tensor expected_variance(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_variance, {15.00, 15.00});
  test::ExpectTensorNear<float>(expected_variance, *GetOutput(2), 0.01);
}

TEST_F(FusedBatchNormOpTest, Inference) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2}), {10, 10});
  AddInputFromArray<float>(TensorShape({2}), {11.67f, 11.67f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

TEST_F(FusedBatchNormOpTest, InferenceIgnoreAvgFactor) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("exponential_avg_factor", 0.5)
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  AddInputFromArray<float>(TensorShape({2}), {4.0, 4.0});
  AddInputFromArray<float>(TensorShape({2}), {2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2}), {10, 10});
  AddInputFromArray<float>(TensorShape({2}), {11.67f, 11.67f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected, {-3.86, -3.86, -1.51, -1.51, 0.83, 0.83,
                                      3.17, 3.17, 5.51, 5.51, 7.86, 7.86});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

TEST_F(FusedBatchNormOpTest, EmptyInput) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 0, 0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), TensorShape({1, 1, 0, 0}));
}

class FusedBatchNormGradOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormGradOpTest, Simple) {
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_grad_op", "FusedBatchNormGrad")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {2, 2, 9, 9, -4, -4, 5, 5, 8, 8, 7, 7});
  AddInputFromArray<float>(TensorShape({1, 1, 6, 2}),
                           {1, 1, 7, 7, 4, 4, -3, -3, -11, -11, 13, 13});
  AddInputFromArray<float>(TensorShape({2}), {4, 4});
  AddInputFromArray<float>(TensorShape({2}), {1.833f, 1.833f});
  AddInputFromArray<float>(TensorShape({2}), {57.472f, 57.472f});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_x(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(&expected_x, {-1.34, -1.34, 2.47, 2.47, -4.44, -4.44,
                                        0.17, 0.17, 1.60, 1.60, 1.53, 1.53});
  test::ExpectTensorNear<float>(expected_x, *GetOutput(0), 0.01);

  Tensor expected_scale(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_scale, {-1.6488, -1.6488});
  test::ExpectTensorNear<float>(expected_scale, *GetOutput(1), 0.01);

  Tensor expected_offset(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_offset, {27, 27});
  test::ExpectTensorNear<float>(expected_offset, *GetOutput(2), 0.01);
}

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

using fp32 = float;
using fp16 = Eigen::half;

template <typename T>
static Graph* FusedBatchNormInference(int n, int h, int w, int c,
                                      bool is_training,
                                      TensorFormat data_format) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;
  Tensor x_t(dtype, data_format == FORMAT_NHWC ? TensorShape({n, h, w, c})
                                               : TensorShape({n, c, h, w}));
  x_t.flat<T>().setRandom();

  Tensor other_t(DT_FLOAT, TensorShape({c}));
  other_t.flat<float>().setRandom();

  Tensor empty_t(DT_FLOAT, TensorShape({0}));

  Node* x = test::graph::Constant(g, x_t, "x");
  Node* other = test::graph::Constant(g, other_t, "other");
  Node* empty = test::graph::Constant(g, empty_t, "empty");

  Node* fused_batch_norm;
  TF_CHECK_OK(NodeBuilder(g->NewName("fused_batch_norm"), "FusedBatchNormV3")
                  .Input(x)
                  .Input(other)                        // scale
                  .Input(other)                        // offset
                  .Input(is_training ? empty : other)  // mean
                  .Input(is_training ? empty : other)  // variance
                  .Attr("T", dtype)
                  .Attr("U", DT_FLOAT)
                  .Attr("epsilon", 0.001)
                  .Attr("is_training", is_training)
                  .Attr("data_format", ToString(data_format))
                  .Finalize(g, &fused_batch_norm));

  return g;
}

template <typename T>
static Graph* FusedBatchNormGrad(int n, int h, int w, int c, bool is_training,
                                 TensorFormat data_format) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;
  TensorShape shape = data_format == FORMAT_NHWC ? TensorShape({n, h, w, c})
                                                 : TensorShape({n, c, h, w});

  Tensor y_backprop_t(dtype, shape);
  y_backprop_t.flat<T>().setRandom();

  Tensor x_t(dtype, shape);
  x_t.flat<T>().setRandom();

  Tensor other_t(DT_FLOAT, TensorShape({c}));
  other_t.flat<float>().setRandom();

  Node* y_backprop = test::graph::Constant(g, y_backprop_t, "y_backprop");
  Node* x = test::graph::Constant(g, x_t, "x");
  Node* other = test::graph::Constant(g, other_t, "other");

  Node* fused_batch_norm;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("fused_batch_norm_grad"), "FusedBatchNormGradV3")
          .Input(y_backprop)
          .Input(x)
          .Input(other)  // scale
          .Input(other)  // saved_mean_or_pop_mean
          .Input(other)  // saved_maybe_inv_var_or_pop_var
          .Input(other)  // reserve_space
          .Attr("T", dtype)
          .Attr("U", DT_FLOAT)
          .Attr("epsilon", 0.001)
          .Attr("is_training", is_training)
          .Attr("data_format", ToString(data_format))
          .Finalize(g, &fused_batch_norm));

  return g;
}

#define BM_NAME(NAME, N, H, W, C, T, IT, FORMAT, DEVICE) \
  BM_##NAME##_##N##_##H##_##W##_##C##_##IT##_##FORMAT##_##T##_##DEVICE

// -------------------------------------------------------------------------- //
// FusedBatchNorm inference
// -------------------------------------------------------------------------- //
// clang-format off
// NOLINTBEGIN
#define BM_FusedBatchNorm(N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)         \
  static void BM_NAME(FusedBatchNorm, N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)(::testing::benchmark::State & state) {                     \
    test::Benchmark(                                                          \
        #DEVICE,                                                              \
        FusedBatchNormInference<T>(N, H, W, C, IS_TRAINING, FORMAT_##FORMAT), \
        /*old_benchmark_api*/ false)                                          \
        .Run(state);                                                          \
    state.SetItemsProcessed(state.iterations() * N * H * W * C);              \
  }                                                                           \
  BENCHMARK(                                                                  \
      BM_NAME(FusedBatchNorm, N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE))    \
      ->UseRealTime();

// NOLINTEND
// clang-format on

BM_FusedBatchNorm(64, 14, 14, 256, fp32, false, NHWC, cpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, false, NHWC, cpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, true, NHWC, cpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, true, NHWC, cpu);

#ifdef GOOGLE_CUDA
BM_FusedBatchNorm(64, 14, 14, 256, fp32, false, NHWC, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, false, NHWC, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, false, NCHW, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, false, NCHW, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, true, NHWC, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, true, NHWC, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, true, NCHW, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, true, NCHW, gpu);
#endif  // GOOGLE_CUDA

// -------------------------------------------------------------------------- //
// FusedBatchNorm gradient
// -------------------------------------------------------------------------- //

#define BM_FusedBatchNormGrad(N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)      \
  static void BM_NAME(FusedBatchNormGrad, N, H, W, C, T, IS_TRAINING, FORMAT,  \
                      DEVICE)(::testing::benchmark::State & state) {           \
    test::Benchmark(                                                           \
        #DEVICE,                                                               \
        FusedBatchNormGrad<T>(N, H, W, C, IS_TRAINING, FORMAT_##FORMAT),       \
        /*old_benchmark_api*/ false)                                           \
        .Run(state);                                                           \
    state.SetItemsProcessed(state.iterations() * N * H * W * C);               \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_NAME(FusedBatchNormGrad, N, H, W, C, T, IS_TRAINING, FORMAT, DEVICE)) \
      ->UseRealTime();

#define BM_FusedBatchNormGradResnetShapes(T, IS_TRAINING, FORMAT, DEVICE) \
  BM_FusedBatchNormGrad(64, 56, 56, 64, T, IS_TRAINING, FORMAT, DEVICE);  \
  BM_FusedBatchNormGrad(64, 56, 56, 128, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 56, 56, 256, T, IS_TRAINING, FORMAT, DEVICE); \
                                                                          \
  BM_FusedBatchNormGrad(64, 28, 28, 128, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 28, 28, 256, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 28, 28, 512, T, IS_TRAINING, FORMAT, DEVICE); \
                                                                          \
  BM_FusedBatchNormGrad(64, 14, 14, 128, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 14, 14, 256, T, IS_TRAINING, FORMAT, DEVICE); \
  BM_FusedBatchNormGrad(64, 14, 14, 1024, T, IS_TRAINING, FORMAT, DEVICE)

BM_FusedBatchNormGradResnetShapes(fp32, true, NHWC, cpu);
BM_FusedBatchNormGradResnetShapes(fp32, false, NHWC, cpu);

#ifdef GOOGLE_CUDA
BM_FusedBatchNormGradResnetShapes(fp32, true, NHWC, gpu);
BM_FusedBatchNormGradResnetShapes(fp16, true, NHWC, gpu);
BM_FusedBatchNormGradResnetShapes(fp32, true, NCHW, gpu);
BM_FusedBatchNormGradResnetShapes(fp16, true, NCHW, gpu);

BM_FusedBatchNormGradResnetShapes(fp32, false, NHWC, gpu);
BM_FusedBatchNormGradResnetShapes(fp16, false, NHWC, gpu);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
