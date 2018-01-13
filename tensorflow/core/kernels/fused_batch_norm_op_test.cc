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

// The current implementation of fused batch norm assumes that several
// cudnn routines can perform the computation in place. This is not
// mentioned by the NVIDIA documentation. If any of the large tests below
// fails, this might mean the assumption is no longer true.

#include <cmath>
#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
class FusedBatchNormOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormOpTest, Training) {
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

TEST_F(FusedBatchNormOpTest, TrainingLargeFormatNHWC) {
#if GOOGLE_CUDA
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:b/replica:0/task:0"));
  SetDevice(DEVICE_GPU, std::move(device));
#endif
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("data_format", "NHWC")
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  int N = 2, C = 4, H = 100, W = 100;
  std::vector<float> input;
  std::vector<float> scale;
  std::vector<float> offset;

  input.reserve(N * H * W * C);
  for (int batch = 0; batch < N; ++batch)
    for (int i = 0; i < H * W; ++i)
      for (int feature = 0; feature < C; ++feature) input.push_back(2 * i + 1);
  AddInputFromArray<float>(TensorShape({N, H, W, C}), input);
  for (int feature = 0; feature < C; ++feature) {
    scale.push_back(2.5);
    offset.push_back(1.5);
  }
  AddInputFromArray<float>(TensorShape({C}), scale);
  AddInputFromArray<float>(TensorShape({C}), offset);
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  float mean = H * W;
  float variance = (2 * H * W - 1) * (2 * H * W + 1) / 3.0 - mean * mean;
  std::vector<float> result;
  result.reserve(N * H * W * C);
  for (int batch = 0; batch < N; ++batch)
    for (int i = 0; i < H * W; ++i)
      for (int feature = 0; feature < C; ++feature)
        result.push_back(2.5 * (2 * i + 1 - mean) / sqrt(variance + 0.001) +
                         1.5);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({N, H, W, C}));
  test::FillValues<float>(&expected, result);
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

  std::vector<float> mean_res, variance_res;
  for (int feature = 0; feature < C; ++feature) {
    mean_res.push_back(mean);
    variance_res.push_back(variance * N * H * W / (N * H * W - 1.0));
  }
  Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({C}));
  test::FillValues<float>(&expected_mean, mean_res);
  test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);
  // For large tests such as this, the accumulated error in the
  // variance computation is higher than 1. This is not surprising,
  // because the variance ~ 10^7.
  Tensor expected_variance(allocator(), DT_FLOAT, TensorShape({C}));
  test::FillValues<float>(&expected_variance, variance_res);
  test::ExpectTensorNear<float>(expected_variance, *GetOutput(2), 25.0);
}

TEST_F(FusedBatchNormOpTest, TrainingLargeFormatNCHW) {
#if GOOGLE_CUDA
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:b/replica:0/task:0"));
  SetDevice(DEVICE_GPU, std::move(device));
#endif
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("data_format", "NCHW")
                   .Attr("is_training", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  int N = 2, C = 4, H = 100, W = 100;
  std::vector<float> input;
  std::vector<float> scale;
  std::vector<float> offset;

  input.reserve(N * C * H * W);
  for (int batch = 0; batch < N; ++batch)
    for (int feature = 0; feature < C; ++feature)
      for (int i = 0; i < H * W; ++i) input.push_back(2 * i + 1);
  AddInputFromArray<float>(TensorShape({N, C, H, W}), input);
  for (int feature = 0; feature < C; ++feature) {
    scale.push_back(2.5);
    offset.push_back(1.5);
  }
  AddInputFromArray<float>(TensorShape({C}), scale);
  AddInputFromArray<float>(TensorShape({C}), offset);
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  Status status = RunOpKernel();
#if GOOGLE_CUDA
  TF_EXPECT_OK(status);

  float mean = H * W;
  float variance = (2 * H * W - 1) * (2 * H * W + 1) / 3.0 - mean * mean;
  std::vector<float> result;
  result.reserve(N * C * H * W);
  for (int batch = 0; batch < N; ++batch)
    for (int feature = 0; feature < C; ++feature)
      for (int i = 0; i < H * W; ++i)
        result.push_back(2.5 * (2 * i + 1 - mean) / sqrt(variance + 0.001) +
                         1.5);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({N, C, H, W}));
  test::FillValues<float>(&expected, result);
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

  std::vector<float> mean_res, variance_res;
  for (int feature = 0; feature < C; ++feature) {
    mean_res.push_back(mean);
    variance_res.push_back(variance * N * H * W / (N * H * W - 1));
  }
  Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({C}));
  test::FillValues<float>(&expected_mean, mean_res);
  test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);

  Tensor expected_variance(allocator(), DT_FLOAT, TensorShape({C}));
  test::FillValues<float>(&expected_variance, variance_res);
  test::ExpectTensorNear<float>(expected_variance, *GetOutput(2), 10.0);
#else  // The CPU version does not support format NCHW.
  EXPECT_TRUE(StringPiece(status.ToString())
                  .contains("The CPU implementation of FusedBatchNorm "
                            "only supports NHWC tensor format for now."))
      << status;
#endif
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

TEST_F(FusedBatchNormOpTest, InferenceLargeFormatNHWC) {
#if GOOGLE_CUDA
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:b/replica:0/task:0"));
  SetDevice(DEVICE_GPU, std::move(device));
#endif
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_op", "FusedBatchNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("data_format", "NHWC")
                   .Attr("is_training", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  int N = 2, C = 4, H = 100, W = 100;
  std::vector<float> input;
  std::vector<float> scale;
  std::vector<float> offset;
  std::vector<float> estimated_mean;
  std::vector<float> estimated_variance;

  input.reserve(N * H * W * C);
  for (int batch = 0; batch < N; ++batch)
    for (int i = 0; i < H * W; ++i)
      for (int feature = 0; feature < C; ++feature)
        input.push_back(1.7 * i + 1);
  AddInputFromArray<float>(TensorShape({N, H, W, C}), input);
  for (int feature = 0; feature < C; ++feature) {
    scale.push_back(3.0);
    offset.push_back(2.0);
    estimated_mean.push_back(2 * feature + 1);
    estimated_variance.push_back(13.0);
  }
  AddInputFromArray<float>(TensorShape({C}), scale);
  AddInputFromArray<float>(TensorShape({C}), offset);
  AddInputFromArray<float>(TensorShape({C}), estimated_mean);
  AddInputFromArray<float>(TensorShape({C}), estimated_variance);

  TF_ASSERT_OK(RunOpKernel());

  std::vector<float> result;
  result.reserve(N * H * W * C);
  for (int batch = 0; batch < N; ++batch)
    for (int i = 0; i < H * W; ++i)
      for (int feature = 0; feature < C; ++feature)
        result.push_back(3 * (1.7 * i - 2 * feature) / sqrt(13.001) + 2.0);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({N, H, W, C}));
  test::FillValues<float>(&expected, result);
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
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
}  // namespace tensorflow
