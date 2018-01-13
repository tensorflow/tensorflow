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

class FusedBatchNormGradOpTest : public OpsTestBase {};

TEST_F(FusedBatchNormGradOpTest, Simple) {
#if GOOGLE_CUDA
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:b/replica:0/task:0"));
  SetDevice(DEVICE_GPU, std::move(device));
#endif
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
#if GOOGLE_CUDA
  AddInputFromArray<float>(TensorShape({2}), {0.1319f, 0.1319f});
#else
  AddInputFromArray<float>(TensorShape({2}), {57.472f, 57.472f});
#endif

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

TEST_F(FusedBatchNormGradOpTest, LargeFormatNHWC) {
#if GOOGLE_CUDA
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:b/replica:0/task:0"));
  SetDevice(DEVICE_GPU, std::move(device));
#endif
  TF_EXPECT_OK(NodeDefBuilder("batch_norm_grad_op", "FusedBatchNormGrad")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Attr("data_format", "NHWC")
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  int N = 2, C = 3, H = 100, W = 100;
  std::vector<float> y_backprop;
  std::vector<float> x;
  std::vector<float> scale;
  std::vector<float> saved_mean;
  std::vector<float> saved_variance;
  float variance;
#if GOOGLE_CUDA  // cudnn saves the inverted variance in the forward pass
  variance = 1.0 / sqrt(13.001);
#else
  variance = 13.0;
#endif

  y_backprop.reserve(N * H * W * C);
  x.reserve(N * H * W * C);
  for (int batch = 0; batch < N; ++batch)
    for (int i = 0; i < H * W; ++i)
      for (int feature = 0; feature < C; ++feature) {
        y_backprop.push_back((i % 7) + 1);
        x.push_back(i % 13);
      }
  for (int feature = 0; feature < C; ++feature) {
    scale.push_back(1.5);
    saved_mean.push_back(6.0 + feature);
    saved_variance.push_back(variance);
  }
  AddInputFromArray<float>(TensorShape({N, H, W, C}), y_backprop);
  AddInputFromArray<float>(TensorShape({N, H, W, C}), x);
  AddInputFromArray<float>(TensorShape({C}), scale);
  AddInputFromArray<float>(TensorShape({C}), saved_mean);
  AddInputFromArray<float>(TensorShape({C}), saved_variance);

  TF_ASSERT_OK(RunOpKernel());

  std::vector<float> x_backprop;
  std::vector<float> res_scale;
  std::vector<float> res_offset;

  float offset = 0.0;
  for (int i = 0; i < H * W; ++i) offset += ((i % 7) + 1) * N;
  for (int feature = 0; feature < C; ++feature) {
    float feat_scale = 0.0;
    for (int i = 0; i < H * W; ++i)
      feat_scale += ((i % 7) + 1) * ((i % 13) - 6.0 - feature) / sqrt(13.001);
    res_scale.push_back(feat_scale * N);
    res_offset.push_back(offset);
  }
  for (int batch = 0; batch < N; ++batch)
    for (int i = 0; i < H * W; ++i)
      for (int feature = 0; feature < C; ++feature) {
        float coef0 = 1.5 / sqrt(13.001);
        float coef1 = res_scale[feature] / (N * H * W * sqrt(13.001));
        float y_mean = offset / (N * H * W);
        float y_centered = (i % 7) + 1 - y_mean;
        x_backprop.push_back(coef0 *
                             (y_centered - ((i % 13) - 6.0 - feature) * coef1));
      }

  Tensor expected_x(allocator(), DT_FLOAT, TensorShape({N, H, W, C}));
  test::FillValues<float>(&expected_x, x_backprop);
  test::ExpectTensorNear<float>(expected_x, *GetOutput(0), 0.01);

  Tensor expected_scale(allocator(), DT_FLOAT, TensorShape({C}));
  test::FillValues<float>(&expected_scale, res_scale);
  test::ExpectTensorNear<float>(expected_scale, *GetOutput(1), 1.0);

  Tensor expected_offset(allocator(), DT_FLOAT, TensorShape({C}));
  test::FillValues<float>(&expected_offset, res_offset);
  test::ExpectTensorNear<float>(expected_offset, *GetOutput(2), 0.01);
}
}  // namespace tensorflow
