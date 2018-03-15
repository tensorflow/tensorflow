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

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/batch_norm_op.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class QuantizedBatchNormOpTest : public OpsTestBase {};

TEST_F(QuantizedBatchNormOpTest, Simple) {
  TF_EXPECT_OK(NodeDefBuilder("quantized_batch_norm_op",
                              "QuantizedBatchNormWithGlobalNormalization")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("scale_after_normalization", false)
                   .Attr("variance_epsilon", 0.001)
                   .Attr("Tinput", DT_QUINT8)
                   .Attr("out_type", DT_QINT32)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const int input_batch = 1;
  const int input_height = 1;
  const int input_width = 6;
  const int input_depth = 2;
  Tensor input_float(DT_FLOAT,
                     {input_batch, input_height, input_width, input_depth});
  test::FillValues<float>(&input_float,
                          {1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);
  const float mean_min = 0.0f;
  const float mean_max = 20.0f;
  Tensor mean_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&mean_float, {10, 20});
  Tensor mean_quantized =
      FloatTensorToQuantized<quint8>(mean_float, mean_min, mean_max);
  const float variance_min = 0.0f;
  const float variance_max = 1.0f;
  Tensor variance_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&variance_float, {0.25, 0.5});
  Tensor variance_quantized = FloatTensorToQuantized<quint8>(
      variance_float, variance_min, variance_max);
  const float beta_min = 0.0f;
  const float beta_max = 1.0f;
  Tensor beta_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&beta_float, {0.1, 0.6});
  Tensor beta_quantized =
      FloatTensorToQuantized<quint8>(beta_float, beta_min, beta_max);
  const float gamma_min = 0.0f;
  const float gamma_max = 1.0f;
  Tensor gamma_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&gamma_float, {0.0, 0.0});
  Tensor gamma_quantized =
      FloatTensorToQuantized<quint8>(gamma_float, gamma_min, gamma_max);

  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {input_min});
  AddInputFromArray<float>(TensorShape({1}), {input_max});
  AddInputFromArray<quint8>(mean_quantized.shape(),
                            mean_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {mean_min});
  AddInputFromArray<float>(TensorShape({1}), {mean_max});
  AddInputFromArray<quint8>(variance_quantized.shape(),
                            variance_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {variance_min});
  AddInputFromArray<float>(TensorShape({1}), {variance_max});
  AddInputFromArray<quint8>(beta_quantized.shape(),
                            beta_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {beta_min});
  AddInputFromArray<float>(TensorShape({1}), {beta_max});
  AddInputFromArray<quint8>(gamma_quantized.shape(),
                            gamma_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {gamma_min});
  AddInputFromArray<float>(TensorShape({1}), {gamma_max});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_float(
      allocator(), DT_FLOAT,
      TensorShape({input_batch, input_height, input_width, input_depth}));
  test::FillValues<float>(
      &expected_float, {-17.86, -22.00, -15.87, -20.59, -13.87, -19.18, -21.86,
                        -33.31, -23.85, -34.72, -25.85, -36.13});
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.1);
}

TEST_F(QuantizedBatchNormOpTest, SameAsFloat) {
  TF_EXPECT_OK(NodeDefBuilder("quantized_batch_norm_op",
                              "QuantizedBatchNormWithGlobalNormalization")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("scale_after_normalization", false)
                   .Attr("variance_epsilon", 0.001)
                   .Attr("Tinput", DT_QUINT8)
                   .Attr("out_type", DT_QINT32)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const int input_batch = 1;
  const int input_height = 1;
  const int input_width = 6;
  const int input_depth = 2;
  Tensor input_float(DT_FLOAT,
                     {input_batch, input_height, input_width, input_depth});
  test::FillValues<float>(&input_float,
                          {1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);
  const float mean_min = 0.0f;
  const float mean_max = 20.0f;
  Tensor mean_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&mean_float, {10, 20});
  Tensor mean_quantized =
      FloatTensorToQuantized<quint8>(mean_float, mean_min, mean_max);
  const float variance_min = 0.0f;
  const float variance_max = 1.0f;
  Tensor variance_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&variance_float, {0.25, 0.5});
  Tensor variance_quantized = FloatTensorToQuantized<quint8>(
      variance_float, variance_min, variance_max);
  const float beta_min = 0.0f;
  const float beta_max = 1.0f;
  Tensor beta_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&beta_float, {0.1, 0.6});
  Tensor beta_quantized =
      FloatTensorToQuantized<quint8>(beta_float, beta_min, beta_max);
  const float gamma_min = 0.0f;
  const float gamma_max = 1.0f;
  Tensor gamma_float(DT_FLOAT, {input_depth});
  test::FillValues<float>(&gamma_float, {0.0, 0.0});
  Tensor gamma_quantized =
      FloatTensorToQuantized<quint8>(gamma_float, gamma_min, gamma_max);

  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {input_min});
  AddInputFromArray<float>(TensorShape({1}), {input_max});
  AddInputFromArray<quint8>(mean_quantized.shape(),
                            mean_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {mean_min});
  AddInputFromArray<float>(TensorShape({1}), {mean_max});
  AddInputFromArray<quint8>(variance_quantized.shape(),
                            variance_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {variance_min});
  AddInputFromArray<float>(TensorShape({1}), {variance_max});
  AddInputFromArray<quint8>(beta_quantized.shape(),
                            beta_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {beta_min});
  AddInputFromArray<float>(TensorShape({1}), {beta_max});
  AddInputFromArray<quint8>(gamma_quantized.shape(),
                            gamma_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {gamma_min});
  AddInputFromArray<float>(TensorShape({1}), {gamma_max});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_float(
      allocator(), DT_FLOAT,
      TensorShape({input_batch, input_height, input_width, input_depth}));
  thread::ThreadPool threadpool(Env::Default(), "test", 1);
  EigenThreadPoolWrapper wrapper(&threadpool);
  Eigen::ThreadPoolDevice eigen_cpu_device(&wrapper, 1);
  const Tensor& const_input_float = input_float;
  const Tensor& const_mean_float = mean_float;
  const Tensor& const_variance_float = variance_float;
  const Tensor& const_beta_float = beta_float;
  const Tensor& const_gamma_float = gamma_float;
  functor::BatchNorm<Eigen::ThreadPoolDevice, float>()(
      eigen_cpu_device, const_input_float.tensor<float, 4>(),
      const_mean_float.vec<float>(), const_variance_float.vec<float>(),
      const_beta_float.vec<float>(), const_gamma_float.vec<float>(), 0.001,
      false, expected_float.tensor<float, 4>());

  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.1);
}

}  // namespace tensorflow
