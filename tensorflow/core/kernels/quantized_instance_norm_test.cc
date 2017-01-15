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

#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {

void ReferenceImpl(const quint8* inp, float inp_min, float inp_max,
                   const TensorShape& shape, float var_eps, float* out) {
  int N = shape.dim_size(0);
  int H = shape.dim_size(1);
  int W = shape.dim_size(2);
  int C = shape.dim_size(3);

  int total = N * H * W * C;
  float inp_scale = (inp_max - inp_min) / 255.0f;
  std::unique_ptr<float[]> dequantized(new float[total]);

  for (int i = 0; i < total; ++i) {
    dequantized[i] = inp_min + inp_scale * static_cast<float>(inp[i]);
  }

  std::unique_ptr<float[]> inp_mean(new float[N * C]);
  std::unique_ptr<float[]> inp_var(new float[N * C]);

  float img_size = static_cast<float>(H) * static_cast<float>(W);

  // Compute mean
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      float sum = 0.0;
      for (int i = 0; i < H * W; ++i) {
        sum += dequantized[n * H * W * C + i * C + c];
      }
      inp_mean[n * C + c] = sum / img_size;
    }
  }

  // Compute var
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      float sum = 0.0;
      for (int i = 0; i < H * W; ++i) {
        float tmp =
            dequantized[n * H * W * C + i * C + c] - inp_mean[n * C + c];
        sum += tmp * tmp;
      }
      inp_var[n * C + c] = sum / img_size;
    }
  }

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < H * W; ++i) {
        out[n * H * W * C + i * C + c] =
            (dequantized[n * H * W * C + i * C + c] - inp_mean[n * C + c]) /
            std::sqrt(inp_var[n * C + c] + var_eps);
      }
    }
  }
}

using namespace ops;  // NOLINT(build/namespaces)

namespace {

void Expect(const Tensor& input, float x_min, float x_max,
            bool output_range_given, float give_y_min, float given_y_max) {
  Scope root = Scope::NewRootScope();

  auto input_ph = Placeholder(root, DT_QUINT8);

  const float variance_eps = 1e-5;
  auto instance_norm = QuantizedInstanceNorm(
      root, input_ph, x_min, x_max,
      QuantizedInstanceNorm::Attrs().VarianceEpsilon(variance_eps));

  Status s = root.status();
  EXPECT_TRUE(s.ok());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  s = session.Run({{input_ph, input}},
                  {instance_norm.y, instance_norm.y_min, instance_norm.y_max},
                  &outputs);

  EXPECT_TRUE(s.ok());
  Tensor expected(DT_FLOAT, input.shape());

  ReferenceImpl(input.flat<quint8>().data(), x_min, x_max, input.shape(),
                variance_eps, expected.flat<float>().data());

  auto out = outputs[0].flat<quint8>();

  float out_min = outputs[1].flat<float>()(0);
  float out_max = outputs[2].flat<float>()(0);
  float out_scale = (out_max - out_min) / 255.0f;

  Eigen::Tensor<float, 0, Eigen::RowMajor> max_diff =
      (expected.flat<float>() - (out_min + out_scale * out.cast<float>()))
          .abs()
          .maximum();
  EXPECT_LE(max_diff(), 0.1);
  LOG(INFO) << "max diff " << max_diff();
}

}  // end namespace

void TestBasic() {
  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  // Random input
  input = input.random(Eigen::internal::UniformRandomGenerator<quint8>());

  Expect(input_tensor, 0.0f, 1.0f, false, 0.0f, 0.0f);
}

void TestZeroInput() {
  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  // Zero input, but input min > 0. Tests that output min and max should be
  // properly separated.
  input = input.setConstant(0);

  Expect(input_tensor, 2.0f, 3.0f, false, 0.0f, 0.0f);
}

void TestMaxInput() {
  Tensor input_tensor(DT_QUINT8, {1, 1, 2, 16});
  auto input = input_tensor.flat<quint8>();
  // Inputs are all FLT_MAX / (number of inputs).
  input = input.setConstant(255);

  Expect(input_tensor, 0.0f,
         std::numeric_limits<float>::max() / static_cast<float>(2 * 16), false,
         0.0f, 0.0f);
}

void TestOutputRangeGiven() {
  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  input = input.random(Eigen::internal::UniformRandomGenerator<quint8>());

  Expect(input_tensor, -10.0f, 10.0f, true, -1.0f, 1.0f);
}

void TestClamp() {
  Tensor input_tensor(DT_QUINT8, {1, 4, 4, 32});
  auto input = input_tensor.flat<quint8>();
  input = input.random(Eigen::internal::UniformRandomGenerator<quint8>());

  // Tests that negative outputs are clamped at 0.0, as the output range is
  // given to be (0.0, 1.0).
  Expect(input_tensor, -10.0f, 10.0f, true, 0.0f, 1.0f);
}

#if !defined(__ANDROID__)

#define RUN_TEST(t) \
  TEST(QuantizedInstanceNormTest, t) { t(); }

RUN_TEST(TestBasic);
RUN_TEST(TestZeroInput);
RUN_TEST(TestMaxInput);
RUN_TEST(TestOutputRangeGiven);
RUN_TEST(TestClamp);

#undef RUN_TEST

#endif  // __ANDROID__

}  // end namespace tensorflow

#if defined(__ANDROID__)
int main(int argc, char** argv) {
  tensorflow::TestBasic();
  tensorflow::TestZeroInput();
  tensorflow::TestMaxInput();
  tensorflow::TestOutputRangeGiven();
  tensorflow::TestClamp();
  return 0;
}
#endif  // __ANDROID__
