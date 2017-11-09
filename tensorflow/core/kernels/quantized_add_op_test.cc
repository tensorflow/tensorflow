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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

using namespace ops;  // NOLINT(build/namespaces)

namespace {

void TestAdd(const std::vector<int64>& x_shape,
             const std::vector<float>& x_values, float x_min_value,
             float x_max_value, const std::vector<int64>& y_shape,
             const std::vector<float>& y_values, float y_min_value,
             float y_max_value, const std::vector<int64>& expected_shape,
             const std::vector<float>& expected_values, double tolerance) {
  Scope root = Scope::NewRootScope();

  Tensor x_float_tensor(DT_FLOAT, TensorShape(x_shape));
  test::FillValues<float>(&x_float_tensor, x_values);
  Tensor x_quantized_tensor(DT_QUINT8, x_float_tensor.shape());
  FloatTensorToQuantizedInPlace<quint8>(x_float_tensor, x_min_value,
                                        x_max_value, &x_quantized_tensor);
  Output x =
      Const(root.WithOpName("x"), Input::Initializer(x_quantized_tensor));
  Output x_min = Const(root.WithOpName("x_min"), x_min_value);
  Output x_max = Const(root.WithOpName("x_max"), x_max_value);

  Tensor y_float_tensor(DT_FLOAT, TensorShape(y_shape));
  test::FillValues<float>(&y_float_tensor, y_values);
  Tensor y_quantized_tensor(DT_QUINT8, y_float_tensor.shape());
  FloatTensorToQuantizedInPlace<quint8>(y_float_tensor, y_min_value,
                                        y_max_value, &y_quantized_tensor);
  Output y =
      Const(root.WithOpName("y"), Input::Initializer(y_quantized_tensor));
  Output y_min = Const(root.WithOpName("y_min"), y_min_value);
  Output y_max = Const(root.WithOpName("y_max"), y_max_value);

  ops::QuantizedAdd add = ops::QuantizedAdd(root.WithOpName("add"), x, y, x_min,
                                            x_max, y_min, y_max);

  TF_EXPECT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run(ClientSession::FeedType(),
                           {add.z, add.min_z, add.max_z}, &outputs));

  const Tensor& z_quantized = outputs[0];
  const float z_min = outputs[1].flat<float>()(0);
  const float z_max = outputs[2].flat<float>()(0);

  Tensor z_float = QuantizedTensorToFloat<qint32>(z_quantized, z_min, z_max);
  Tensor expected_z_float(DT_FLOAT, TensorShape(expected_shape));
  test::FillValues<float>(&expected_z_float, expected_values);
  test::ExpectTensorNear<float>(expected_z_float, z_float, tolerance);
}

void TestAddShape(const std::vector<int64>& x_shape,
                  const std::vector<int64>& y_shape) {
  const size_t x_num_elements = TensorShape(x_shape).num_elements();
  std::vector<float> x_values(x_num_elements);
  for (int i = 0; i < x_num_elements; ++i) {
    x_values[i] = i % 256;
  }
  const float x_min_value = 0.0f;
  const float x_max_value = 256.0f;

  const size_t y_num_elements = TensorShape(y_shape).num_elements();
  std::vector<float> y_values(y_num_elements);
  for (int i = 0; i < y_num_elements; ++i) {
    y_values[i] = ((i + 23) % 123) - 50;
  }
  const float y_min_value = -150.0f;
  const float y_max_value = 150.0f;

  Scope root = Scope::NewRootScope();

  Tensor x_float_tensor(DT_FLOAT, TensorShape(x_shape));
  test::FillValues<float>(&x_float_tensor, x_values);
  Output x = Const(root.WithOpName("x"), Input::Initializer(x_float_tensor));

  Tensor y_float_tensor(DT_FLOAT, TensorShape(y_shape));
  test::FillValues<float>(&y_float_tensor, y_values);
  Output y = Const(root.WithOpName("y"), Input::Initializer(y_float_tensor));

  Add add = Add(root.WithOpName("add"), x, y);

  TF_EXPECT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run(ClientSession::FeedType(), {add.z}, &outputs));

  const Tensor& expected_values_tensor = outputs[0];
  const float* expected_values_data =
      expected_values_tensor.flat<float>().data();
  std::vector<float> expected_values(
      expected_values_data,
      expected_values_data + expected_values_tensor.NumElements());
  std::vector<int64> expected_shape;
  for (const int64 dim : expected_values_tensor.shape().dim_sizes()) {
    expected_shape.push_back(dim);
  }
  TestAdd(x_shape, x_values, x_min_value, x_max_value, y_shape, y_values,
          y_min_value, y_max_value, expected_shape, expected_values, 256.0);
}

void TimeAdd(const std::vector<int64>& x_shape,
             const std::vector<int64>& y_shape, int64 iterations) {
  TestAddShape(x_shape, y_shape);

  Scope root = Scope::NewRootScope();

  Tensor x_quantized_tensor(DT_QUINT8, TensorShape(x_shape));
  Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_QUINT8);
  Output x_min = Const(root.WithOpName("x_min"), 0.0f);
  Output x_max = Const(root.WithOpName("x_max"), 1.0f);

  Tensor y_quantized_tensor(DT_QUINT8, TensorShape(y_shape));
  Output y =
      Const(root.WithOpName("y"), Input::Initializer(y_quantized_tensor));
  Output y_min = Const(root.WithOpName("y_min"), 0.0f);
  Output y_max = Const(root.WithOpName("y_max"), 1.0f);

  ops::QuantizedAdd add = ops::QuantizedAdd(root.WithOpName("add"), placeholder,
                                            y, x_min, x_max, y_min, y_max);

  TF_EXPECT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  int64 total_duration = 0;
  for (int i = 0; i < iterations; ++i) {
    const int64 start_time = Env::Default()->NowMicros();
    TF_EXPECT_OK(session.Run({{placeholder, x_quantized_tensor}},
                             {add.z, add.min_z, add.max_z}, &outputs));
    const int64 end_time = Env::Default()->NowMicros();
    total_duration += end_time - start_time;
  }
  const int64 one_run_duration = total_duration / iterations;

  const int64 num_ops = outputs[0].NumElements();

  const double million_ops_per_second =
      (iterations * num_ops) / static_cast<double>(total_duration);

  LOG(INFO) << "TimeAdd: " << TensorShape(x_shape).DebugString() << " * "
            << TensorShape(y_shape).DebugString()
            << ": iterations=" << iterations
            << ", MOps/s=" << million_ops_per_second
            << ", one_run_duration=" << one_run_duration
            << ", total_duration=" << total_duration;
}

}  // namespace

void TestManualScalar() {
  TestAdd(
      {10}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
      10.0f, {1}, {10.0f}, -100.0f, 100.0f, {10},
      {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      1.0f);
  TestAdd(
      {1}, {10.0f}, -100.0f, 100.0f, {10},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
      10.0f, {10},
      {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      1.0f);
}

void TestScalar() {
  TestAddShape({100}, {1});
  TestAddShape({1}, {100});
}

void TestManualVector() {
  TestAdd({10}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
          0.0f, 10.0f, {10},
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
          10.0f, {10},
          {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f},
          1.0f);
}

void TestVector() { TestAddShape({100}, {100}); }

void TestManualVectorPlusTensor() {
  TestAdd(
      {10}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
      10.0f, {2, 10},
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      0.0f, 20.0f, {2, 10},
      {2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f,
       12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f},
      1.0f);
  TestAdd({2, 10}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                    8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                    15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
          0.0f, 20.0f, {10},
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, 0.0f,
          10.0f, {2, 10}, {2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f, 14.0f,
                           16.0f, 18.0f, 20.0f, 12.0f, 14.0f, 16.0f, 18.0f,
                           20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f},
          1.0f);
  TestAdd(
      {5, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
      0.0f, 10.0f, {2, 5, 2},
      {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
       11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
      0.0f, 20.0f, {2, 5, 2},
      {2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f,
       12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f},
      1.0f);
}

void TestVectorPlusTensor() {
  TestAddShape({100}, {2, 100});
  TestAddShape({2, 100}, {100});
  TestAddShape({5, 2}, {2, 5, 2});
}

void BenchmarkTensorScalar() {
  TimeAdd({200}, {1}, 1000);
  TimeAdd({10000}, {1}, 100);
  TimeAdd({1000000}, {1}, 10);
  TimeAdd({10000000}, {1}, 1);
}

void BenchmarkVector() {
  TimeAdd({200}, {200}, 1000);
  TimeAdd({10000}, {10000}, 100);
  TimeAdd({1000000}, {1000000}, 10);
  TimeAdd({10000000}, {10000000}, 1);
}

void BenchmarkVectorPlusTensor() {
  TimeAdd({10, 20}, {20}, 100);
  TimeAdd({10, 1000}, {1000}, 10);
  TimeAdd({1000, 1000}, {1000}, 1);
  TimeAdd({10000, 1000}, {1000}, 1);
  TimeAdd({100, 100}, {100}, 10);
  TimeAdd({10000, 100}, {100}, 1);
  TimeAdd({100000, 100}, {100}, 1);
}

#if !defined(__ANDROID__)

#define RUN_TEST(t) \
  TEST(QuantizedAddOpTest, t) { t(); }

RUN_TEST(TestManualScalar);
RUN_TEST(TestManualVector);
RUN_TEST(TestManualVectorPlusTensor);
RUN_TEST(TestScalar);
RUN_TEST(TestVector);
RUN_TEST(TestVectorPlusTensor);

#undef RUN_TEST

#endif  // __ANDROID__

}  // end namespace tensorflow

#if defined(__ANDROID__)
int main(int argc, char** argv) {
  LOG(INFO) << "TestManualScalar:";
  tensorflow::TestManualScalar();
  LOG(INFO) << "TestManualVector:";
  tensorflow::TestManualVector();
  LOG(INFO) << "TestManualVectorPlusTensor:";
  tensorflow::TestManualVectorPlusTensor();
  tensorflow::BenchmarkTensorScalar();
  tensorflow::BenchmarkVector();
  tensorflow::BenchmarkVectorPlusTensor();
  LOG(INFO) << "All tests complete";
  return 0;
}
#endif  // __ANDROID__
