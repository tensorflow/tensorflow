/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace {
template <typename T>
void TestOneResizeBilinear(int batch, int depth, int input_width,
                           int input_height, int output_width,
                           int output_height, float error_threshold) {
  RuntimeShape input_dims_inference({batch, input_height, input_width, depth});
  RuntimeShape output_dims_inference(
      {batch, output_height, output_width, depth});

  const int input_buffer_size = input_dims_inference.FlatSize();
  const int output_buffer_size = output_dims_inference.FlatSize();

  std::vector<T> input_data(input_buffer_size, 0);
  std::vector<T> reference_output_data(output_buffer_size, 0);
  // Initialize the output data with something other than zero, so we can catch
  // issue with kernels failing to initialize the output.
  std::vector<T> output_data(output_buffer_size, 3);

  const T min_amplitude = static_cast<T>(0);
  const T max_amplitude = static_cast<T>(255);
  FillRandom(&input_data, min_amplitude, max_amplitude);

  RuntimeShape output_size_dims({1, 1, 1, 2});
  std::vector<int32> output_size_data = {output_height, output_width};

  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = false;

  reference_ops::ResizeBilinear(op_params, input_dims_inference,
                                input_data.data(), output_size_dims,
                                output_size_data.data(), output_dims_inference,
                                reference_output_data.data());
  optimized_ops::ResizeBilinear(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());

  double sum_diff = 0;
  float max_abs_val = 0;
  for (int i = 0; i < output_buffer_size; i++) {
    sum_diff += std::abs(static_cast<float>(output_data[i]) -
                         static_cast<float>(reference_output_data[i]));
    max_abs_val = std::max(
        max_abs_val, std::abs(static_cast<float>(reference_output_data[i])));
  }

  if (sum_diff != 0.f) {
    const float mean_diff = static_cast<float>(sum_diff / output_buffer_size);
    const float relative_error = std::abs(mean_diff) / max_abs_val;
    ASSERT_LT(relative_error, error_threshold);
  }
}

TEST(ResizeBilinear, TestResizeBilinear8Bit) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 100 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_height = ExponentialRandomPositiveInt(0.9f, 20, 200);

    TestOneResizeBilinear<uint8>(batch, depth, input_width, input_height,
                                 output_width, output_height, 0.025);
  }
}

TEST(ResizeBilinear2x2, TestResizeBilinear8Bit) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 100 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = input_width * 2;
    const int output_height = input_height * 2;

    TestOneResizeBilinear<uint8>(batch, depth, input_width, input_height,
                                 output_width, output_height, 1e-5);
  }
}

TEST(ResizeBilinear, TestResizeBilinear) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 100 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_height = ExponentialRandomPositiveInt(0.9f, 20, 200);

    TestOneResizeBilinear<float>(batch, depth, input_width, input_height,
                                 output_width, output_height, 1e-5);
  }
}

TEST(ResizeBilinear2x2, TestResizeBilinear) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 100 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = input_width * 2;
    const int output_height = input_height * 2;

    TestOneResizeBilinear<float>(batch, depth, input_width, input_height,
                                 output_width, output_height, 1e-5);
  }
}
}  // namespace
}  // namespace tflite
