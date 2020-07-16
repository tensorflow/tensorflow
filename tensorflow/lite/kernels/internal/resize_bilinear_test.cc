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
#include <list>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace {
template <typename T>
void TestOneResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           int batch, int depth, int input_width,
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

class ResizeBilinearImplTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<tflite::ResizeBilinearParams> {};

TEST_P(ResizeBilinearImplTest, TestResizeBilinear8Bit) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 1000;
  const tflite::ResizeBilinearParams op_params = GetParam();

  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = UniformRandomInt(1, 2);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_height = ExponentialRandomPositiveInt(0.9f, 20, 200);

    TestOneResizeBilinear<uint8>(op_params, batch, depth, input_width,
                                 input_height, output_width, output_height,
                                 0.025);
  }
}

TEST_P(ResizeBilinearImplTest, TestResizeBilinear8Bit_2x2) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 1000;
  const tflite::ResizeBilinearParams op_params = GetParam();

  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = UniformRandomInt(1, 2);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = input_width * 2;
    const int output_height = input_height * 2;

    float error_threshold = 1e-5;
    if (op_params.align_corners) {
      // Align_corners causes small discrepencies between reference & optimized
      // versions.
      error_threshold = 3e-4;
    }
    TestOneResizeBilinear<uint8>(op_params, batch, depth, input_width,
                                 input_height, output_width, output_height,
                                 error_threshold);
  }
}

TEST_P(ResizeBilinearImplTest, TestResizeBilinear) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 1000;
  const tflite::ResizeBilinearParams op_params = GetParam();

  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = UniformRandomInt(1, 2);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_height = ExponentialRandomPositiveInt(0.9f, 20, 200);

    float error_threshold = 1e-5;
    if (op_params.align_corners) {
      // align_corners causes small discrepencies between reference & optimized
      // versions.
      error_threshold = 1e-4;
    }
    TestOneResizeBilinear<float>(op_params, batch, depth, input_width,
                                 input_height, output_width, output_height,
                                 error_threshold);
  }
}

TEST_P(ResizeBilinearImplTest, TestResizeBilinear_2x2) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 1000;
  const tflite::ResizeBilinearParams op_params = GetParam();

  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = UniformRandomInt(1, 2);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = input_width * 2;
    const int output_height = input_height * 2;

    float error_threshold = 1e-5;
    if (op_params.align_corners) {
      // Align_corners causes small discrepencies between reference & optimized
      // versions.
      error_threshold = 1e-4;
    }
    TestOneResizeBilinear<float>(op_params, batch, depth, input_width,
                                 input_height, output_width, output_height,
                                 error_threshold);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ResizeBilinear, ResizeBilinearImplTest,
    ::testing::ValuesIn(std::list<tflite::ResizeBilinearParams>({
        {/**align_corners**/ false, /**half_pixel_centers**/ false},
        {/**align_corners**/ false, /**half_pixel_centers**/ true},
        {/**align_corners**/ true, /**half_pixel_centers**/ false},
    })));

// A couple of tests to ensure the math behind half_pixel_centers works fine.

TEST(ResizeBilinear, TestResizeBilinearHalfPixelCenters_3x3to2x2) {
  // Input: 3x3
  RuntimeShape input_dims_inference({1, 3, 3, 1});
  // clang-format off
  std::vector<float> input_data = {1, 2, 3,
                                   4, 5, 6,
                                   7, 8, 9};
  // clang-format on

  // Output: 2x2
  RuntimeShape output_dims_inference({1, 2, 2, 1});
  // Initialize the output data with something other than zero, so we can catch
  // issue with kernels failing to initialize the output.
  const int output_buffer_size = output_dims_inference.FlatSize();
  std::vector<float> output_data(output_buffer_size, 3);

  RuntimeShape output_size_dims({1, 1, 1, 2});
  std::vector<int32> output_size_data = {2, 2};

  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = false;
  op_params.half_pixel_centers = false;

  // Test with half_pixel_centers = false.
  reference_ops::ResizeBilinear(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  // clang-format off
  std::vector<float> reference_half_pixel_centers_false = {1, 2.5,
                                                           5.5, 7};
  // clang-format on
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_false[i]));
  }

  // Test with half_pixel_centers = true.
  op_params.half_pixel_centers = true;
  reference_ops::ResizeBilinear(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  // clang-format off
  std::vector<float> reference_half_pixel_centers_true = {2, 3.5,
                                                          6.5, 8};
  // clang-format on
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_true[i]));
  }
}

TEST(ResizeBilinear, TestResizeBilinearHalfPixelCenters_2x2to4x4) {
  // Input: 2x2
  RuntimeShape input_dims_inference({1, 2, 2, 1});
  // clang-format off
  std::vector<float> input_data = {1, 2,
                                   3, 4};
  // clang-format on

  // Output: 2x2
  RuntimeShape output_dims_inference({1, 4, 4, 1});
  // Initialize the output data with something other than zero, so we can catch
  // issue with kernels failing to initialize the output.
  const int output_buffer_size = output_dims_inference.FlatSize();
  std::vector<float> output_data(output_buffer_size, 3);

  RuntimeShape output_size_dims({1, 1, 1, 2});
  std::vector<int32> output_size_data = {4, 4};

  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = false;
  op_params.half_pixel_centers = false;

  // Test with half_pixel_centers = false.
  reference_ops::ResizeBilinear(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  // clang-format off
  std::vector<float> reference_half_pixel_centers_false =
      {1,  1.5, 2, 2,
       2,  2.5, 3, 3,
       3,  3.5, 4, 4,
       3,  3.5, 4, 4};
  // clang-format on
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_false[i]));
  }

  // Test with half_pixel_centers = true.
  op_params.half_pixel_centers = true;
  reference_ops::ResizeBilinear(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  // clang-format off
  std::vector<float> reference_half_pixel_centers_true =
      {1,    1.25, 1.75, 2,
       1.5,  1.75, 2.25, 2.5,
       2.5,  2.75, 3.25, 3.5,
       3,    3.25, 3.75, 4};
  // clang-format on
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_true[i]));
  }
}

}  // namespace
}  // namespace tflite
