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
#include "tensorflow/lite/kernels/internal/optimized/resize_bilinear.h"

#include <algorithm>
#include <cmath>
#include <list>
#include <type_traits>
#include <typeinfo>
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

  // For typical integers, use full range. Clip to moderate range for floating.
  const T min_amplitude = static_cast<T>(
      std::max(-32768.0, static_cast<double>(std::numeric_limits<T>::min())));
  const T max_amplitude = static_cast<T>(
      std::min(65535.0, static_cast<double>(std::numeric_limits<T>::max())));
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

  bool strict_match = false;
  if (std::is_same<T, uint8>::value && ((depth % 8) == 0) &&
      ((input_width * 8) == output_width) &&
      ((input_height * 8) == output_height)) {
    strict_match = true;
  }

  double sum_diff = 0;
  float max_abs_val = 0;
  for (int i = 0; i < output_buffer_size; i++) {
    sum_diff += std::abs(static_cast<float>(output_data[i]) -
                         static_cast<float>(reference_output_data[i]));
    max_abs_val = std::max(
        max_abs_val, std::abs(static_cast<float>(reference_output_data[i])));
  }

  if (strict_match) {
    if (sum_diff > 0) {
      ASSERT_EQ(sum_diff, 0);
    }
  } else {
    if (sum_diff != 0.f) {
      const float mean_diff = static_cast<float>(sum_diff / output_buffer_size);
      const float relative_error = std::abs(mean_diff) / max_abs_val;
      ASSERT_LT(relative_error, error_threshold);
    }
  }
}

class ResizeBilinearImplTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<tflite::ResizeBilinearParams> {};

TEST_P(ResizeBilinearImplTest, TestResizeBilinearUint8) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 500;
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

TEST_P(ResizeBilinearImplTest, TestResizeBilinearUint8_2x2) {
  RandomEngine().seed(96743);
  const int kTestsToRun = 500;
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
      error_threshold = 1e-3;
    }
    TestOneResizeBilinear<uint8>(op_params, batch, depth, input_width,
                                 input_height, output_width, output_height,
                                 error_threshold);
  }
}

TEST_P(ResizeBilinearImplTest, TestResizeBilinearFloat) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 500;
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
      error_threshold = 1e-3;
    }
    TestOneResizeBilinear<float>(op_params, batch, depth, input_width,
                                 input_height, output_width, output_height,
                                 error_threshold);
  }
}

TEST_P(ResizeBilinearImplTest, TestResizeBilinearFloat_2x2) {
  RandomEngine().seed(38291);
  const int kTestsToRun = 500;
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
      error_threshold = 1e-3;
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

TEST(ResizeBilinear, TestResizeBilinearHalfPixelCentersFloat_3x3to2x2) {
  // Input: 3x3
  RuntimeShape input_dims_inference({1, 3, 3, 1});
  std::vector<float> input_data = {1, 2, 3,  //
                                   4, 5, 6,  //
                                   7, 8, 9};

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
  std::vector<float> reference_half_pixel_centers_false = {1, 2.5,  //
                                                           5.5, 7};
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_false[i]));
  }

  // Test with half_pixel_centers = true.
  op_params.half_pixel_centers = true;
  reference_ops::ResizeBilinear(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  std::vector<float> reference_half_pixel_centers_true = {2, 3.5,  //
                                                          6.5, 8};
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_true[i]));
  }
}

TEST(ResizeBilinear, TestResizeBilinearHalfPixelCentersFloat_2x2to4x4) {
  // Input: 2x2
  RuntimeShape input_dims_inference({1, 2, 2, 1});
  std::vector<float> input_data = {1, 2,  //
                                   3, 4};

  // Output: 4x4
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
  std::vector<float> reference_half_pixel_centers_false = {1, 1.5, 2, 2,  //
                                                           2, 2.5, 3, 3,  //
                                                           3, 3.5, 4, 4,  //
                                                           3, 3.5, 4, 4};
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_false[i]));
  }

  // Test with half_pixel_centers = true.
  op_params.half_pixel_centers = true;
  reference_ops::ResizeBilinear(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  std::vector<float> reference_half_pixel_centers_true = {
      1,   1.25, 1.75, 2,    //
      1.5, 1.75, 2.25, 2.5,  //
      2.5, 2.75, 3.25, 3.5,  //
      3,   3.25, 3.75, 4};
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<float>(output_data[i]),
              static_cast<float>(reference_half_pixel_centers_true[i]));
  }
}

template <typename T>
void TestResizeBilinearHalfPixelCenters_2x2to4x6() {
  // Input: 2x2
  RuntimeShape input_dims_inference({1, 2, 2, 1});
  // clang-format off
  std::vector<T> input_data = {127, -128, 64, 0};
  // clang-format on

  // Output: 4x6
  RuntimeShape output_dims_inference({1, 4, 6, 1});
  // Initialize the output data with something other than zero, so we can catch
  // issue with kernels failing to initialize the output.
  const int output_buffer_size = output_dims_inference.FlatSize();
  std::vector<T> output_data(output_buffer_size, 3);

  RuntimeShape output_size_dims({1, 1, 1, 2});
  std::vector<int32> output_size_data = {4, 6};

  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = false;
  op_params.half_pixel_centers = false;

  // Test with half_pixel_centers = false.
  reference_ops::ResizeBilinearInteger(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  // clang-format off
  std::vector<T> reference_half_pixel_centers_false =
      {  127,   42,  -43, -128,  -128, -128,
          96,   42,  -11,  -64,   -64,  -64,
          64,   43,   21,    0,     0,    0,
          64,   43,   21,    0,     0,    0};
  // Float results =
  // {127.000000, 41.999996, -43.000004, -128.000000, -128.000000, -128.000000,
  //   95.500000, 42.333328, -10.833336,  -64.000000,  -64.000000,  -64.000000,
  //   64.000000, 42.666664,  21.333332,    0.000000,    0.000000,    0.000000,
  //   64.000000, 42.666664,  21.333332,    0.000000,    0.000000,    0.000000};

  // clang-format on
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<T>(output_data[i]),
              static_cast<T>(reference_half_pixel_centers_false[i]));
  }

  // Test with half_pixel_centers = true.
  op_params.half_pixel_centers = true;
  reference_ops::ResizeBilinearInteger(
      op_params, input_dims_inference, input_data.data(), output_size_dims,
      output_size_data.data(), output_dims_inference, output_data.data());
  // clang-format off
  std::vector<T> reference_half_pixel_centers_true =
      {  127,  127,   42,  -43, -128, -128,
         111,  111,   42,  -27,  -96,  -96,
          80,   80,   43,    5,  -32,  -32,
          64,   64,   43,   21,    0,    0};
  // Float result =
  // {127.000000, 127.000000, 41.999992, -43.000023, -128.000000, -128.000000,
  //  111.249992, 111.250000, 42.166660, -26.916683,  -96.000000,  -96.000000,
  //   79.749992,  79.750000, 42.499996,   5.249992,  -32.000000,  -32.000000,
  //   63.999996,  64.000000, 42.666664,  21.333328,    0.000000,    0.000000};

  // clang-format on
  for (int i = 0; i < output_buffer_size; i++) {
    EXPECT_EQ(static_cast<T>(output_data[i]),
              static_cast<T>(reference_half_pixel_centers_true[i]));
  }
}

TEST(ResizeBilinear, TestResizeBilinearHalfPixelCenters_2x2to4x6_Int8) {
  TestResizeBilinearHalfPixelCenters_2x2to4x6<int8_t>();
}

TEST(ResizeBilinear, TestResizeBilinearHalfPixelCenters_2x2to4x6_Int16) {
  TestResizeBilinearHalfPixelCenters_2x2to4x6<int16_t>();
}

class ResizeBilinearImplX8ChannelTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<tflite::ResizeBilinearParams> {};

// Test when channel count is multiple of 8, and scaling is 2, 4, 8, same in
// both directions.
//
// Version for uint8.
TEST_P(ResizeBilinearImplX8ChannelTest, TestResizeBilinearX8ChannelUint8) {
  RandomEngine().seed(85935);
  const int kTestsToRun = 500;
  const tflite::ResizeBilinearParams op_params = GetParam();

  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = UniformRandomInt(1, 2);
    const int depth = ExponentialRandomPositiveInt(0.4f, 1, 6) * 8;
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 100);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 100);
    const int scale_factor = 1 << UniformRandomInt(1, 3);  // 2, 4, 8;
    const int output_width = input_width * scale_factor;
    const int output_height = input_height * scale_factor;

    TestOneResizeBilinear<uint8>(op_params, batch, depth, input_width,
                                 input_height, output_width, output_height,
                                 0.025);
  }
}

// Test when channel count is multiple of 8, and scaling is 2, 4, 8, same in
// both directions.
//
// Version for int8.
TEST_P(ResizeBilinearImplX8ChannelTest, TestResizeBilinearX8ChannelInt8) {
  RandomEngine().seed(27496);
  const int kTestsToRun = 500;
  const tflite::ResizeBilinearParams op_params = GetParam();

  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = UniformRandomInt(1, 2);
    const int depth = ExponentialRandomPositiveInt(0.4f, 1, 6) * 8;
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 100);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 100);
    const int scale_factor = 1 << UniformRandomInt(1, 3);  // 2, 4, 8;
    const int output_width = input_width * scale_factor;
    const int output_height = input_height * scale_factor;

    TestOneResizeBilinear<int8>(op_params, batch, depth, input_width,
                                input_height, output_width, output_height,
                                0.025);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ResizeBilinear, ResizeBilinearImplX8ChannelTest,
    ::testing::ValuesIn(std::list<tflite::ResizeBilinearParams>({
        // For present purposes we do not test align_corners=true, because
        // that configuration is becoming less popular and so we are likely not
        // to optimize for it.
        {/**align_corners**/ false, /**half_pixel_centers**/ true},
    })));

}  // namespace
}  // namespace tflite
