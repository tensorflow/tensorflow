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
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

#define ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"

namespace tflite {
namespace {

// Runs the DepthwiseConv and compares against the reference implementation.
void TestOneDepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape) {
  const int output_buffer_size = output_shape.FlatSize();
  std::vector<float> output_data(output_buffer_size);
  std::vector<float> reference_output_data(output_buffer_size);
  reference_ops::DepthwiseConv(params, input_shape, input_data, filter_shape,
                               filter_data, bias_shape, bias_data, output_shape,
                               reference_output_data.data());
  optimized_ops::DepthwiseConvImpl(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data.data(), CpuFlags(),
      /*thread_start=*/0,
      /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);

  double sum_abs_diff = 0;
  float max_abs_val = 0;
  for (int i = 0; i < output_buffer_size; i++) {
    sum_abs_diff += std::abs(output_data[i] - reference_output_data[i]);
    max_abs_val = std::max(max_abs_val, std::abs(reference_output_data[i]));
  }
  if (sum_abs_diff != 0.f) {
    const float mean_diff =
        static_cast<float>(sum_abs_diff / output_buffer_size);
    const float relative_error = std::abs(mean_diff) / max_abs_val;
    ASSERT_LT(relative_error, 1e-5f);
  }
}

// This function picks some random DepthwiseConv params, which may or may not
// be legal. If they're not legal, it returns false. If they're legal,
// it runs the DepthwiseConv test and returns true. This allows the caller
// to loop until a test has been run.
bool TryTestOneDepthwiseConv() {
  // We have to pick a lot of positive values, where we are particularly
  // interested in small values because they are most likely to be special
  // cases in optimized implementations, and secondarily because they allow
  // tests to run fast, which means we can run more tests and get more
  // coverage.
  const int batch = UniformRandomInt(1, 2);
  const int input_depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
  const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int filter_width = ExponentialRandomPositiveInt(0.9f, 4, 10);
  const int filter_height = ExponentialRandomPositiveInt(0.9f, 4, 10);
  const int depth_multiplier = ExponentialRandomPositiveInt(0.8f, 6, 50);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const int output_depth = input_depth * depth_multiplier;
  const int dilation_width_factor = RandomElement(std::vector<int>({1, 2, 4}));
  const int dilation_height_factor = RandomElement(std::vector<int>({1, 2, 4}));
  float output_activation_min, output_activation_max;
  FusedActivationFunctionType ac =
      RandomElement(std::vector<FusedActivationFunctionType>(
          {FusedActivationFunctionType::kNone,
           FusedActivationFunctionType::kRelu,
           FusedActivationFunctionType::kRelu1,
           FusedActivationFunctionType::kRelu6}));
  GetActivationMinMax(ac, &output_activation_min, &output_activation_max);
  // The optimized DepthwiseConv implementation currently uses a fixed-size
  // accumulator buffer on the stack, with that size. This currently means
  // that it does not support larger output depths. It CHECK's for it,
  // so it's safe in the sense that if a larger output depth was encountered,
  // it would explicitly fail. We just need to adjust our testing to that
  // constraint.
  const int kMaxSupportedOutputDepth = 1024;
  if (output_depth > kMaxSupportedOutputDepth) {
    return false;
  }
  RuntimeShape input_shape_inference(
      {batch, input_height, input_width, input_depth});
  RuntimeShape output_shape_inference;
  int pad_width, pad_height;
  const auto padding_type =
      UniformRandomInt(0, 1) ? PaddingType::kSame : PaddingType::kValid;
  if (!ComputeConvSizes(input_shape_inference, output_depth, filter_width,
                        filter_height, stride, dilation_width_factor,
                        dilation_height_factor, padding_type,
                        &output_shape_inference, &pad_width, &pad_height)) {
    return false;
  }
  RuntimeShape filter_shape_inference(
      {1, filter_height, filter_width, output_depth});
  RuntimeShape bias_shape_inference({1, 1, 1, output_depth});
  const int input_buffer_size = input_shape_inference.FlatSize();
  const int filter_buffer_size = filter_shape_inference.FlatSize();
  std::vector<float> input_data(input_buffer_size);
  std::vector<float> filter_data(filter_buffer_size);
  std::vector<float> bias_data(output_depth);
  const float input_amplitude = 1.f;
  const float filter_amplitude = 1.f;
  const float bias_amplitude =
      filter_width * filter_height * input_amplitude * filter_amplitude;
  FillRandom(&input_data, -input_amplitude, input_amplitude);
  FillRandom(&filter_data, -filter_amplitude, filter_amplitude);
  FillRandom(&bias_data, -bias_amplitude, bias_amplitude);
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride;
  op_params.stride_height = stride;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.depth_multiplier = depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  TestOneDepthwiseConv(op_params, input_shape_inference, input_data.data(),
                       filter_shape_inference, filter_data.data(),
                       bias_shape_inference, bias_data.data(),
                       output_shape_inference);
  return true;
}

void TestOneDepthwiseConv() {
  while (!TryTestOneDepthwiseConv()) {
  }
}

TEST(TestDepthwiseConv, TestDepthwiseConv) {
  const int kTestsToRun = 10 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv();
  }
}
}  // namespace
}  // namespace tflite
