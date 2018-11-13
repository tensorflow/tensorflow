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
#include "tensorflow/contrib/lite/kernels/internal/test_util.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

#define ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#include "tensorflow/contrib/lite/kernels/internal/optimized/depthwiseconv_float.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/depthwiseconv_float.h"

namespace tflite {
namespace {

// Runs the DepthwiseConv and compares against the reference implementation.
template <FusedActivationFunctionType Ac>
void TestOneDepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride, int pad_width, int pad_height,
                          int depth_multiplier, const Dims<4>& output_dims) {
  const int output_buffer_size = RequiredBufferSizeForDims(output_dims);
  std::vector<float> output_data(output_buffer_size);
  std::vector<float> reference_output_data(output_buffer_size);
  reference_ops::DepthwiseConv<Ac>(input_data, input_dims, filter_data,
                                   filter_dims, bias_data, bias_dims, stride,
                                   pad_width, pad_height, depth_multiplier,
                                   reference_output_data.data(), output_dims);
  optimized_ops::DepthwiseConv<Ac>(input_data, input_dims, filter_data,
                                   filter_dims, bias_data, bias_dims, stride,
                                   pad_width, pad_height, depth_multiplier,
                                   output_data.data(), output_dims);
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

void TestOneDepthwiseConv(FusedActivationFunctionType Ac,
                          const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride, int pad_width, int pad_height,
                          int depth_multiplier, const Dims<4>& output_dims) {
#define TOCO_HANDLE_CASE(AC_TYPE)                                            \
  if (AC_TYPE == Ac) {                                                       \
    TestOneDepthwiseConv<AC_TYPE>(input_data, input_dims, filter_data,       \
                                  filter_dims, bias_data, bias_dims, stride, \
                                  pad_width, pad_height, depth_multiplier,   \
                                  output_dims);                              \
    return;                                                                  \
  }
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kNone)
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kRelu)
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kRelu1)
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kRelu6)
#undef TOCO_HANDLE_CASE
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
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
  const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int filter_width = ExponentialRandomPositiveInt(0.9f, 4, 10);
  const int filter_height = ExponentialRandomPositiveInt(0.9f, 4, 10);
  const int depth_multiplier = ExponentialRandomPositiveInt(0.8f, 6, 50);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const int output_depth = input_depth * depth_multiplier;
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
  const auto ac = RandomElement(std::vector<FusedActivationFunctionType>(
      {FusedActivationFunctionType::kNone, FusedActivationFunctionType::kRelu,
       FusedActivationFunctionType::kRelu6,
       FusedActivationFunctionType::kRelu1}));
  Dims<4> input_dims_inference =
      MakeDimsForInference(input_depth, input_width, input_height, batch);
  Dims<4> output_dims_inference;
  int pad_width, pad_height;
  const auto padding_type =
      UniformRandomInt(0, 1) ? PaddingType::kSame : PaddingType::kValid;
  if (!ComputeConvSizes(input_dims_inference, output_depth, filter_width,
                        filter_height, stride, padding_type,
                        &output_dims_inference, &pad_width, &pad_height)) {
    return false;
  }
  Dims<4> filter_dims_inference =
      MakeDimsForInference(output_depth, filter_width, filter_height, 1);
  Dims<4> bias_dims_inference = MakeDimsForInference(output_depth, 1, 1, 1);
  const int input_buffer_size = RequiredBufferSizeForDims(input_dims_inference);
  const int filter_buffer_size =
      RequiredBufferSizeForDims(filter_dims_inference);
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
  TestOneDepthwiseConv(ac, input_data.data(), input_dims_inference,
                       filter_data.data(), filter_dims_inference,
                       bias_data.data(), bias_dims_inference, stride, pad_width,
                       pad_height, depth_multiplier, output_dims_inference);
  return true;
}

void TestOneDepthwiseConv() {
  while (!TryTestOneDepthwiseConv()) {
  }
}

TEST(TestDepthwiseConv, TestDepthwiseConv) {
  const int kTestsToRun = 100 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv();
  }
}
}  // namespace
}  // namespace tflite
