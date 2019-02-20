/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/log_softmax.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/string.h"

namespace tflite {
namespace {

void RunLogSoftmaxFloatReference(const uint8* input_data,
                                 const RuntimeShape& shape_common,
                                 int32 input_offset, const double input_scale,
                                 int stride, float beta,
                                 uint8* reference_output_data) {
  const int ref_buffer_size = shape_common.FlatSize();
  std::vector<float> reference_dequant_data(ref_buffer_size);
  std::vector<float> reference_output_float_data(ref_buffer_size);

  // Reference data generated via Dequant of input into float, and then applying
  // float LogSoftmax.
  DequantizationParams dq_params;
  dq_params.zero_point = input_offset;
  dq_params.scale = input_scale;
  reference_ops::Dequantize(dq_params, shape_common, input_data, shape_common,
                            reference_dequant_data.data());
  SoftmaxParams sm_params;
  optimized_ops::LogSoftmax(sm_params, shape_common,
                            reference_dequant_data.data(), shape_common,
                            reference_output_float_data.data());
  // Work with quantized scaling for LogSoftmax, under which 255 represents 0,
  // and -16 gets nudged up to 0.
  for (int i = 0; i < ref_buffer_size; i++) {
    reference_output_data[i] = std::max(
        0, static_cast<int>(
               255 + std::round(16.0f * reference_output_float_data[i])));
  }
}

// Same as above except for the following change:
// - input and output data type
// - Dequnatize function
// - clamping values
void RunLogSoftmaxFloatReference(const int8* input_data,
                                 const RuntimeShape& shape_common,
                                 int32 input_offset, const double input_scale,
                                 int stride, float beta,
                                 int8* reference_output_data) {
  const int ref_buffer_size = shape_common.FlatSize();
  std::vector<float> reference_dequant_data(ref_buffer_size);
  std::vector<float> reference_output_float_data(ref_buffer_size);

  // Reference data generated via Dequant of input into float, and then applying
  // float LogSoftmax.
  DequantizationParams dq_params;
  dq_params.zero_point = input_offset;
  dq_params.scale = input_scale;
  reference_integer_ops::Dequantize(dq_params, shape_common, input_data,
                                    shape_common,
                                    reference_dequant_data.data());
  SoftmaxParams sm_params;
  optimized_ops::LogSoftmax(sm_params, shape_common,
                            reference_dequant_data.data(), shape_common,
                            reference_output_float_data.data());
  // Work with quantized scaling for LogSoftmax, under which 255 represents 0,
  // and -16 gets nudged up to 0.
  for (int i = 0; i < ref_buffer_size; i++) {
    reference_output_data[i] = std::max(
        -128, static_cast<int>(
                  127 + std::round(16.0f * reference_output_float_data[i])));
  }
}

template <typename T>
void CheckOutputData(const T* test_output, const T* reference_output,
                     const RuntimeShape& shape_common,
                     const string& check_label, bool be_exacting) {
  const int buffer_size = shape_common.FlatSize();
  // While calculating some metrics in floating point, we work with quantized
  // scaling.
  std::vector<int> diff(buffer_size);
  int64_t sum_diff = 0;
  int64_t sum_abs_diff = 0;
  for (int i = 0; i < buffer_size; i++) {
    diff[i] = static_cast<int>(test_output[i]) - reference_output[i];
    sum_diff += diff[i];
    sum_abs_diff += std::abs(diff[i]);
  }
  // These stats help understand test failures.
  std::sort(std::begin(diff), std::end(diff));
  const int min_diff = diff.front();
  const int max_diff = diff.back();
  const int median_diff = diff[diff.size() / 2];
  const float mean_diff = static_cast<float>(sum_diff) / buffer_size;
  const float mean_abs_diff = static_cast<float>(sum_abs_diff) / buffer_size;
  // We either check for bit exactness (against the reference quantized version)
  // or for general accuracy, allowing off-by-one (against the float reference).
  if (be_exacting) {
    ASSERT_TRUE(std::abs(min_diff) == 0 && std::abs(max_diff) == 0)
        << check_label << ": "
        << "std::abs(min_diff)=" << std::abs(min_diff)
        << ", std::abs(max_diff)=" << std::abs(max_diff);
  } else {
    // For small numbers of samples, the estimates of the means vary more.
    // Rather than widen the tolerances, we skip the smaller tests.
    ASSERT_TRUE(((std::abs(mean_diff) < 2e-2f && mean_abs_diff < 3e-2f) ||
                 buffer_size < 10000) &&
                std::abs(median_diff) == 0 && std::abs(min_diff) <= 1 &&
                std::abs(max_diff) <= 1)
        << check_label << ": "
        << "buffer_size=" << buffer_size << ", mean_diff=" << mean_diff
        << ", mean_abs_diff=" << mean_abs_diff
        << ", median_diff=" << median_diff << ", min_diff=" << min_diff
        << ", max_diff=" << max_diff;
  }
}

// Runs the LogSoftmax and compares against the float reference implementation
// and the quantized reference implementation.
void RunOneLogSoftmaxTest(const uint8* input_data,
                          const RuntimeShape& shape_common, int32 input_offset,
                          const double input_scale, int stride, float beta) {
  const int buffer_size = shape_common.FlatSize();
  std::vector<uint8> optimized_logsoftmax_output(buffer_size);
  std::vector<uint8> reference_float_logsoftmax_output(buffer_size);
  std::vector<uint8> reference_quant_logsoftmax_output(buffer_size);

  RunLogSoftmaxFloatReference(input_data, shape_common, input_offset,
                              input_scale, stride, beta,
                              reference_float_logsoftmax_output.data());

  int32 input_beta_multiplier;
  int input_beta_left_shift;
  int32 reverse_scaling_divisor;
  int reverse_scaling_right_shift;
  static const int kScaledDiffIntegerBits = 5;
  tflite::PreprocessLogSoftmaxScalingExp(
      beta, input_scale, kScaledDiffIntegerBits, &input_beta_multiplier,
      &input_beta_left_shift, &reverse_scaling_divisor,
      &reverse_scaling_right_shift);
  reverse_scaling_right_shift *= -1;
  // diff_min has a negative value, and is used to limit the maximum magnitude
  // of the diffs, which are <= 0.
  const int diff_min = -tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                                     input_beta_left_shift);

  SoftmaxParams params;
  params.input_multiplier = input_beta_multiplier;
  params.input_left_shift = input_beta_left_shift;
  params.reverse_scaling_divisor = reverse_scaling_divisor;
  params.reverse_scaling_right_shift = reverse_scaling_right_shift;
  params.diff_min = diff_min;
  optimized_ops::LogSoftmax(params, shape_common, input_data, shape_common,
                            optimized_logsoftmax_output.data());
  reference_ops::LogSoftmax(params, shape_common, input_data, shape_common,
                            reference_quant_logsoftmax_output.data());

  CheckOutputData<uint8_t>(optimized_logsoftmax_output.data(),
                           reference_float_logsoftmax_output.data(),
                           shape_common, "Optimized vs float reference", false);
  CheckOutputData<uint8_t>(optimized_logsoftmax_output.data(),
                           reference_quant_logsoftmax_output.data(),
                           shape_common, "Optimized vs quant reference", true);
  CheckOutputData<uint8_t>(reference_quant_logsoftmax_output.data(),
                           reference_float_logsoftmax_output.data(),
                           shape_common, "Quant reference vs float reference",
                           false);
}

// Runs the LogSoftmax and compares against the float reference implementation
// and the int8 quantized reference implementation.
void RunOneLogSoftmaxTest(const int8* input_data,
                          const RuntimeShape& shape_common, int32 input_offset,
                          const double input_scale, int stride, float beta) {
  const int buffer_size = shape_common.FlatSize();
  std::vector<int8> quantized_logsoftmax_reference_implementation(buffer_size);
  std::vector<int8> float_logsoftmax_optimized_implementation(buffer_size);

  RunLogSoftmaxFloatReference(input_data, shape_common, input_offset,
                              input_scale, stride, beta,
                              float_logsoftmax_optimized_implementation.data());

  int32 input_beta_multiplier;
  int input_beta_left_shift;
  int32 reverse_scaling_divisor;
  int reverse_scaling_right_shift;
  static const int kScaledDiffIntegerBits = 5;
  tflite::PreprocessLogSoftmaxScalingExp(
      beta, input_scale, kScaledDiffIntegerBits, &input_beta_multiplier,
      &input_beta_left_shift, &reverse_scaling_divisor,
      &reverse_scaling_right_shift);
  reverse_scaling_right_shift *= -1;
  // diff_min has a negative value, and is used to limit the maximum magnitude
  // of the diffs, which are <= 0.
  const int diff_min = -tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                                     input_beta_left_shift);

  const int outer_size =
      shape_common.Dims(0) * shape_common.Dims(1) * shape_common.Dims(2);
  const int inner_size = shape_common.Dims(3);
  reference_integer_ops::LogSoftmax(
      input_beta_multiplier, input_beta_left_shift, reverse_scaling_divisor,
      reverse_scaling_right_shift, diff_min, outer_size, inner_size, input_data,
      quantized_logsoftmax_reference_implementation.data());

  CheckOutputData<int8_t>(quantized_logsoftmax_reference_implementation.data(),
                          float_logsoftmax_optimized_implementation.data(),
                          shape_common, "Quant reference vs float reference",
                          false);
}

// This function picks some random LogSoftmax params, which are checked for
// desirability.  If not acceptable, it returns false. If they're OK,
// it runs the LogSoftmax test and returns true. This allows the caller
// to loop until a test has been run.
//
// Currently we do not reject for any reason.
template <typename T>
bool TryOneUniformLogSoftmax() {
  // We pick mostly positive values, on the whole emphasizing smaller values and
  // therefore faster tests.  We test a wider range of depths.  In the case of
  // LogSoftmax, the width and height really just create test repetitions.
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = ExponentialRandomPositiveInt(0.75f, 175, 500);
  const int input_width = ExponentialRandomPositiveInt(0.8f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.8f, 20, 200);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const double input_scale = std::pow(10.0, UniformRandomFloat(-2.0, 1.0));
  const int32 input_offset = UniformRandomInt(-256, 0);
  static constexpr float beta = 1.0f;

  auto shape_common =
      RuntimeShape({batch, input_height, input_width, input_depth});
  const int buffer_size = shape_common.FlatSize();

  std::vector<T> input_data(buffer_size);
  FillRandom(&input_data);
  RunOneLogSoftmaxTest(input_data.data(), shape_common, input_offset,
                       input_scale, stride, beta);
  return true;
}

// See TryOneUniformLogSoftmax() for a general description.
//
// Tests with "skyscraper" input patterns are included for two reasons. (a)
// Bimodal distributions are potentially challenging and perhaps more
// realistic than simple uniform random inputs.  (b) Some implementations of
// LogSoftmax may adapt as they traverse the depth, and so we test handling of
// cases where relatively small values are encountered at the beginning and end.
bool TryOneSkyscraperLogSoftmax(bool small_depth) {
  // We pick mostly positive values, on the whole emphasizing smaller values and
  // therefore faster tests.  We test a wider range of depths.  In the case of
  // LogSoftmax, the width and height really just create test repetitions.
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = small_depth
                              ? ExponentialRandomPositiveInt(0.75f, 40, 500)
                              : ExponentialRandomPositiveInt(0.75f, 175, 500);
  const int input_width = ExponentialRandomPositiveInt(0.7f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.7f, 20, 200);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const double input_scale = std::pow(10.0, UniformRandomFloat(-2.0, 1.0));
  const int32 input_offset = UniformRandomInt(-256, 0);
  static constexpr float beta = 1.0f;
  // Extra parameters for skyscraper input patterns.
  const double middle_proportion =
      ExponentialRandomPositiveFloat(0.65f, 0.1, 1.0);
  const int middle_min = UniformRandomInt(0, 255);
  const int sides_max = UniformRandomInt(0, middle_min);

  auto shape_common =
      RuntimeShape({batch, input_height, input_width, input_depth});
  const int buffer_size = shape_common.FlatSize();

  std::vector<uint8> input_data(buffer_size);
  FillRandomSkyscraper(&input_data, input_depth, middle_proportion, middle_min,
                       sides_max);
  RunOneLogSoftmaxTest(input_data.data(), shape_common, input_offset,
                       input_scale, stride, beta);
  return true;
}

TEST(TestQuantizedLogSoftmax, UniformLogSoftmaxUint8Tests) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneUniformLogSoftmax<uint8_t>()) {
    }
  }
}

TEST(TestQuantizedLogSoftmax, UniformLogSoftmaxUint8Int8Tests) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneUniformLogSoftmax<int8_t>()) {
    }
  }
}

TEST(TestQuantizedLogSoftmax, SkyscraperLogSoftmaxUint8Tests) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneSkyscraperLogSoftmax(false)) {
    }
  }
}

TEST(TestQuantizedLogSoftmax, SmallSkyscraperLogSoftmaxUint8Tests) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneSkyscraperLogSoftmax(true)) {
    }
  }
}
}  // namespace
}  // namespace tflite
