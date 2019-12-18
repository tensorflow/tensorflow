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
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/softmax.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

void RunSoftmaxFloatReference(const uint8* input_data,
                              const RuntimeShape& shape_common,
                              int32 input_offset, const double input_scale,
                              int stride, float beta,
                              uint8* reference_output_data) {
  const int ref_buffer_size = shape_common.FlatSize();
  std::vector<float> reference_dequant_data(ref_buffer_size);
  std::vector<float> reference_output_float_data(ref_buffer_size);

  // Reference data generated via Dequant of input into float, and then applying
  // float Softmax.
  DequantizationParams dq_params;
  dq_params.zero_point = input_offset;
  dq_params.scale = input_scale;
  reference_ops::Dequantize(dq_params, shape_common, input_data, shape_common,
                            reference_dequant_data.data());
  SoftmaxParams sm_params;
  sm_params.beta = beta;
  optimized_ops::Softmax(sm_params, shape_common, reference_dequant_data.data(),
                         shape_common, reference_output_float_data.data());
  // Work with quantized scaling for Softmax, under which 256 represents 1, but
  // we limit this to 255.
  for (int i = 0; i < ref_buffer_size; i++) {
    reference_output_data[i] = std::min(
        255,
        static_cast<int>(std::round(256.0f * reference_output_float_data[i])));
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
    ASSERT_TRUE(std::abs(min_diff) == 0 && std::abs(max_diff) == 0);
  } else {
    // For small numbers of samples, the estimates of the means vary more.
    // Rather than widen the tolerances, we skip the smaller tests.
    ASSERT_TRUE(((std::abs(mean_diff) < 2e-2f && mean_abs_diff < 3e-2f) ||
                 buffer_size < 10000) &&
                std::abs(median_diff) == 0 && std::abs(min_diff) <= 1 &&
                std::abs(max_diff) <= 1);
  }
}

// Runs the Softmax and compares against the float reference implementation and
// the quantized reference implementation.
void RunOneSoftmaxTest(const uint8* input_data,
                       const RuntimeShape& shape_common, int32 input_offset,
                       const double input_scale, int stride, float beta) {
  const int buffer_size = shape_common.FlatSize();
  std::vector<uint8> optimized_softmax_output(buffer_size);
  std::vector<uint8> reference_float_softmax_output(buffer_size);
  std::vector<uint8> reference_quant_softmax_output(buffer_size);

  RunSoftmaxFloatReference(input_data, shape_common, input_offset, input_scale,
                           stride, beta, reference_float_softmax_output.data());

  int32 input_beta_multiplier;
  int input_beta_left_shift;
  static const int kScaledDiffIntegerBits = 5;
  tflite::PreprocessSoftmaxScaling(beta, input_scale, kScaledDiffIntegerBits,
                                   &input_beta_multiplier,
                                   &input_beta_left_shift);
  // diff_min has a negative value, and is used to limit the maximum magnitude
  // of the diffs, which are <= 0.
  const int diff_min = -tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                                     input_beta_left_shift);

  SoftmaxParams params;
  float table[256];
  params.input_multiplier = input_beta_multiplier;
  params.input_left_shift = input_beta_left_shift;
  params.diff_min = diff_min;
  params.scale = 1.0f / 256;
  params.zero_point = 0;
  params.table = table;
  optimized_ops::PopulateSoftmaxLookupTable(&params, input_scale, beta);
  optimized_ops::Softmax(params, shape_common, input_data, shape_common,
                         optimized_softmax_output.data());
  reference_ops::Softmax(params, shape_common, input_data, shape_common,
                         reference_quant_softmax_output.data());

  CheckOutputData<uint8_t>(optimized_softmax_output.data(),
                           reference_float_softmax_output.data(), shape_common,
                           "Optimized vs float reference", false);
  CheckOutputData<uint8_t>(optimized_softmax_output.data(),
                           reference_quant_softmax_output.data(), shape_common,
                           "Optimized vs quant reference", false);
  CheckOutputData<uint8_t>(reference_quant_softmax_output.data(),
                           reference_float_softmax_output.data(), shape_common,
                           "Quant reference vs float reference", false);
}

// This function picks some random Softmax params, which are checked for
// desirability.  If not acceptable, it returns false. If they're OK,
// it runs the Softmax test and returns true. This allows the caller
// to loop until a test has been run.
//
// Currently we do not reject for any reason.
bool TryOneUniformSoftmax() {
  // We pick mostly positive values, on the whole emphasizing smaller values and
  // therefore faster tests.  We test a wider range of depths.  In the case of
  // Softmax, the width and height really just create test repetitions.
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = ExponentialRandomPositiveInt(0.75f, 175, 500);
  const int input_width = ExponentialRandomPositiveInt(0.8f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.8f, 20, 200);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const double input_scale = std::pow(10.0, UniformRandomFloat(-2.0, 1.0));
  const int32 input_offset = UniformRandomInt(-256, 0);
  const float beta = 1.0f + ExponentialRandomPositiveFloat(0.9f, 2, 10);

  auto shape_common =
      RuntimeShape({batch, input_height, input_width, input_depth});
  const int buffer_size = shape_common.FlatSize();

  std::vector<uint8> input_data(buffer_size);
  FillRandom(&input_data);
  RunOneSoftmaxTest(input_data.data(), shape_common, input_offset, input_scale,
                    stride, beta);
  return true;
}

// Runs the Int8 quatnized Softmax and compares reference implementation with
// optimized implementation.
void RunOneSoftmaxTestInt8(const int8* input_data,
                           const RuntimeShape& shape_common, int32 input_offset,
                           const double input_scale, int stride, float beta) {
  const int buffer_size = shape_common.FlatSize();
  std::vector<int8> optimized_quant_softmax_output(buffer_size);
  std::vector<int8> reference_quant_softmax_output(buffer_size);

  int32 input_beta_multiplier;
  int input_beta_left_shift;
  static const int kScaledDiffIntegerBits = 5;
  tflite::PreprocessSoftmaxScaling(beta, input_scale, kScaledDiffIntegerBits,
                                   &input_beta_multiplier,
                                   &input_beta_left_shift);
  // diff_min has a negative value, and is used to limit the maximum magnitude
  // of the diffs, which are <= 0.
  const int diff_min = -tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                                     input_beta_left_shift);

  SoftmaxParams params;
  params.input_multiplier = input_beta_multiplier;
  params.input_left_shift = input_beta_left_shift;
  params.diff_min = diff_min;
  optimized_integer_ops::Softmax(params, shape_common, input_data, shape_common,
                                 optimized_quant_softmax_output.data());
  reference_integer_ops::Softmax(params, shape_common, input_data, shape_common,
                                 reference_quant_softmax_output.data());

  CheckOutputData<int8_t>(optimized_quant_softmax_output.data(),
                          reference_quant_softmax_output.data(), shape_common,
                          "Int8 quant refernece vs optimized", true);
}

// This function picks some random Softmax params, which are checked for
// desirability.  If not acceptable, it returns false. If they're OK,
// it runs the Softmax and test the results between reference int8 and optimized
// int8 kernels.
bool TryOneUniformSoftmaxInt8() {
  // We pick mostly positive values, on the whole emphasizing smaller values and
  // therefore faster tests.  We test a wider range of depths.  In the case of
  // Softmax, the width and height really just create test repetitions.
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = ExponentialRandomPositiveInt(0.75f, 175, 500);
  const int input_width = ExponentialRandomPositiveInt(0.8f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.8f, 20, 200);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const double input_scale = std::pow(10.0, UniformRandomFloat(-2.0, 1.0));
  const int32 input_offset = UniformRandomInt(-128, 127);
  const float beta = 1.0f + ExponentialRandomPositiveFloat(0.9f, 2, 10);

  auto shape_common =
      RuntimeShape({batch, input_height, input_width, input_depth});
  const int buffer_size = shape_common.FlatSize();

  std::vector<int8> input_data(buffer_size);
  FillRandom(&input_data);
  RunOneSoftmaxTestInt8(input_data.data(), shape_common, input_offset,
                        input_scale, stride, beta);
  return true;
}

// See TryOneUniformSoftmax() for a general description.
//
// Tests with "skyscraper" input patterns are included for two reasons. (a)
// Bimodal distributions are potentially challenging and perhaps more
// realistic than simple uniform random inputs.  (b) Some implementations of
// Softmax may adapt as they traverse the depth, and so we test handling of
// cases where relatively small values are encountered at the beginning and end.
bool TryOneSkyscraperSoftmax(bool small_depth) {
  // We pick mostly positive values, on the whole emphasizing smaller values and
  // therefore faster tests.  We test a wider range of depths.  In the case of
  // Softmax, the width and height really just create test repetitions.
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = small_depth
                              ? ExponentialRandomPositiveInt(0.75f, 40, 500)
                              : ExponentialRandomPositiveInt(0.75f, 175, 500);
  const int input_width = ExponentialRandomPositiveInt(0.7f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.7f, 20, 200);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const double input_scale = std::pow(10.0, UniformRandomFloat(-2.0, 1.0));
  const int32 input_offset = UniformRandomInt(-256, 0);
  const float beta = 1.0f + ExponentialRandomPositiveFloat(0.9f, 2, 10);
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
  RunOneSoftmaxTest(input_data.data(), shape_common, input_offset, input_scale,
                    stride, beta);
  return true;
}

TEST(TestQuantizedSoftmax, UniformSoftmaxTests) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneUniformSoftmax()) {
    }
  }
}

TEST(TestQuantizedSoftmax, UniformSoftmaxTestsInt8) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneUniformSoftmaxInt8()) {
    }
  }
}

TEST(TestQuantizedSoftmax, SkyscraperSoftmaxTests) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneSkyscraperSoftmax(false)) {
    }
  }
}

TEST(TestQuantizedSoftmax, SmallSkyscraperSoftmaxTests) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    while (!TryOneSkyscraperSoftmax(true)) {
    }
  }
}
}  // namespace
}  // namespace tflite
