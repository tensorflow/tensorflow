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
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/kernels/internal/test_util.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

#define ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#include "tensorflow/contrib/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/depthwiseconv_uint8.h"

namespace tflite {
namespace {

// Runs the DepthwiseConv and compares against the reference implementation.
template <FusedActivationFunctionType Ac>
int TestOneDepthwiseConvWithGivenOutputShift(
    const std::uint8_t* input_data, const Dims<4>& input_dims,
    std::int32_t input_offset, const std::uint8_t* filter_data,
    const Dims<4>& filter_dims, std::int32_t filter_offset,
    const std::int32_t* bias_data, const Dims<4>& bias_dims, int stride,
    int pad_width, int pad_height, int depth_multiplier,
    std::int32_t output_offset, std::int32_t output_multiplier,
    int output_shift, std::int32_t output_activation_min,
    std::int32_t output_activation_max, const Dims<4>& output_dims) {
  const int output_buffer_size = RequiredBufferSizeForDims(output_dims);
  std::vector<std::uint8_t> output_data(output_buffer_size);
  std::vector<std::uint8_t> reference_output_data(output_buffer_size);
  reference_ops::DepthwiseConv<Ac>(
      input_data, input_dims, input_offset, filter_data, filter_dims,
      filter_offset, bias_data, bias_dims, stride, pad_width, pad_height,
      depth_multiplier, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max,
      reference_output_data.data(), output_dims);
  optimized_ops::DepthwiseConv<Ac>(
      input_data, input_dims, input_offset, filter_data, filter_dims,
      filter_offset, bias_data, bias_dims, stride, pad_width, pad_height,
      depth_multiplier, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max, output_data.data(),
      output_dims);
  int saturated_min = 0;
  int saturated_max = 0;
  std::vector<int> diff(output_buffer_size);
  std::int64_t sum_diff = 0;
  std::int64_t sum_abs_diff = 0;
  for (int i = 0; i < output_buffer_size; i++) {
    diff[i] = static_cast<int>(output_data[i]) -
              static_cast<int>(reference_output_data[i]);
    sum_diff += diff[i];
    sum_abs_diff += std::abs(diff[i]);
    saturated_min += output_data[i] == output_activation_min;
    saturated_max += output_data[i] == output_activation_max;
  }
  // These stats help understand test failures.
  std::sort(std::begin(diff), std::end(diff));
  const int min_diff = diff.front();
  const int max_diff = diff.back();
  const int median_diff = diff[diff.size() / 2];
  const float mean_diff = static_cast<float>(sum_diff) / output_buffer_size;
  const float mean_abs_diff =
      static_cast<float>(sum_abs_diff) / output_buffer_size;
  // Normally we should require bit-for-bit exact results. Unfortunately a bug
  // in the Intel arm_neon_sse.h translation header that we use for x86 tests
  // causes 1-bit inaccuracy in
  // the vqrdmulh_n_s32 intrinsic, which causes off-by-1 errors in quantized
  // DepthwiseConv ops. So we have to live with a few off-by-one errors for now,
  // yet still ensure that no more than a small minority of values are wrong.
  EXPECT_TRUE(std::abs(mean_diff) < 1e-5f && mean_abs_diff < 1e-5f &&
              std::abs(median_diff) == 0 && std::abs(min_diff) <= 1 &&
              std::abs(max_diff) <= 1);
  if (saturated_min > 2 * saturated_max) {
    return -1;
  }
  if (saturated_max > 2 * saturated_min) {
    return 1;
  }
  return 0;
}

// The point of this function is that we can't practically know which
// output_shift value to pass to test DepthwiseConv. It's not easy to guess (we
// could do some
// statistics for large size, but they would be fragile at smaller sizes), and
// guessing wrong would mean that all the values get saturated so the test
// becomes
// vacuous. So we just bisect our way to reasonable output_shift values.
template <FusedActivationFunctionType Ac>
void TestOneDepthwiseConvBisectOutputShift(
    const std::uint8_t* input_data, const Dims<4>& input_dims,
    std::int32_t input_offset, const std::uint8_t* filter_data,
    const Dims<4>& filter_dims, std::int32_t filter_offset,
    const std::int32_t* bias_data, const Dims<4>& bias_dims, int stride,
    int pad_width, int pad_height, int depth_multiplier,
    std::int32_t output_offset, std::int32_t output_multiplier,
    int output_activation_bisect_start, int output_activation_bisect_end,
    std::int32_t output_activation_min, std::int32_t output_activation_max,
    const Dims<4>& output_dims) {
  ASSERT_LT(output_activation_bisect_start, output_activation_bisect_end)
      << "Bisection failed ?!?!";
  int output_shift_bisect_midpoint =
      (output_activation_bisect_start + output_activation_bisect_end) / 2;
  int bisect_result = TestOneDepthwiseConvWithGivenOutputShift<Ac>(
      input_data, input_dims, input_offset, filter_data, filter_dims,
      filter_offset, bias_data, bias_dims, stride, pad_width, pad_height,
      depth_multiplier, output_offset, output_multiplier,
      output_shift_bisect_midpoint, output_activation_min,
      output_activation_max, output_dims);
  // At this point we know that the test succeeded (otherwise it would have
  // aborted).
  if (bisect_result == 0) {
    // The result isn't particularly saturated on one or the other side.
    // All good, we're done.
    return;
  }
  if (output_activation_bisect_start == output_activation_bisect_end - 1) {
    // There is still some saturation on one side, but the bisection is
    // finished anyways. We're done; nothing more we can do about it. This
    // happens
    // in particular when using an activation with a narrow range.
    return;
  }
  // Continue the bisection based on the present result.
  int new_output_activation_bisect_start = bisect_result == 1
                                               ? output_shift_bisect_midpoint
                                               : output_activation_bisect_start;
  int new_output_activation_bisect_end = bisect_result == 1
                                             ? output_activation_bisect_end
                                             : output_shift_bisect_midpoint;
  TestOneDepthwiseConvBisectOutputShift<Ac>(
      input_data, input_dims, input_offset, filter_data, filter_dims,
      filter_offset, bias_data, bias_dims, stride, pad_width, pad_height,
      depth_multiplier, output_offset, output_multiplier,
      new_output_activation_bisect_start, new_output_activation_bisect_end,
      output_activation_min, output_activation_max, output_dims);
}

template <FusedActivationFunctionType Ac>
void TestOneDepthwiseConv(
    const std::uint8_t* input_data, const Dims<4>& input_dims,
    std::int32_t input_offset, const std::uint8_t* filter_data,
    const Dims<4>& filter_dims, std::int32_t filter_offset,
    const std::int32_t* bias_data, const Dims<4>& bias_dims, int stride,
    int pad_width, int pad_height, int depth_multiplier,
    std::int32_t output_offset, std::int32_t output_multiplier,
    std::int32_t output_activation_min, std::int32_t output_activation_max,
    const Dims<4>& output_dims) {
  TestOneDepthwiseConvBisectOutputShift<Ac>(
      input_data, input_dims, input_offset, filter_data, filter_dims,
      filter_offset, bias_data, bias_dims, stride, pad_width, pad_height,
      depth_multiplier, output_offset, output_multiplier, 0, 32,
      output_activation_min, output_activation_max, output_dims);
}

void TestOneDepthwiseConv(
    FusedActivationFunctionType Ac, const std::uint8_t* input_data,
    const Dims<4>& input_dims, std::int32_t input_offset,
    const std::uint8_t* filter_data, const Dims<4>& filter_dims,
    std::int32_t filter_offset, const std::int32_t* bias_data,
    const Dims<4>& bias_dims, int stride, int pad_width, int pad_height,
    int depth_multiplier, std::int32_t output_offset,
    std::int32_t output_multiplier, std::int32_t output_activation_min,
    std::int32_t output_activation_max, const Dims<4>& output_dims) {
#define TOCO_HANDLE_CASE(AC_TYPE)                                           \
  if (AC_TYPE == Ac) {                                                      \
    TestOneDepthwiseConv<AC_TYPE>(                                          \
        input_data, input_dims, input_offset, filter_data, filter_dims,     \
        filter_offset, bias_data, bias_dims, stride, pad_width, pad_height, \
        depth_multiplier, output_offset, output_multiplier,                 \
        output_activation_min, output_activation_max, output_dims);         \
    return;                                                                 \
  }
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kNone)
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kRelu)
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kRelu1)
  TOCO_HANDLE_CASE(FusedActivationFunctionType::kRelu6)
#undef TOCO_HANDLE_CASE
}

bool TryTestDepthwiseConv(int batch, int input_depth, int input_width,
                          int input_height, int filter_width, int filter_height,
                          int depth_multiplier, int stride,
                          PaddingType padding_type) {
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
  int output_activation_min = 0;
  int output_activation_max = 255;
  if (ac != FusedActivationFunctionType::kNone && UniformRandomInt(0, 1)) {
    output_activation_min = UniformRandomInt(0, 50);
    output_activation_max = UniformRandomInt(200, 255);
  }
  const std::int32_t output_multiplier =
      UniformRandomInt(1 << 29, std::numeric_limits<std::int32_t>::max());
  const std::int32_t input_offset = UniformRandomInt(-256, 0);
  const std::int32_t filter_offset = UniformRandomInt(-256, 0);
  const std::int32_t output_offset = UniformRandomInt(-256, 0);
  Dims<4> input_dims_inference =
      MakeDimsForInference(input_depth, input_width, input_height, batch);
  Dims<4> output_dims_inference;
  int pad_width, pad_height;
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
  std::vector<std::uint8_t> input_data(input_buffer_size);
  std::vector<std::uint8_t> filter_data(filter_buffer_size);
  std::vector<std::int32_t> bias_data(output_depth);
  FillRandom(&input_data);
  FillRandom(&filter_data);
  FillRandom(&bias_data, -10000, 10000);
  TestOneDepthwiseConv(ac, input_data.data(), input_dims_inference,
                       input_offset, filter_data.data(), filter_dims_inference,
                       filter_offset, bias_data.data(), bias_dims_inference,
                       stride, pad_width, pad_height, depth_multiplier,
                       output_offset, output_multiplier, output_activation_min,
                       output_activation_max, output_dims_inference);
  return true;
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
  const auto padding_type =
      UniformRandomInt(0, 1) ? PaddingType::kSame : PaddingType::kValid;

  return TryTestDepthwiseConv(batch, input_depth, input_width, input_height,
                              filter_width, filter_height, depth_multiplier,
                              stride, padding_type);
}

// Tests parameters for the 3x3 filter kernel.
bool TryTestOneDepthwiseConv3x3Filter() {
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = 8 * ExponentialRandomPositiveInt(0.9f, 10, 50);
  const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int filter_width = 3;
  const int filter_height = 3;
  const int depth_multiplier = 1;
  const int stride = UniformRandomInt(1, 2);
  // Although the kernel supports only kValid padding, we test that kSame
  // is using the correct code path.
  const auto padding_type =
      UniformRandomInt(0, 1) ? PaddingType::kSame : PaddingType::kValid;

  return TryTestDepthwiseConv(batch, input_depth, input_width, input_height,
                              filter_width, filter_height, depth_multiplier,
                              stride, padding_type);
}

void TestOneDepthwiseConv() {
  while (!TryTestOneDepthwiseConv()) {
  }
}

void TestOneDepthwiseConv3x3Filter() {
  while (!TryTestOneDepthwiseConv3x3Filter()) {
  }
}

TEST(TestDepthwiseConv, TestDepthwiseConv) {
  const int kTestsToRun = 10 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv();
  }
}

TEST(TestDepthwiseConv3x3Filter, TestDepthwiseConv) {
  const int kTestsToRun = 3 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv3x3Filter();
  }
}

}  // namespace
}  // namespace tflite
