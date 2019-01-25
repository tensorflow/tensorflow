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
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

#define ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"

namespace tflite {
namespace {

enum class ForceKernelInvocation {
  // Run all tests against kUseStandardEntry even if also testing another
  // kernel, since we need to be sure that the main DepthwiseConv() function in
  // optimized_ops.h dispatches to a correctly-executing kernel.
  kNone = 0,  // The "default" option: use the normal DepthwiseConv
              // kernel (entry) function.
  kUseGenericKernel,
  kUseNeon3x3,            // 3x3 kernel that uses NEON when available.
  kUseNeon3x3DotProduct,  // 3x3 kernel that uses dot-product enabled NEON when
                          // available.
};

inline void DispatchDepthwiseConv(
    ForceKernelInvocation forced_invocation, const DepthwiseParams& params,
    const RuntimeShape& input_shape, const uint8* input_data,
    const RuntimeShape& filter_shape, const uint8* filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const RuntimeShape& output_shape, uint8* output_data) {
  switch (forced_invocation) {
    case ForceKernelInvocation::kUseNeon3x3: {
// Enable for arm64 except for the Nvidia Linux 4 Tegra (L4T) running on
// Jetson TX-2. This compiler does not support the offsetof() macro.
#if defined(__aarch64__) && !defined(GOOGLE_L4T)
      const int stride_width = params.stride_width;
      const int stride_height = params.stride_height;
      const int pad_width = params.padding_values.width;
      const int pad_height = params.padding_values.height;
      const int output_shift = params.output_shift;
      const int depth_multiplier = params.depth_multiplier;
      const int dilation_width_factor = params.dilation_width_factor;
      const int dilation_height_factor = params.dilation_height_factor;

      // Check that parameter combination is supported.
      const bool basic_3x3_kernel_supported =
          optimized_ops::Fast3x3FilterKernelSupported(
              input_shape, filter_shape, stride_width, stride_height,
              dilation_width_factor, dilation_height_factor, pad_width,
              pad_height, depth_multiplier, output_shape, output_shift);
      ASSERT_TRUE(basic_3x3_kernel_supported)
          << "pad_width = " << params.padding_values.width
          << " pad_height = " << params.padding_values.height
          << " input_width = " << input_shape.Dims(1)
          << " input_height = " << input_shape.Dims(2)
          << " output_width = " << output_shape.Dims(1)
          << " output_height = " << output_shape.Dims(2);

      // Call kernel optimized for depthwise convolutions using 3x3 filters.
      optimized_ops::DepthwiseConv3x3Filter(
          params, input_shape, input_data, filter_shape, filter_data,
          bias_shape, bias_data, output_shape, output_data);
      return;
#else
      break;
#endif
    }
    case ForceKernelInvocation::kUseNeon3x3DotProduct: {
// Enable for arm64 except for the Nvidia Linux 4 Tegra (L4T) running on
// Jetson TX-2. This compiler does not support the offsetof() macro.
#if defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__) && \
    !defined(GOOGLE_L4T)
      using optimized_ops::DotProduct3x3KernelType;
      DotProduct3x3KernelType kernel_type =
          optimized_ops::CategorizeDotProductKernel(params);
      switch (kernel_type) {
        case DotProduct3x3KernelType::kPlain:
          // TODO(b/118430534): Implement optimized kernel.
          optimized_ops::DepthwiseConv3x3Filter(
              params, input_shape, input_data, filter_shape, filter_data,
              bias_shape, bias_data, output_shape, output_data);
          return;
        case DotProduct3x3KernelType::kWithDepthMultiplication:
          // TODO(b/118430338): Implement optimized kernel.
          optimized_ops::DepthwiseConvGeneral(
              params, input_shape, input_data, filter_shape, filter_data,
              bias_shape, bias_data, output_shape, output_data);
          return;
        case DotProduct3x3KernelType::kWithPad0Stride2:
          // TODO(b/118430338): Implement optimized kernel.
          optimized_ops::DepthwiseConv3x3Filter(
              params, input_shape, input_data, filter_shape, filter_data,
              bias_shape, bias_data, output_shape, output_data);
          return;
        case DotProduct3x3KernelType::kWithPad1Stride1:
          // TODO(b/118430338): Implement optimized kernel.
          optimized_ops::DepthwiseConvGeneral(
              params, input_shape, input_data, filter_shape, filter_data,
              bias_shape, bias_data, output_shape, output_data);
          return;
        case DotProduct3x3KernelType::kNone:
        default:
          break;
      }
#endif
      break;
    }
    case ForceKernelInvocation::kUseGenericKernel: {
      optimized_ops::DepthwiseConvGeneral(params, input_shape, input_data,
                                          filter_shape, filter_data, bias_shape,
                                          bias_data, output_shape, output_data);
      return;
    }
    case ForceKernelInvocation::kNone:
    default:
      break;
  }
  optimized_ops::DepthwiseConv(params, input_shape, input_data, filter_shape,
                               filter_data, bias_shape, bias_data, output_shape,
                               output_data);
}

// Runs the DepthwiseConv and compares against the reference implementation.
int TestOneDepthwiseConvWithGivenOutputShift(
    ForceKernelInvocation forced_invocation, const std::uint8_t* input_data,
    const RuntimeShape& input_shape, std::int32_t input_offset,
    const std::uint8_t* filter_data, const RuntimeShape& filter_shape,
    std::int32_t filter_offset, const std::int32_t* bias_data,
    const RuntimeShape& bias_shape, int stride, PaddingType padding_type,
    int pad_width, int pad_height, int depth_multiplier,
    std::int32_t output_offset, std::int32_t output_multiplier,
    int output_shift, std::int32_t output_activation_min,
    std::int32_t output_activation_max, const RuntimeShape& output_shape) {
  const int output_buffer_size = output_shape.FlatSize();
  std::vector<std::uint8_t> output_data(output_buffer_size);
  std::vector<std::uint8_t> reference_output_data(output_buffer_size);

  tflite::DepthwiseParams op_params;
  op_params.padding_type = padding_type;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride;
  op_params.stride_height = stride;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.depth_multiplier = depth_multiplier;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = -output_shift;
  reference_ops::DepthwiseConv(op_params, input_shape, input_data, filter_shape,
                               filter_data, bias_shape, bias_data, output_shape,
                               reference_output_data.data());
  DispatchDepthwiseConv(forced_invocation, op_params, input_shape, input_data,
                        filter_shape, filter_data, bias_shape, bias_data,
                        output_shape, output_data.data());
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
void TestOneDepthwiseConvBisectOutputShift(
    ForceKernelInvocation forced_invocation, const std::uint8_t* input_data,
    const RuntimeShape& input_shape, std::int32_t input_offset,
    const std::uint8_t* filter_data, const RuntimeShape& filter_shape,
    std::int32_t filter_offset, const std::int32_t* bias_data,
    const RuntimeShape& bias_shape, int stride, PaddingType padding_type,
    int pad_width, int pad_height, int depth_multiplier,
    std::int32_t output_offset, std::int32_t output_multiplier,
    int output_activation_bisect_start, int output_activation_bisect_end,
    std::int32_t output_activation_min, std::int32_t output_activation_max,
    const RuntimeShape& output_shape) {
  ASSERT_LT(output_activation_bisect_start, output_activation_bisect_end)
      << "Bisection failed ?!?!";
  int output_shift_bisect_midpoint =
      (output_activation_bisect_start + output_activation_bisect_end) / 2;
  int bisect_result = TestOneDepthwiseConvWithGivenOutputShift(
      forced_invocation, input_data, input_shape, input_offset, filter_data,
      filter_shape, filter_offset, bias_data, bias_shape, stride, padding_type,
      pad_width, pad_height, depth_multiplier, output_offset, output_multiplier,
      output_shift_bisect_midpoint, output_activation_min,
      output_activation_max, output_shape);
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
  TestOneDepthwiseConvBisectOutputShift(
      forced_invocation, input_data, input_shape, input_offset, filter_data,
      filter_shape, filter_offset, bias_data, bias_shape, stride, padding_type,
      pad_width, pad_height, depth_multiplier, output_offset, output_multiplier,
      new_output_activation_bisect_start, new_output_activation_bisect_end,
      output_activation_min, output_activation_max, output_shape);
}

void TestOneDepthwiseConv(
    ForceKernelInvocation forced_invocation, const std::uint8_t* input_data,
    const RuntimeShape& input_shape, std::int32_t input_offset,
    const std::uint8_t* filter_data, const RuntimeShape& filter_shape,
    std::int32_t filter_offset, const std::int32_t* bias_data,
    const RuntimeShape& bias_shape, int stride, PaddingType padding_type,
    int pad_width, int pad_height, int depth_multiplier,
    std::int32_t output_offset, std::int32_t output_multiplier,
    std::int32_t output_activation_min, std::int32_t output_activation_max,
    const RuntimeShape& output_shape) {
  TestOneDepthwiseConvBisectOutputShift(
      forced_invocation, input_data, input_shape, input_offset, filter_data,
      filter_shape, filter_offset, bias_data, bias_shape, stride, padding_type,
      pad_width, pad_height, depth_multiplier, output_offset, output_multiplier,
      0, 32, output_activation_min, output_activation_max, output_shape);
}

bool TryTestDepthwiseConv(ForceKernelInvocation forced_invocation, int batch,
                          int input_depth, int input_width, int input_height,
                          int filter_width, int filter_height,
                          int depth_multiplier, int stride,
                          int dilation_width_factor, int dilation_height_factor,
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
  int output_activation_min = 0;
  int output_activation_max = 255;
  if (UniformRandomInt(0, 1)) {
    output_activation_min = UniformRandomInt(0, 50);
    output_activation_max = UniformRandomInt(200, 255);
  }
  const std::int32_t output_multiplier =
      UniformRandomInt(1 << 29, std::numeric_limits<std::int32_t>::max());
  const std::int32_t input_offset = UniformRandomInt(-256, 0);
  const std::int32_t filter_offset = UniformRandomInt(-256, 0);
  const std::int32_t output_offset = UniformRandomInt(-256, 0);
  RuntimeShape input_shape_inference(
      {batch, input_height, input_width, input_depth});
  RuntimeShape output_shape_inference;
  int pad_width, pad_height;
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
  std::vector<std::uint8_t> input_data(input_buffer_size);
  std::vector<std::uint8_t> filter_data(filter_buffer_size);
  std::vector<std::int32_t> bias_data(output_depth);
  FillRandom(&input_data);
  FillRandom(&filter_data);
  FillRandom(&bias_data, -10000, 10000);
  TestOneDepthwiseConv(
      forced_invocation, input_data.data(), input_shape_inference, input_offset,
      filter_data.data(), filter_shape_inference, filter_offset,
      bias_data.data(), bias_shape_inference, stride, padding_type, pad_width,
      pad_height, depth_multiplier, output_offset, output_multiplier,
      output_activation_min, output_activation_max, output_shape_inference);
  return true;
}

// This function picks some random DepthwiseConv params, which may or may not
// be legal. If they're not legal, it returns false. If they're legal,
// it runs the DepthwiseConv test and returns true. This allows the caller
// to loop until a test has been run.
bool TryTestOneDepthwiseConv(ForceKernelInvocation forced_invocation) {
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
  const int dilation_width_factor = RandomElement(std::vector<int>({1, 2, 4}));
  const int dilation_height_factor = RandomElement(std::vector<int>({1, 2, 4}));
  const auto padding_type =
      UniformRandomInt(0, 1) ? PaddingType::kSame : PaddingType::kValid;

  return TryTestDepthwiseConv(
      forced_invocation, batch, input_depth, input_width, input_height,
      filter_width, filter_height, depth_multiplier, stride,
      dilation_width_factor, dilation_height_factor, padding_type);
}

// Tests parameters for the 3x3 filter kernel.
bool TryTestOneDepthwiseConv3x3Filter(ForceKernelInvocation forced_invocation) {
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = 8 * ExponentialRandomPositiveInt(0.9f, 10, 50);
  int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
  int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int filter_width = 3;
  const int filter_height = 3;
  const int depth_multiplier = 1;
  const int stride = UniformRandomInt(1, 2);
  // We don't support dilations in the 3x3 filter.
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;
  const auto padding_type =
      UniformRandomInt(0, 1) ? PaddingType::kSame : PaddingType::kValid;

  // Adjust for, or reject, special cases.
  if (forced_invocation != ForceKernelInvocation::kNone) {
    // With stride == 2 and SAME, padding width and height are the left and top
    // padding amounts. When there is an even input dimension, padding + 1 is
    // required on the right / bottom. This is not handled by these kernels, so
    // we bump the input dimensions.
    if (padding_type == PaddingType::kSame && stride == 2) {
      input_width = 2 * (input_width / 2) + 1;
      input_height = 2 * (input_height / 2) + 1;
    }

    // The padded 3x3 kernel (with kSame) does not support input_width == 1 when
    // input_height > 1, and vice versa.
    if (padding_type == PaddingType::kSame &&
        (input_width > 1) != (input_height > 1)) {
      return false;
    }
  }

  return TryTestDepthwiseConv(
      forced_invocation, batch, input_depth, input_width, input_height,
      filter_width, filter_height, depth_multiplier, stride,
      dilation_width_factor, dilation_height_factor, padding_type);
}

// Tests with parameters suited to dot-product-NEON 3x3 filter kernels.
bool TryTestOneNeonDot3x3(ForceKernelInvocation forced_invocation,
                          bool test_stride, bool test_pad,
                          bool test_depth_multiplier) {
  const int batch = 1;
  const int input_depth = test_depth_multiplier
                              ? 1
                              : 8 * ExponentialRandomPositiveInt(0.9f, 10, 50);
  const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int filter_width = 3;
  const int filter_height = 3;
  const int depth_multiplier =
      test_depth_multiplier ? 8 * ExponentialRandomPositiveInt(0.8f, 1, 6) : 1;
  const int stride = test_stride ? 2 : 1;
  // We don't support dilations in the 3x3 filter.
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;
  const auto padding_type = test_pad ? PaddingType::kSame : PaddingType::kValid;

  return TryTestDepthwiseConv(
      forced_invocation, batch, input_depth, input_width, input_height,
      filter_width, filter_height, depth_multiplier, stride,
      dilation_width_factor, dilation_height_factor, padding_type);
}

void TestOneDepthwiseConv(ForceKernelInvocation forced_invocation) {
  while (!TryTestOneDepthwiseConv(forced_invocation)) {
  }
}

void TestOneDepthwiseConv3x3Filter(ForceKernelInvocation forced_invocation) {
  while (!TryTestOneDepthwiseConv3x3Filter(forced_invocation)) {
  }
}

void TestOneNeonDot3x3(ForceKernelInvocation forced_invocation,
                       bool test_stride, bool test_pad,
                       bool test_depth_multiplier) {
  while (!TryTestOneNeonDot3x3(forced_invocation, test_stride, test_pad,
                               test_depth_multiplier)) {
  }
}

TEST(TestDepthwiseConv, TestDepthwiseConv) {
  const int kTestsToRun = 10 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv(ForceKernelInvocation::kNone);
  }
}

// Run basic coverage test against the generic kernel.
TEST(TestDepthwiseConv, TestGenericKernel) {
  const int kTestsToRun = 10 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv(ForceKernelInvocation::kUseGenericKernel);
  }
}

TEST(TestDepthwiseConv, TestKernel3x3Filter) {
  const int kTestsToRun = 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv3x3Filter(ForceKernelInvocation::kNone);
  }
}

// While the 3x3 coverage test is primarily targeted at specialized kernels, we
// also run it against the generic kernel, optionally with fewer invocations.
TEST(TestDepthwiseConv, TestGenericKernel3x3Filter) {
  const int kTestsToRun = 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv3x3Filter(ForceKernelInvocation::kUseGenericKernel);
  }
}

TEST(TestDepthwiseConv, TestNeon3x3Filter) {
  const int kTestsToRun = 3 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv3x3Filter(ForceKernelInvocation::kUseNeon3x3);
  }
}

// No stride, no depth multiplier, no pad.
TEST(TestDepthwiseConv, TestNeonDot3x3Plain) {
  const int kTestsToRun = 3 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneNeonDot3x3(ForceKernelInvocation::kUseNeon3x3DotProduct,
                      /*test_stride=*/false, /*test_pad=*/false,
                      /*test_depth_multiplier=*/false);
  }
}

TEST(TestDepthwiseConv, TestNeonDot3x3DepthMultiplier) {
  const int kTestsToRun = 3 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneNeonDot3x3(ForceKernelInvocation::kUseNeon3x3DotProduct,
                      /*test_stride=*/false, /*test_pad=*/false,
                      /*test_depth_multiplier=*/true);
  }
}

TEST(TestDepthwiseConv, TestNeonDot3x3Stride2) {
  const int kTestsToRun = 3 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneNeonDot3x3(ForceKernelInvocation::kUseNeon3x3DotProduct,
                      /*test_stride=*/true, /*test_pad=*/false,
                      /*test_depth_multiplier=*/false);
  }
}

TEST(TestDepthwiseConv, TestNeonDot3x3Pad1) {
  const int kTestsToRun = 3 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneNeonDot3x3(ForceKernelInvocation::kUseNeon3x3DotProduct,
                      /*test_stride=*/false, /*test_pad=*/true,
                      /*test_depth_multiplier=*/false);
  }
}

}  // namespace
}  // namespace tflite
