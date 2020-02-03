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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ruy/context.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

#define ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#include "absl/strings/substitute.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8_transitional.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"

namespace tflite {
namespace {

using optimized_ops::depthwise_conv::DotProduct3x3KernelType;
using optimized_ops::depthwise_conv::QuantizationType;
using optimized_ops::depthwise_conv::QuantizationTypeImpl;
using ::testing::Bool;
using ::testing::Values;

#if defined(__aarch64__)
static constexpr bool kLooseIntrinsicsTolerance = false;
#else
static constexpr bool kLooseIntrinsicsTolerance = true;
#endif

// Currently, this is used in place of a Boolean "is symmetric?".
enum class ParamsSpecialization {
  kNone = 0,
  kSymmetric,  // Symmetric quantization: zero represented by 128.
};

static constexpr int kSymmetricZeroPoint = 128;

// Extend coverage distribution in a specific aspect, either explicitly chosen
// or randomly chosen as in a mixture distribution.
enum class CoverageExtension {
  kNone = 0,
  kLargeHeights = 1,
  kLargeWidths = 2,
  kNumOptions
};

// The TestParam structure below is the preferred parameterization of tests. A
// tuple version is defined in order to support value-parameterized tests.
typedef std::tuple<DepthwiseConvImplementation, int, QuantizationType, bool,
                   bool, bool, DepthwiseConvOutputRounding, int, bool>
    TestParamTuple;

struct TestParam {
  TestParam() = default;

  explicit TestParam(TestParamTuple param_tuple)
      : forced_invocation(::testing::get<0>(param_tuple)),
        tests_to_run(::testing::get<1>(param_tuple)),
        quantization_type(::testing::get<2>(param_tuple)),
        test_stride(::testing::get<3>(param_tuple)),
        test_pad(::testing::get<4>(param_tuple)),
        test_depth_multiplier(::testing::get<5>(param_tuple)),
        output_rounding(::testing::get<6>(param_tuple)),
        num_threads(::testing::get<7>(param_tuple)),
        loose_tolerance(::testing::get<8>(param_tuple)) {}

  static std::string TestNameSuffix(
      const ::testing::TestParamInfo<TestParamTuple>& info) {
    const TestParam param(info.param);
    return absl::Substitute(
        "invocation_$0_quantization_$1_stride_$2_pad_$3_depth_mult_$4",
        static_cast<int>(param.quantization_type),
        static_cast<int>(param.forced_invocation), param.test_stride,
        param.test_pad, param.test_depth_multiplier);
  }

  DepthwiseConvImplementation forced_invocation =
      DepthwiseConvImplementation::kNone;
  int tests_to_run = 0;
  QuantizationType quantization_type = QuantizationType::kNonPerChannelUint8;
  bool test_stride = false;
  bool test_pad = false;
  bool test_depth_multiplier = false;
  DepthwiseConvOutputRounding output_rounding =
      DepthwiseConvOutputRounding::kNone;
  int num_threads = 1;
  bool loose_tolerance = false;
};

template <QuantizationType quantization_type>
inline void DispatchDepthwiseConvGeneral(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        input_data,
    const RuntimeShape& filter_shape,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const std::int32_t* output_shift_adjust,
    const std::int32_t* output_multiplier_adjust,
    const RuntimeShape& output_shape,
    typename QuantizationTypeImpl<quantization_type>::ExternalType* output_data,
    int thread_start, int thread_end, int thread_dim) {
  optimized_ops::depthwise_conv::DepthwiseConvGeneral(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data, thread_start, thread_end,
      thread_dim);
}

template <QuantizationType quantization_type>
inline void DispatchDepthwiseConvImpl(
    const TestParam& test_param, const DepthwiseParams& params,
    const RuntimeShape& input_shape,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        input_data,
    const RuntimeShape& filter_shape,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const RuntimeShape& output_shape,
    typename QuantizationTypeImpl<quantization_type>::ExternalType*
        output_data) {
  switch (test_param.forced_invocation) {
    case DepthwiseConvImplementation::kUseNeon3x3: {
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
          optimized_ops::depthwise_conv::Fast3x3FilterKernelSupported(
              input_shape, filter_shape, stride_width, stride_height,
              dilation_width_factor, dilation_height_factor, pad_width,
              pad_height, depth_multiplier, output_shape, output_shift);
      ASSERT_TRUE(basic_3x3_kernel_supported)
          << "pad_width = " << params.padding_values.width
          << " pad_height = " << params.padding_values.height
          << " input_width = " << input_shape.Dims(2)
          << " input_height = " << input_shape.Dims(1)
          << " output_width = " << output_shape.Dims(2)
          << " output_height = " << output_shape.Dims(1);

      // Call kernel optimized for depthwise convolutions using 3x3 filters.
      switch (test_param.output_rounding) {
        case DepthwiseConvOutputRounding::kAwayFromZero:
          optimized_ops::depthwise_conv::DepthwiseConv3x3Filter<
              DepthwiseConvOutputRounding::kAwayFromZero>(
              params, input_shape, input_data, filter_shape, filter_data,
              bias_shape, bias_data, output_shape, output_data,
              /*thread_start=*/0,
              /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
          return;
        case DepthwiseConvOutputRounding::kUpward:
          optimized_ops::depthwise_conv::DepthwiseConv3x3Filter<
              DepthwiseConvOutputRounding::kUpward>(
              params, input_shape, input_data, filter_shape, filter_data,
              bias_shape, bias_data, output_shape, output_data,
              /*thread_start=*/0,
              /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
          return;
        default:
          break;
      }
#endif
      break;
    }
    case DepthwiseConvImplementation::kUseNeon3x3DotProduct: {
      // This is compiled-in even if dot-product instructions are unavailable.
      // However, tests should skip dot-product testing in that case and not
      // call this code.
#if defined(__aarch64__) && !defined(GOOGLE_L4T) && defined(__ANDROID__) && \
    defined(__clang__)
      DotProduct3x3KernelType kernel_type =
          optimized_ops::depthwise_conv::CategorizeDotProductKernel(
              input_shape, filter_shape, output_shape, params);

      ASSERT_NE(kernel_type, DotProduct3x3KernelType::kNone)
          << "Kernel type = " << static_cast<int>(kernel_type);

      optimized_ops::depthwise_conv::DepthwiseConvDotProduct3x3Impl<
          DepthwiseConvImplementation::kUseNeon3x3DotProduct,
          quantization_type>(
          params, input_shape, input_data, filter_shape, filter_data,
          bias_shape, bias_data, output_shape, output_data, /*thread_start=*/0,
          /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
      return;
#endif
      break;
    }
    case DepthwiseConvImplementation::kUseCModel3x3DotProduct: {
      DotProduct3x3KernelType kernel_type =
          optimized_ops::depthwise_conv::CategorizeDotProductKernel(
              input_shape, filter_shape, output_shape, params);

      ASSERT_TRUE(
          kernel_type == DotProduct3x3KernelType::kPlain ||
          kernel_type == DotProduct3x3KernelType::kStride2 ||
          kernel_type ==
              DotProduct3x3KernelType::kWithDepthMultiplicationStride1 ||
          kernel_type ==
              DotProduct3x3KernelType::kWithDepthMultiplicationStride2)
          << "Kernel type = " << static_cast<int>(kernel_type)
          << " depth_multiplier = " << params.depth_multiplier
          << " pad_width = " << params.padding_values.width
          << " pad_height = " << params.padding_values.height
          << " stride_width = " << params.stride_width
          << " stride_height = " << params.stride_height
          << " input_width = " << input_shape.Dims(2)
          << " input_height = " << input_shape.Dims(1)
          << " output_width = " << output_shape.Dims(2)
          << " output_height = " << output_shape.Dims(1)
          << " depth = " << input_shape.Dims(3)
          << " buffer need = " << input_shape.Dims(3) * input_shape.Dims(2) * 6
          << " input_offset = " << params.input_offset;

      optimized_ops::depthwise_conv::DepthwiseConvDotProduct3x3Impl<
          DepthwiseConvImplementation::kUseCModel3x3DotProduct,
          quantization_type>(
          params, input_shape, input_data, filter_shape, filter_data,
          bias_shape, bias_data, output_shape, output_data, /*thread_start=*/0,
          /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
      return;
    }
    case DepthwiseConvImplementation::kUseUnwound3x3DotProduct: {
      DotProduct3x3KernelType kernel_type =
          optimized_ops::depthwise_conv::CategorizeDotProductKernel(
              input_shape, filter_shape, output_shape, params);
      ASSERT_TRUE(
          kernel_type == DotProduct3x3KernelType::kPlain ||
          kernel_type == DotProduct3x3KernelType::kStride2 ||
          kernel_type ==
              DotProduct3x3KernelType::kWithDepthMultiplicationStride1 ||
          kernel_type ==
              DotProduct3x3KernelType::kWithDepthMultiplicationStride2);
      optimized_ops::depthwise_conv::DepthwiseConvDotProduct3x3Impl<
          DepthwiseConvImplementation::kUseUnwound3x3DotProduct,
          quantization_type>(
          params, input_shape, input_data, filter_shape, filter_data,
          bias_shape, bias_data, output_shape, output_data, /*thread_start=*/0,
          /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
      return;
    }
    case DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct: {
#if defined(USE_NEON)
      DotProduct3x3KernelType kernel_type =
          optimized_ops::depthwise_conv::CategorizeDotProductKernel(
              input_shape, filter_shape, output_shape, params);

      ASSERT_TRUE(
          kernel_type == DotProduct3x3KernelType::kPlain ||
          kernel_type == DotProduct3x3KernelType::kStride2 ||
          kernel_type ==
              DotProduct3x3KernelType::kWithDepthMultiplicationStride1 ||
          kernel_type ==
              DotProduct3x3KernelType::kWithDepthMultiplicationStride2);
      optimized_ops::depthwise_conv::DepthwiseConvDotProduct3x3Impl<
          DepthwiseConvImplementation::kUseIntrinsics3x3DotProduct,
          quantization_type>(
          params, input_shape, input_data, filter_shape, filter_data,
          bias_shape, bias_data, output_shape, output_data, /*thread_start=*/0,
          /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
      return;
#else
      break;
#endif
    }
    case DepthwiseConvImplementation::kUseGenericKernel: {
      DispatchDepthwiseConvGeneral<quantization_type>(
          params, input_shape, input_data, filter_shape, filter_data,
          bias_shape, bias_data, nullptr, nullptr, output_shape, output_data,
          /*thread_start=*/0,
          /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
      return;
    }
    case DepthwiseConvImplementation::kNone:
    default:
      break;
  }

  EXPECT_EQ(test_param.forced_invocation, DepthwiseConvImplementation::kNone)
      << "TODO(b/118426582) requested kernel was not invoked / available yet: "
      << " forced_invocation = "
      << static_cast<int>(test_param.forced_invocation)
      << " depth_multiplier = " << params.depth_multiplier
      << " pad_width = " << params.padding_values.width
      << " pad_height = " << params.padding_values.height
      << " stride_width = " << params.stride_width
      << " stride_height = " << params.stride_height
      << " input_width = " << input_shape.Dims(2)
      << " input_height = " << input_shape.Dims(1)
      << " output_width = " << output_shape.Dims(2)
      << " output_height = " << output_shape.Dims(1)
      << " depth = " << input_shape.Dims(3)
      << " buffer need = " << input_shape.Dims(3) * input_shape.Dims(2) * 6
      << " input_offset = " << params.input_offset;

  CpuBackendContext backend_context;
  backend_context.SetMaxNumThreads(test_param.num_threads);
  optimized_ops::DepthwiseConv<
      typename QuantizationTypeImpl<quantization_type>::ExternalType, int32>(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data, &backend_context);
}

template <QuantizationType quantization_type>
inline void DispatchDepthwiseConv(
    const TestParam& test_param, const DepthwiseParams& params,
    const RuntimeShape& input_shape,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        input_data,
    const RuntimeShape& filter_shape,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        filter_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    const RuntimeShape& output_shape,
    typename QuantizationTypeImpl<quantization_type>::ExternalType*
        output_data) {
  DispatchDepthwiseConvImpl<quantization_type>(
      test_param, params, input_shape, input_data, filter_shape, filter_data,
      bias_shape, bias_data, output_shape, output_data);
}

template <QuantizationType quantization_type>
struct ReferenceRunner {};

template <>
struct ReferenceRunner<QuantizationType::kNonPerChannelUint8> {
  static inline void Run(
      const TestParam& test_param, const tflite::DepthwiseParams& op_params,
      const uint8* input_data, const RuntimeShape& input_shape,
      const uint8* filter_data, const RuntimeShape& filter_shape,
      const std::int32_t* bias_data, const RuntimeShape& bias_shape,
      const RuntimeShape& output_shape, uint8* reference_output_data) {
    switch (test_param.output_rounding) {
      case DepthwiseConvOutputRounding::kUpward:
        reference_ops::depthwise_conv::DepthwiseConvBasicKernel<
            DepthwiseConvOutputRounding::kUpward>::Run(op_params, input_shape,
                                                       input_data, filter_shape,
                                                       filter_data, bias_shape,
                                                       bias_data, output_shape,
                                                       reference_output_data);
        break;
      case DepthwiseConvOutputRounding::kAwayFromZero:
        reference_ops::DepthwiseConv(
            op_params, input_shape, input_data, filter_shape, filter_data,
            bias_shape, bias_data, output_shape, reference_output_data);
        break;
      case DepthwiseConvOutputRounding::kNone:
      default:
        EXPECT_NE(test_param.output_rounding,
                  DepthwiseConvOutputRounding::kNone);
        break;
    }
  }
};

template <QuantizationType quantization_type>
// Runs the DepthwiseConv and compares against the reference implementation.
int TestOneDepthwiseConvWithGivenOutputShift(
    const TestParam& test_param,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        input_data,
    const RuntimeShape& input_shape, std::int32_t input_offset,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        filter_data,
    const RuntimeShape& filter_shape, std::int32_t filter_offset,
    const std::int32_t* bias_data, const RuntimeShape& bias_shape, int stride,
    PaddingType padding_type, int pad_width, int pad_height,
    int depth_multiplier, std::int32_t output_offset,
    std::int32_t output_multiplier, const std::int32_t* output_shift_adjust,
    const std::int32_t* output_multiplier_adjust, int output_shift,
    std::int32_t output_activation_min, std::int32_t output_activation_max,
    const RuntimeShape& output_shape) {
  const int output_buffer_size = output_shape.FlatSize();
  std::vector<typename QuantizationTypeImpl<quantization_type>::ExternalType>
      output_data(output_buffer_size, 42);
  std::vector<typename QuantizationTypeImpl<quantization_type>::ExternalType>
      reference_output_data(output_buffer_size);

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

  const int depth = output_shape.Dims(3);
  std::vector<int32> output_multiplier_per_channel(depth, output_multiplier);
  std::vector<int32> output_shift_per_channel(depth, -output_shift);
  if (output_multiplier_adjust != nullptr) {
    for (int i = 0; i < depth; ++i) {
      output_multiplier_per_channel[i] += output_multiplier_adjust[i];
      output_shift_per_channel[i] += output_shift_adjust[i];
      output_shift_per_channel[i] = std::max(-31, output_shift_per_channel[i]);
    }
  }
  op_params.output_multiplier_per_channel =
      output_multiplier_per_channel.data();
  op_params.output_shift_per_channel =
      output_shift_per_channel.data();  // Negated wrt output_shift.

  ReferenceRunner<quantization_type>::Run(
      test_param, op_params, input_data, input_shape, filter_data, filter_shape,
      bias_data, bias_shape, output_shape, reference_output_data.data());

  DispatchDepthwiseConv<quantization_type>(
      test_param, op_params, input_shape, input_data, filter_shape, filter_data,
      bias_shape, bias_data, output_shape, output_data.data());
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

  int diff_mean_tolerance = 1;
  int diff_median_tolerance = 0;
  // The tolerance that we apply to means is tight, but we allow for a rounding
  // difference in one pixel, and loosen by another 1% for float comparison.
  float mean_tolerance = std::max(
      1e-5f, 1.01f / output_buffer_size * std::sqrt(1.f * depth_multiplier));
  if (test_param.loose_tolerance) {
    mean_tolerance = 500.f;
    diff_mean_tolerance = 256;
    diff_median_tolerance = 225;
  }

  // Normally we should require bit-for-bit exact results. Unfortunately a bug
  // in the Intel arm_neon_sse.h translation header that we use for x86 tests
  // causes 1-bit inaccuracy in the vqrdmulh_n_s32 intrinsic, which causes
  // off-by-1 errors in quantized DepthwiseConv ops. So we have to live with a
  // few off-by-one errors for now, yet still ensure that no more than a small
  // minority of values are wrong.
  EXPECT_LT(std::abs(mean_diff), mean_tolerance);
  EXPECT_LT(mean_abs_diff, mean_tolerance);
  EXPECT_LE(std::abs(median_diff), diff_median_tolerance);
  EXPECT_LE(std::abs(min_diff), diff_mean_tolerance);
  EXPECT_LE(std::abs(max_diff), diff_mean_tolerance);
  EXPECT_TRUE(std::abs(mean_diff) < mean_tolerance &&
              mean_abs_diff < mean_tolerance &&
              std::abs(median_diff) <= diff_median_tolerance &&
              std::abs(min_diff) <= diff_mean_tolerance &&
              std::abs(max_diff) <= diff_mean_tolerance)
      << "pad_width = " << op_params.padding_values.width
      << " pad_height = " << op_params.padding_values.height
      << " input_width = " << input_shape.Dims(2)
      << " input_height = " << input_shape.Dims(1)
      << " output_width = " << output_shape.Dims(2)
      << " output_height = " << output_shape.Dims(1)
      << " depth = " << input_shape.Dims(3)
      << " output_offset = " << op_params.output_offset
      << " output_multiplier = " << op_params.output_multiplier
      << " output_shift = " << op_params.output_shift;

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
// could do some statistics for large size, but they would be fragile at smaller
// sizes), and guessing wrong would mean that all the values get saturated so
// the test becomes vacuous. So we just bisect our way to reasonable
// output_shift values.
template <QuantizationType quantization_type>
void TestOneDepthwiseConvBisectOutputShift(
    const TestParam& test_param,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        input_data,
    const RuntimeShape& input_shape, std::int32_t input_offset,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        filter_data,
    const RuntimeShape& filter_shape, std::int32_t filter_offset,
    const std::int32_t* bias_data, const RuntimeShape& bias_shape, int stride,
    PaddingType padding_type, int pad_width, int pad_height,
    int depth_multiplier, std::int32_t output_offset,
    std::int32_t output_multiplier, const std::int32_t* output_shift_adjust,
    const std::int32_t* output_multiplier_adjust,
    int output_activation_bisect_start, int output_activation_bisect_end,
    std::int32_t output_activation_min, std::int32_t output_activation_max,
    const RuntimeShape& output_shape) {
  ASSERT_LT(output_activation_bisect_start, output_activation_bisect_end)
      << "Bisection failed ?!?!";
  int output_shift_bisect_midpoint =
      (output_activation_bisect_start + output_activation_bisect_end) / 2;
  int bisect_result =
      TestOneDepthwiseConvWithGivenOutputShift<quantization_type>(
          test_param, input_data, input_shape, input_offset, filter_data,
          filter_shape, filter_offset, bias_data, bias_shape, stride,
          padding_type, pad_width, pad_height, depth_multiplier, output_offset,
          output_multiplier, output_shift_adjust, output_multiplier_adjust,
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
  TestOneDepthwiseConvBisectOutputShift<quantization_type>(
      test_param, input_data, input_shape, input_offset, filter_data,
      filter_shape, filter_offset, bias_data, bias_shape, stride, padding_type,
      pad_width, pad_height, depth_multiplier, output_offset, output_multiplier,
      output_shift_adjust, output_multiplier_adjust,
      new_output_activation_bisect_start, new_output_activation_bisect_end,
      output_activation_min, output_activation_max, output_shape);
}

template <QuantizationType quantization_type>
void TestOneDepthwiseConv(
    const TestParam& test_param,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        input_data,
    const RuntimeShape& input_shape, std::int32_t input_offset,
    const typename QuantizationTypeImpl<quantization_type>::ExternalType*
        filter_data,
    const RuntimeShape& filter_shape, std::int32_t filter_offset,
    const std::int32_t* bias_data, const RuntimeShape& bias_shape, int stride,
    PaddingType padding_type, int pad_width, int pad_height,
    int depth_multiplier, std::int32_t output_offset,
    std::int32_t output_multiplier, const std::int32_t* output_shift_adjust,
    const std::int32_t* output_multiplier_adjust,
    std::int32_t output_activation_min, std::int32_t output_activation_max,
    const RuntimeShape& output_shape) {
  TestOneDepthwiseConvBisectOutputShift<quantization_type>(
      test_param, input_data, input_shape, input_offset, filter_data,
      filter_shape, filter_offset, bias_data, bias_shape, stride, padding_type,
      pad_width, pad_height, depth_multiplier, output_offset, output_multiplier,
      output_shift_adjust, output_multiplier_adjust, 0, 32,
      output_activation_min, output_activation_max, output_shape);
}

bool TryTestDepthwiseConv(const TestParam& test_param,
                          ParamsSpecialization params_specialization, int batch,
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

  int output_activation_min;
  int output_activation_max;
  std::int32_t output_multiplier;
  std::int32_t input_offset;
  std::int32_t output_offset;

  output_activation_min = 0;
  output_activation_max = 255;
  if (UniformRandomInt(0, 1)) {
    output_activation_min = UniformRandomInt(0, 50);
    output_activation_max = UniformRandomInt(200, 255);
  }
  output_multiplier =
      UniformRandomInt(1 << 29, std::numeric_limits<std::int32_t>::max());
  input_offset = UniformRandomInt(-255, 0);
  output_offset = UniformRandomInt(0, 255);

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
  std::vector<std::int32_t> bias_data(output_depth);
  FillRandom(&bias_data, -10000, 10000);

  std::vector<std::uint8_t> input_data(input_buffer_size);
  std::vector<std::uint8_t> filter_data(filter_buffer_size);
  FillRandom(&input_data);
  FillRandom(&filter_data);

  std::int32_t filter_offset = -kSymmetricZeroPoint;
  if (params_specialization != ParamsSpecialization::kSymmetric) {
    filter_offset = UniformRandomInt(-255, 0);
  }

  TestOneDepthwiseConv<QuantizationType::kNonPerChannelUint8>(
      test_param, input_data.data(), input_shape_inference, input_offset,
      filter_data.data(), filter_shape_inference, filter_offset,
      bias_data.data(), bias_shape_inference, stride, padding_type, pad_width,
      pad_height, depth_multiplier, output_offset, output_multiplier,
      nullptr /*=output_shift_adjust*/, nullptr /*=output_multiplier_adjust*/,
      output_activation_min, output_activation_max, output_shape_inference);

  return true;
}

// This function picks some random DepthwiseConv params, which may or may not
// be legal. If they're not legal, it returns false. If they're legal,
// it runs the DepthwiseConv test and returns true. This allows the caller
// to loop until a test has been run.
bool TryTestOneDepthwiseConv(const TestParam& test_param,
                             ParamsSpecialization params_specialization) {
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
      test_param, params_specialization, batch, input_depth, input_width,
      input_height, filter_width, filter_height, depth_multiplier, stride,
      dilation_width_factor, dilation_height_factor, padding_type);
}

// Tests parameters for the 3x3 filter kernel.
bool TryTestOneDepthwiseConv3x3Filter(
    const TestParam& test_param, ParamsSpecialization params_specialization) {
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
  if (test_param.forced_invocation != DepthwiseConvImplementation::kNone) {
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
      test_param, params_specialization, batch, input_depth, input_width,
      input_height, filter_width, filter_height, depth_multiplier, stride,
      dilation_width_factor, dilation_height_factor, padding_type);
}

// Tests with parameters suited to dot-product-NEON 3x3 filter kernels.
bool TryTestOneNeonDot3x3(const TestParam& test_param,
                          ParamsSpecialization params_specialization) {
  const CoverageExtension coverage_extension = static_cast<CoverageExtension>(
      UniformRandomInt(0, static_cast<int>(CoverageExtension::kNumOptions)));

  const int batch = 1;
  const int input_depth = test_param.test_depth_multiplier
                              ? 1
                              : 8 * ExponentialRandomPositiveInt(0.9f, 3, 50);
  const int input_width = coverage_extension == CoverageExtension::kLargeWidths
                              ? ExponentialRandomPositiveInt(0.9f, 50, 200)
                              : ExponentialRandomPositiveInt(0.9f, 20, 60);
  const int input_height =
      coverage_extension == CoverageExtension::kLargeHeights
          ? ExponentialRandomPositiveInt(0.9f, 50, 200)
          : ExponentialRandomPositiveInt(0.9f, 20, 60);
  const int filter_width = 3;
  const int filter_height = 3;
  const int depth_multiplier =
      test_param.test_depth_multiplier
          ? 8 * ExponentialRandomPositiveInt(0.2f, 1, 9)
          : 1;
  const int stride = test_param.test_stride ? 2 : 1;
  // We don't support dilations in the 3x3 filter.
  const int dilation_width_factor = 1;
  const int dilation_height_factor = 1;
  const auto padding_type =
      test_param.test_pad ? PaddingType::kSame : PaddingType::kValid;

  return TryTestDepthwiseConv(
      test_param, params_specialization, batch, input_depth, input_width,
      input_height, filter_width, filter_height, depth_multiplier, stride,
      dilation_width_factor, dilation_height_factor, padding_type);
}

void TestOneDepthwiseConv(DepthwiseConvImplementation forced_invocation,
                          DepthwiseConvOutputRounding output_rounding) {
  TestParam test_param;
  test_param.forced_invocation = forced_invocation;
  test_param.output_rounding = output_rounding;
  while (!TryTestOneDepthwiseConv(test_param, ParamsSpecialization::kNone)) {
  }
}

void TestOneDepthwiseConv3x3Filter(
    DepthwiseConvImplementation forced_invocation,
    DepthwiseConvOutputRounding output_rounding) {
  TestParam test_param;
  test_param.forced_invocation = forced_invocation;
  test_param.output_rounding = output_rounding;
  while (!TryTestOneDepthwiseConv3x3Filter(test_param,
                                           ParamsSpecialization::kNone)) {
  }
}

void TestOneNeonDot3x3(const TestParam& test_param) {
#if defined(__aarch64__) && !defined(GOOGLE_L4T) && defined(__ANDROID__) && \
    defined(__clang__)
  CpuBackendContext backend_context;
  ruy::Context* ruy_context = backend_context.ruy_context();
  const auto ruy_paths = ruy_context != nullptr
                             ? ruy_context->GetRuntimeEnabledPaths()
                             : ruy::Path::kNone;
  const bool has_dot_product_instructions =
      (ruy_paths & ruy::Path::kNeonDotprod) != ruy::Path::kNone;
  if (test_param.forced_invocation ==
          DepthwiseConvImplementation::kUseNeon3x3DotProduct &&
      !has_dot_product_instructions) {
    return;
  }
#endif

  while (!TryTestOneNeonDot3x3(test_param, ParamsSpecialization::kSymmetric)) {
  }
}

TEST(TestDepthwiseConv, TestDepthwiseConv) {
  const int kTestsToRun = 10 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv(DepthwiseConvImplementation::kNone,
                         DepthwiseConvOutputRounding::kAwayFromZero);
  }
}

// Run basic coverage test against the generic kernel.
TEST(TestDepthwiseConv, TestGenericKernel) {
  const int kTestsToRun = 10 * 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv(DepthwiseConvImplementation::kUseGenericKernel,
                         DepthwiseConvOutputRounding::kAwayFromZero);
  }
}

#if defined(__aarch64__) && !defined(GOOGLE_L4T)
TEST(TestDepthwiseConv, TestNeon3x3FilterAway) {
  const int kTestsToRun = 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv3x3Filter(DepthwiseConvImplementation::kUseNeon3x3,
                                  DepthwiseConvOutputRounding::kAwayFromZero);
  }
}

TEST(TestDepthwiseConv, TestNeon3x3FilterUpward) {
  const int kTestsToRun = 1000;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv3x3Filter(DepthwiseConvImplementation::kUseNeon3x3,
                                  DepthwiseConvOutputRounding::kUpward);
  }
}
#endif

// While 3x3 coverage tests are primarily targeted at specialized kernels, we
// also run it against the generic kernel.
TEST(TestDepthwiseConv, TestGenericKernel3x3Filter) {
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    TestOneDepthwiseConv3x3Filter(
        DepthwiseConvImplementation::kUseGenericKernel,
        DepthwiseConvOutputRounding::kAwayFromZero);
  }
}

class DepthwiseConvTest : public ::testing::TestWithParam<TestParamTuple> {};

TEST_P(DepthwiseConvTest, NeonDot3x3) {
  const TestParam param(GetParam());
  for (int i = 0; i < param.tests_to_run; i++) {
    TestOneNeonDot3x3(param);
  }
}

#if defined(__aarch64__) && !defined(GOOGLE_L4T)
INSTANTIATE_TEST_SUITE_P(
    Neon3x3KernelAway, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::kUseNeon3x3),  // forced_invocation
        Values(1000),                                      // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),     // quantization_type
        Bool(),                                            // test_stride
        Values(false),                                     // test_pad
        Values(false),  // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kAwayFromZero),  // output_rounding
        Values(1),                                           // num_threads
        Values(false)                                        // loose_tolerance
        ),
    TestParam::TestNameSuffix);

INSTANTIATE_TEST_SUITE_P(
    Neon3x3KernelUpward, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::kUseNeon3x3),  // forced_invocation
        Values(1000),                                      // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),     // quantization_type
        Bool(),                                            // test_stride
        Values(false),                                     // test_pad
        Values(false),                                 // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kUpward),  // output_rounding
        Values(1),                                     // num_threads
        Values(false)                                  // loose_tolerance
        ),
    TestParam::TestNameSuffix);
#endif  // __aarch64__ && !GOOGLE_L4T

// While 3x3 coverage tests are primarily targeted at specialized kernels, we
// also run it against the generic kernel.
INSTANTIATE_TEST_SUITE_P(
    GenericKernel, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::
                   kUseGenericKernel),                  // forced_invocation
        Values(100),                                    // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),  // quantization_type
        Bool(),                                         // test_stride
        Bool(),                                         // test_pad
        Bool(),                                         // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kAwayFromZero),  // output_rounding
        Values(1),                                           // num_threads
        Values(false)                                        // loose_tolerance
        ),
    TestParam::TestNameSuffix);

INSTANTIATE_TEST_SUITE_P(
    CModel, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::
                   kUseCModel3x3DotProduct),            // forced_invocation
        Values(1000),                                   // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),  // quantization_type
        Bool(),                                         // test_stride
        Bool(),                                         // test_pad
        Bool(),                                         // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kUpward),   // output_rounding
        Values(1),                                      // num_threads
        Values(false)                                   // loose_tolerance
        ),
    TestParam::TestNameSuffix);

INSTANTIATE_TEST_SUITE_P(
    Unwound, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::
                   kUseUnwound3x3DotProduct),           // forced_invocation
        Values(1000),                                   // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),  // quantization_type
        Bool(),                                         // test_stride
        Bool(),                                         // test_pad
        Bool(),                                         // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kUpward),   // output_rounding
        Values(1),                                      // num_threads
        Values(false)                                   // loose_tolerance
        ),
    TestParam::TestNameSuffix);

#if defined(USE_NEON)
// Intrinsics tests are run in emulation mode (such as for dot-product
// instructions) unless the tests are built specifically with dot-product
// instructions enabled.
INSTANTIATE_TEST_SUITE_P(
    Intrinsics, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::
                   kUseIntrinsics3x3DotProduct),        // forced_invocation
        Values(1000),                                   // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),  // quantization_type
        Bool(),                                         // test_stride
        Bool(),                                         // test_pad
        Bool(),                                         // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kUpward),   // output_rounding
        Values(1),                                      // num_threads
        Values(kLooseIntrinsicsTolerance)               // loose_tolerance
        ),
    TestParam::TestNameSuffix);
#endif

#if defined(__aarch64__) && !defined(GOOGLE_L4T) && defined(__ANDROID__) && \
    defined(__clang__)
INSTANTIATE_TEST_SUITE_P(
    NeonAsm, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::
                   kUseNeon3x3DotProduct),              // forced_invocation
        Values(1000),                                   // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),  // quantization_type
        Bool(),                                         // test_stride
        Bool(),                                         // test_pad
        Bool(),                                         // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kUpward),   // output_rounding
        Values(1),                                      // num_threads
        Values(false)                                   // loose_tolerance
        ),
    TestParam::TestNameSuffix);

// Apply the 3x3 tests through the dispatch.
// Also test multi-threading. This assumes upward rounding.
INSTANTIATE_TEST_SUITE_P(
    Dispatch3x3, DepthwiseConvTest,
    testing::Combine(
        Values(DepthwiseConvImplementation::kNone),     // forced_invocation
        Values(1000),                                   // tests_to_run
        Values(QuantizationType::kNonPerChannelUint8),  // quantization_type
        Bool(),                                         // test_stride
        Bool(),                                         // test_pad
        Bool(),                                         // test_depth_multiplier
        Values(DepthwiseConvOutputRounding::kUpward),   // output_rounding
        Values(4),                                      // num_threads
        Values(false)                                   // loose_tolerance
        ),
    TestParam::TestNameSuffix);
#endif

}  // namespace
}  // namespace tflite
