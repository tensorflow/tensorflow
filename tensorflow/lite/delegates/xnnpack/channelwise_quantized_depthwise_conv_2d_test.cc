/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/quantized_depthwise_conv_2d_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(ChannelwiseQuantizedDepthwiseConv2D, 1x1) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(1)
      .KernelWidth(1)
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, 2x2) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(2)
      .KernelWidth(2)
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, 3x3) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(3)
      .KernelWidth(3)
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, 3x3Stride2) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(3)
      .KernelWidth(3)
      .StrideHeight(2)
      .StrideWidth(2)
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, 5x5) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(3)
      .KernelWidth(3)
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, 5x5Stride2) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(3)
      .KernelWidth(3)
      .StrideHeight(2)
      .StrideWidth(2)
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, SmallKernelWithSamePadding) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 7), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, SmallKernelWithValidPadding) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 7), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .ValidPadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, StrideWithSamePadding) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, StrideWithValidPadding) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .ValidPadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, DilationWithSamePadding) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto dilation_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .DilationHeight(dilation_rng())
      .DilationWidth(dilation_rng())
      .SamePadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, DilationWithValidPadding) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto dilation_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .DilationHeight(dilation_rng())
      .DilationWidth(dilation_rng())
      .ValidPadding()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, DepthMultiplier) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));
  auto multiplier_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 8), std::ref(rng));

  const int32_t num_input_channels = channel_rng();
  const int32_t depth_multiplier = multiplier_rng();
  const int32_t num_output_channels = num_input_channels * depth_multiplier;
  std::vector<float> filter_scales;
  filter_scales.reserve(num_output_channels);
  std::generate_n(std::back_inserter(filter_scales), num_output_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_input_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .DepthMultiplier(depth_multiplier)
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, ReluActivation) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(0)
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .ReluActivation()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, Relu6Activation) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(0)
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .Relu6Activation()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, ReluMinus1To1Activation) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .ReluMinus1To1Activation()
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, WeightsCache) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);
  delegate_options.weights_cache = weights_cache.get();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .WeightsCache(weights_cache.get())
      .Test(xnnpack_delegate.get());
}

TEST(ChannelwiseQuantizedDepthwiseConv2D, TransientIndirectionBuffer) {
  TfLiteXNNPackDelegateOptions xnnpack_options =
      TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_options.num_threads = 2;
  xnnpack_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&xnnpack_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto scale_rng = std::bind(
      std::uniform_real_distribution<float>(0.25f, 1.25f), std::ref(rng));
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  const int32_t num_channels = channel_rng();
  std::vector<float> filter_scales;
  filter_scales.reserve(num_channels);
  std::generate_n(std::back_inserter(filter_scales), num_channels,
                  std::ref(scale_rng));

  QuantizedDepthwiseConv2DTester()
      .InputZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .KernelScales(filter_scales)
      .BatchSize(batch_rng())
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .InputChannels(num_channels)
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
