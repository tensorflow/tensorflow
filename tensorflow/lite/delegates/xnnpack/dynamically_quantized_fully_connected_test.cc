/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/dynamically_quantized_fully_connected_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

// Dummy class to use with parameterized test.
class DynamicallyQuantizedFullyConnectedTest
    : public testing::TestWithParam<WeightsType> {};

int GenInputChannels(const std::function<int()> &rng,
                     WeightsType weights_type) {
  switch (weights_type) {
    case WeightsType::kChannelWiseQuantizedInt8:
    case WeightsType::kTensorWiseQuantizedInt8:
      return rng();
    case WeightsType::kChannelWiseQuantizedInt4:
      // Int4 quantized kernels only support even number of channels.
      return (rng() / 2) * 2;
  }
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 1D) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 2D) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 2DKeepDims) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 3D) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 3DReshape) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .WeightsType(GetParam())
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 3DKeepDims) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 4D) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, height, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, 4DKeepDims) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, height, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, NoBias) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .NoBias()
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, ReluActivation) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReluActivation()
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, Relu6Activation) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .Relu6Activation()
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, ReluMinus1To1Activation) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReluMinus1To1Activation()
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

TEST_P(DynamicallyQuantizedFullyConnectedTest, WeightsCache) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
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
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  WeightsType weights_type = GetParam();
  const auto input_channels = GenInputChannels(channels_rng, weights_type);
  const auto output_channels = channels_rng();

  DynamicallyQuantizedFullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .WeightsCache(weights_cache.get())
      .WeightsType(weights_type)
      .Test(xnnpack_delegate.get());
}

// Returns a human readable string representation of the test parameter.
std::string TestParamToString(testing::TestParamInfo<WeightsType> param) {
  switch (param.param) {
    case WeightsType::kChannelWiseQuantizedInt4:
      return "ChannelWiseQuantizedInt4";
    case WeightsType::kChannelWiseQuantizedInt8:
      return "ChannelWiseQuantizedInt8";
    case WeightsType::kTensorWiseQuantizedInt8:
      return "TensorWiseQuantizedInt8";
    default:
      assert(false);
      return "???";
  }
}

INSTANTIATE_TEST_SUITE_P(
    DynamicallyQuantizedFullyConnectedTest,
    DynamicallyQuantizedFullyConnectedTest,
    testing::Values(WeightsType::kTensorWiseQuantizedInt8,
                    WeightsType::kChannelWiseQuantizedInt4,
                    WeightsType::kChannelWiseQuantizedInt8),
    TestParamToString);

}  // namespace xnnpack
}  // namespace tflite
