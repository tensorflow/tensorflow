/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/fingerprint_test_helpers.h"
#include "tensorflow/lite/delegates/xnnpack/fully_connected_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

struct FullyConnectedTest : public DelegateTest {};

TEST_F(FullyConnectedTest, 1D) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 1DKeepDims) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 2D) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 2DKeepDims) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 3D) {
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

  FullyConnectedTester tester;
  tester.InputShape({batch, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 3DReshape) {
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

  FullyConnectedTester tester;
  tester.InputShape({batch, width, input_channels})
      .InputChannels(width * input_channels)
      .OutputChannels(output_channels)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 3DKeepDims) {
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

  FullyConnectedTester tester;
  tester.InputShape({batch, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 4D) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, height, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, 4DKeepDims) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, height, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, NoBias) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .NoBias()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, FP16Weights) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .FP16Weights()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, FP16WeightsNoBias) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .FP16Weights()
      .NoBias()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, DynamicWeights) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .DynamicWeights()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, DynamicWeightsNoBias) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .DynamicWeights()
      .NoBias()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, DynamicBias) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .DynamicBias()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, DynamicWeightsAndBias) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .DynamicWeights()
      .DynamicBias()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, TensorWiseQuantizedInt8Weights) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .TensorWiseQuantizedInt8Weights()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, TensorWiseQuantizedInt8WeightsNoBias) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .TensorWiseQuantizedInt8Weights()
      .NoBias()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, ChannelWiseQuantizedInt8Weights) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ChannelWiseQuantizedInt8Weights()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, ChannelWiseQuantizedInt8WeightsNoBias) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ChannelWiseQuantizedInt8Weights()
      .NoBias()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, ReluActivation) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReluActivation()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, Relu6Activation) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .Relu6Activation()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, ReluMinus1To1Activation) {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReluMinus1To1Activation()
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  UseCustomDelegate(delegate_options);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

TEST_F(FullyConnectedTest, WeightsCache) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);
  delegate_options.weights_cache = weights_cache.get();
  UseCustomDelegate(delegate_options);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester tester;
  tester.InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .WeightsCache(weights_cache.get())
      .ReuseGeneratedModel(true);
  tester.Test(xnnpack_delegate.get());
  // Second run to test cache lookup runs.
  tester.Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
