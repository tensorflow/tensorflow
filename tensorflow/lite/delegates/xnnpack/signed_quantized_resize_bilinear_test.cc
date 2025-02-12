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
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(SignedQuantizedResizeBilinear, AlignCenters) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  QuantizedResizeBilinearTester()
      .HalfPixelCenters(true)
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedResizeBilinear, AlignCentersTF1X) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  QuantizedResizeBilinearTester()
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedResizeBilinear, AlignCorners) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  QuantizedResizeBilinearTester()
      .AlignCorners(true)
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedResizeBilinear, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  QuantizedResizeBilinearTester()
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
