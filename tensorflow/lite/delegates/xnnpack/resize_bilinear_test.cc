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
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/resize_bilinear_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(ResizeBilinear, AlignCenters) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  ResizeBilinearTester()
      .HalfPixelCenters(true)
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(ResizeBilinear, AlignCentersTF1X) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  ResizeBilinearTester()
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(ResizeBilinear, AlignCorners) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  ResizeBilinearTester()
      .AlignCorners(true)
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(ResizeBilinear, MultiThreading) {
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

  ResizeBilinearTester()
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(ResizeBilinear, TransientIndirectionBuffer) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  ResizeBilinearTester()
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
