/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(SignedQuantizedDepthToSpace, SinglePixel) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  DepthToSpaceTester()
      .BatchSize(batch_rng())
      .InputHeight(1)
      .InputWidth(1)
      .OutputChannels(channel_rng())
      .BlockSize(block_rng())
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedDepthToSpace, SingleRow) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto width_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  DepthToSpaceTester()
      .BatchSize(batch_rng())
      .InputHeight(1)
      .InputWidth(width_rng())
      .OutputChannels(channel_rng())
      .BlockSize(block_rng())
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedDepthToSpace, SingleColumn) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto height_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  DepthToSpaceTester()
      .BatchSize(batch_rng())
      .InputHeight(height_rng())
      .InputWidth(1)
      .OutputChannels(channel_rng())
      .BlockSize(block_rng())
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedDepthToSpace, FullImage) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  DepthToSpaceTester()
      .BatchSize(batch_rng())
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputChannels(channel_rng())
      .BlockSize(block_rng())
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedDepthToSpace, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  DepthToSpaceTester()
      .BatchSize(batch_rng())
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputChannels(channel_rng())
      .BlockSize(block_rng())
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
