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
#include "tensorflow/lite/delegates/xnnpack/pad_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(Pad, Full4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({pad_rng(), pad_rng(), pad_rng(), pad_rng()})
      .InputPostPaddings({pad_rng(), pad_rng(), pad_rng(), pad_rng()})
      .InputShape({shape_rng(), shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Batch4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({pad_rng(), 0, 0, 0})
      .InputPostPaddings({pad_rng(), 0, 0, 0})
      .InputShape({shape_rng(), shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, HeightAndWidth4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({0, pad_rng(), pad_rng(), 0})
      .InputPostPaddings({0, pad_rng(), pad_rng(), 0})
      .InputShape({shape_rng(), shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Channels4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({0, 0, 0, pad_rng()})
      .InputPostPaddings({0, 0, 0, pad_rng()})
      .InputShape({shape_rng(), shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Full3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({pad_rng(), pad_rng(), pad_rng()})
      .InputPostPaddings({pad_rng(), pad_rng(), pad_rng()})
      .InputShape({shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Batch3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({pad_rng(), 0, 0})
      .InputPostPaddings({pad_rng(), 0, 0})
      .InputShape({shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Width3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({0, pad_rng(), 0})
      .InputPostPaddings({0, pad_rng(), 0})
      .InputShape({shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Channels3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({0, 0, pad_rng()})
      .InputPostPaddings({0, 0, pad_rng()})
      .InputShape({shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Full2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({pad_rng(), pad_rng()})
      .InputPostPaddings({pad_rng(), pad_rng()})
      .InputShape({shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Batch2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({pad_rng(), 0})
      .InputPostPaddings({pad_rng(), 0})
      .InputShape({shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, Channels2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({0, pad_rng()})
      .InputPostPaddings({0, pad_rng()})
      .InputShape({shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, 1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({pad_rng(), pad_rng()})
      .InputPostPaddings({pad_rng(), pad_rng()})
      .InputShape({shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

TEST(Pad, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto pad_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 3), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  PadTester()
      .InputPrePaddings({0, 0, 0, pad_rng()})
      .InputPostPaddings({0, 0, 0, pad_rng()})
      .InputShape({shape_rng(), shape_rng(), shape_rng(), shape_rng()})
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
