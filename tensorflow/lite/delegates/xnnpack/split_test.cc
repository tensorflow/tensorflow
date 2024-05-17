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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/split_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(Split, 1D_to_2_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng() * 2});

  for (int i = -1; i < 1; i++) {
    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(2)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 2D_to_2_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -2; i < 2; i++) {
    std::vector<int32_t> shape({shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 2;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(2)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 3D_to_2_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -3; i < 3; i++) {
    std::vector<int32_t> shape({shape_rng(), shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 2;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(2)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 4D_to_2_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -4; i < 4; i++) {
    std::vector<int32_t> shape(
        {shape_rng(), shape_rng(), shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 2;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(2)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 1D_to_3_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng() * 3});

  for (int i = -1; i < 1; i++) {
    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(3)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 2D_to_3_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -2; i < 2; i++) {
    std::vector<int32_t> shape({shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 3;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(3)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 3D_to_3_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -3; i < 3; i++) {
    std::vector<int32_t> shape({shape_rng(), shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 3;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(3)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 4D_to_3_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -4; i < 4; i++) {
    std::vector<int32_t> shape(
        {shape_rng(), shape_rng(), shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 3;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(3)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 1D_to_4_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng() * 4});

  for (int i = -1; i < 1; i++) {
    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(4)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 2D_to_4_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -2; i < 2; i++) {
    std::vector<int32_t> shape({shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 4;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(4)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 3D_to_4_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -3; i < 3; i++) {
    std::vector<int32_t> shape({shape_rng(), shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 4;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(4)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

TEST(Split, 4D_to_4_outputs) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto split_dim_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));

  for (int i = -4; i < 4; i++) {
    std::vector<int32_t> shape(
        {shape_rng(), shape_rng(), shape_rng(), shape_rng()});
    shape[i < 0 ? i + shape.size() : i] = split_dim_rng() * 4;

    // clang-format off
    SplitTester()
        .InputShape(shape)
        .SplitDimension(i)
        .NumSplits(4)
        .Test(TensorType_FLOAT32, xnnpack_delegate.get());
    // clang-format on
  }
}

}  // namespace xnnpack
}  // namespace tflite
