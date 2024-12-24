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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/reshape_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

TEST(Reshape, 4DShapeAsInput) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 4DShapeAsParam) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(false)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 3DShapeAsInput) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 3DShapeAsParam) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(false)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 2DShapeAsInput) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{{shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 2DShapeAsParam) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{{shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(false)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 1DShapeAsInput) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng()});

  ReshapeTester()
      .InputShape(shape)
      .OutputShape(shape)
      .OutputShapeAsInput(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 1DShapeAsParam) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng()});

  ReshapeTester()
      .InputShape(shape)
      .OutputShape(shape)
      .OutputShapeAsInput(false)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, 0D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  ReshapeTester()
      .InputShape(std::vector<int32_t>())
      .OutputShape(std::vector<int32_t>())
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Reshape, UnsupportedOutputRank) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  std::vector<int32_t> input_shape;
  std::generate_n(std::back_inserter(input_shape), XNN_MAX_TENSOR_DIMS,
                  shape_rng);

  // Construct an output shape greater than XNN_MAX_TENSOR_DIMS. This will
  // prevent this node from being delegated to XNNPACK.
  std::vector<int32_t> output_shape = input_shape;
  output_shape.push_back(1);
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
