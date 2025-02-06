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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_DYNAMICALLY_QUANTIZED_FULLY_CONNECTED_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_DYNAMICALLY_QUANTIZED_FULLY_CONNECTED_TESTER_H_

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

enum class WeightsType {
  kChannelWiseQuantizedInt4,
  kChannelWiseQuantizedInt8,
  kTensorWiseQuantizedInt8,
};

class DynamicallyQuantizedFullyConnectedTester {
 public:
  DynamicallyQuantizedFullyConnectedTester() = default;
  DynamicallyQuantizedFullyConnectedTester(
      const DynamicallyQuantizedFullyConnectedTester&) = delete;
  DynamicallyQuantizedFullyConnectedTester& operator=(
      const DynamicallyQuantizedFullyConnectedTester&) = delete;

  inline DynamicallyQuantizedFullyConnectedTester& InputShape(
      std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input_size_ = ComputeSize(input_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline int32_t InputSize() const { return input_size_; }

  inline DynamicallyQuantizedFullyConnectedTester& InputChannels(
      int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline DynamicallyQuantizedFullyConnectedTester& OutputChannels(
      int32_t output_channels) {
    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const { return output_channels_; }

  std::vector<int32_t> OutputShape() const;

  inline DynamicallyQuantizedFullyConnectedTester& WeightsType(
      WeightsType weights_type) {
    weights_type_ = weights_type;
    return *this;
  }

  inline DynamicallyQuantizedFullyConnectedTester& FilterZeroPoint(
      int32_t filter_zero_point) {
    filter_zero_point_ = filter_zero_point;
    return *this;
  }

  inline int32_t FilterZeroPoint() const { return filter_zero_point_; }

  inline DynamicallyQuantizedFullyConnectedTester& FilterScale(
      float filter_scale) {
    filter_scale_ = filter_scale;
    return *this;
  }

  inline float FilterScale() const { return filter_scale_; }

  inline DynamicallyQuantizedFullyConnectedTester& KeepDims(bool keep_dims) {
    keep_dims_ = keep_dims;
    return *this;
  }

  inline bool KeepDims() const { return keep_dims_; }

  inline DynamicallyQuantizedFullyConnectedTester& NoBias() {
    has_bias_ = false;
    return *this;
  }

  inline DynamicallyQuantizedFullyConnectedTester& WithBias() {
    has_bias_ = true;
    return *this;
  }

  inline DynamicallyQuantizedFullyConnectedTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline DynamicallyQuantizedFullyConnectedTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline DynamicallyQuantizedFullyConnectedTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline DynamicallyQuantizedFullyConnectedTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline bool HasBias() const { return has_bias_; }

  inline ::tflite::ActivationFunctionType Activation() const {
    return activation_;
  }

  inline enum WeightsType WeightsType() const { return weights_type_; }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  int32_t input_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;

  enum WeightsType weights_type_ = WeightsType::kTensorWiseQuantizedInt8;
  int32_t filter_zero_point_ = 0;
  float filter_scale_ = 0.75f;
  bool keep_dims_ = false;
  bool has_bias_ = true;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_DYNAMICALLY_QUANTIZED_FULLY_CONNECTED_TESTER_H_
