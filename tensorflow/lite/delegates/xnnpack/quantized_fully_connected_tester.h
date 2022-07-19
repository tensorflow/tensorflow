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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_FULLY_CONNECTED_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_FULLY_CONNECTED_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class QuantizedFullyConnectedTester {
 public:
  QuantizedFullyConnectedTester() = default;
  QuantizedFullyConnectedTester(const QuantizedFullyConnectedTester&) = delete;
  QuantizedFullyConnectedTester& operator=(
      const QuantizedFullyConnectedTester&) = delete;

  inline QuantizedFullyConnectedTester& InputShape(
      std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input_size_ = ComputeSize(input_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline int32_t InputSize() const { return input_size_; }

  inline QuantizedFullyConnectedTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline QuantizedFullyConnectedTester& OutputChannels(
      int32_t output_channels) {
    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const { return output_channels_; }

  std::vector<int32_t> OutputShape() const;

  inline QuantizedFullyConnectedTester& InputZeroPoint(
      int32_t input_zero_point) {
    input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int32_t InputZeroPoint() const { return input_zero_point_; }

  inline QuantizedFullyConnectedTester& FilterZeroPoint(
      int32_t filter_zero_point) {
    filter_zero_point_ = filter_zero_point;
    return *this;
  }

  inline int32_t FilterZeroPoint() const { return filter_zero_point_; }

  inline QuantizedFullyConnectedTester& OutputZeroPoint(
      int32_t output_zero_point) {
    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const { return output_zero_point_; }

  inline QuantizedFullyConnectedTester& InputScale(float input_scale) {
    input_scale_ = input_scale;
    return *this;
  }

  inline float InputScale() const { return input_scale_; }

  inline QuantizedFullyConnectedTester& FilterScale(float filter_scale) {
    filter_scale_ = filter_scale;
    return *this;
  }

  inline float FilterScale() const { return filter_scale_; }

  inline QuantizedFullyConnectedTester& OutputScale(float output_scale) {
    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const { return output_scale_; }

  inline QuantizedFullyConnectedTester& KeepDims(bool keep_dims) {
    keep_dims_ = keep_dims;
    return *this;
  }

  inline bool KeepDims() const { return keep_dims_; }

  inline bool Unsigned() const { return filter_zero_point_ != 0; }

  inline QuantizedFullyConnectedTester& NoBias() {
    has_bias_ = false;
    return *this;
  }

  inline QuantizedFullyConnectedTester& WithBias() {
    has_bias_ = true;
    return *this;
  }

  inline QuantizedFullyConnectedTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline QuantizedFullyConnectedTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline QuantizedFullyConnectedTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline QuantizedFullyConnectedTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline bool HasBias() const { return has_bias_; }

  inline ::tflite::ActivationFunctionType Activation() const {
    return activation_;
  }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  int32_t input_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;
  int32_t input_zero_point_ = 0;
  int32_t filter_zero_point_ = 0;
  int32_t output_zero_point_ = 0;
  float input_scale_ = 0.8f;
  float filter_scale_ = 0.75f;
  float output_scale_ = 1.5f;
  bool keep_dims_ = false;
  bool has_bias_ = true;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_FULLY_CONNECTED_TESTER_H_
