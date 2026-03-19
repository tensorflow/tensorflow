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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_POOL_2D_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_POOL_2D_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class QuantizedPool2DTester {
 public:
  QuantizedPool2DTester() = default;
  QuantizedPool2DTester(const QuantizedPool2DTester&) = delete;
  QuantizedPool2DTester& operator=(const QuantizedPool2DTester&) = delete;

  inline QuantizedPool2DTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline QuantizedPool2DTester& Channels(int32_t channels) {
    EXPECT_GT(channels, 0);
    channels_ = channels;
    return *this;
  }

  inline int32_t Channels() const { return channels_; }

  inline QuantizedPool2DTester& InputHeight(int32_t input_height) {
    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const { return input_height_; }

  inline QuantizedPool2DTester& InputWidth(int32_t input_width) {
    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const { return input_width_; }

  inline int32_t OutputWidth() const {
    if (Padding() == ::tflite::Padding_SAME) {
      return (InputWidth() - 1) / StrideWidth() + 1;
    } else {
      return (InputWidth() - PoolingWidth()) / StrideWidth() + 1;
    }
  }

  inline int32_t OutputHeight() const {
    if (Padding() == ::tflite::Padding_SAME) {
      return (InputHeight() - 1) / StrideHeight() + 1;
    } else {
      return (InputHeight() - PoolingHeight()) / StrideHeight() + 1;
    }
  }

  inline QuantizedPool2DTester& PoolingHeight(int32_t pooling_height) {
    EXPECT_GT(pooling_height, 0);
    pooling_height_ = pooling_height;
    return *this;
  }

  inline int32_t PoolingHeight() const { return pooling_height_; }

  inline QuantizedPool2DTester& PoolingWidth(int32_t pooling_width) {
    EXPECT_GT(pooling_width, 0);
    pooling_width_ = pooling_width;
    return *this;
  }

  inline int32_t PoolingWidth() const { return pooling_width_; }

  inline QuantizedPool2DTester& StrideHeight(int32_t stride_height) {
    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const { return stride_height_; }

  inline QuantizedPool2DTester& StrideWidth(int32_t stride_width) {
    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const { return stride_width_; }

  inline QuantizedPool2DTester& SamePadding() {
    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline QuantizedPool2DTester& ValidPadding() {
    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline QuantizedPool2DTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline QuantizedPool2DTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline QuantizedPool2DTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline QuantizedPool2DTester& TanhActivation() {
    activation_ = ::tflite::ActivationFunctionType_TANH;
    return *this;
  }

  inline QuantizedPool2DTester& SignBitActivation() {
    activation_ = ::tflite::ActivationFunctionType_SIGN_BIT;
    return *this;
  }

  inline QuantizedPool2DTester& ZeroPoint(int32_t zero_point) {
    zero_point_ = zero_point;
    return *this;
  }

  inline int32_t ZeroPoint() const { return zero_point_; }

  inline QuantizedPool2DTester& Scale(float scale) {
    scale_ = scale;
    return *this;
  }

  inline float Scale() const { return scale_; }

  inline QuantizedPool2DTester& Unsigned(bool is_unsigned) {
    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const { return unsigned_; }

  template <class T>
  void Test(tflite::BuiltinOperator pool_op, Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(tflite::BuiltinOperator pool_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator pool_op) const;

  inline ::tflite::Padding Padding() const { return padding_; }

  inline ::tflite::ActivationFunctionType Activation() const {
    return activation_;
  }

  int32_t batch_size_ = 1;
  int32_t channels_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t pooling_height_ = 1;
  int32_t pooling_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
  int32_t zero_point_ = 7;
  float scale_ = 0.5f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_POOL_2D_TESTER_H_
