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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_BINARY_ELEMENTWISE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_BINARY_ELEMENTWISE_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class QuantizedBinaryElementwiseTester {
 public:
  QuantizedBinaryElementwiseTester() = default;
  QuantizedBinaryElementwiseTester(const QuantizedBinaryElementwiseTester&) =
      delete;
  QuantizedBinaryElementwiseTester& operator=(
      const QuantizedBinaryElementwiseTester&) = delete;

  inline QuantizedBinaryElementwiseTester& Input1Shape(
      std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input1_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& Input1Shape() const {
    return input1_shape_;
  }

  inline QuantizedBinaryElementwiseTester& Input2Shape(
      std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input2_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& Input2Shape() const {
    return input2_shape_;
  }

  std::vector<int32_t> OutputShape() const;

  inline QuantizedBinaryElementwiseTester& Input1Static(bool is_static) {
    input1_static_ = is_static;
    return *this;
  }

  inline bool Input1Static() const { return input1_static_; }

  inline QuantizedBinaryElementwiseTester& Input2Static(bool is_static) {
    input2_static_ = is_static;
    return *this;
  }

  inline bool Input2Static() const { return input2_static_; }

  inline QuantizedBinaryElementwiseTester& Input1ZeroPoint(
      int32_t input1_zero_point) {
    input1_zero_point_ = input1_zero_point;
    return *this;
  }

  inline int32_t Input1ZeroPoint() const { return input1_zero_point_; }

  inline QuantizedBinaryElementwiseTester& Input2ZeroPoint(
      int32_t input2_zero_point) {
    input2_zero_point_ = input2_zero_point;
    return *this;
  }

  inline int32_t Input2ZeroPoint() const { return input2_zero_point_; }

  inline QuantizedBinaryElementwiseTester& OutputZeroPoint(
      int32_t output_zero_point) {
    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const { return output_zero_point_; }

  inline QuantizedBinaryElementwiseTester& Input1Scale(float input1_scale) {
    input1_scale_ = input1_scale;
    return *this;
  }

  inline float Input1Scale() const { return input1_scale_; }

  inline QuantizedBinaryElementwiseTester& Input2Scale(float input2_scale) {
    input2_scale_ = input2_scale;
    return *this;
  }

  inline float Input2Scale() const { return input2_scale_; }

  inline QuantizedBinaryElementwiseTester& OutputScale(float output_scale) {
    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const { return output_scale_; }

  inline QuantizedBinaryElementwiseTester& Unsigned(bool is_unsigned) {
    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const { return unsigned_; }

  inline QuantizedBinaryElementwiseTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline QuantizedBinaryElementwiseTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline QuantizedBinaryElementwiseTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(tflite::BuiltinOperator binary_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator binary_op) const;

  inline ::tflite::ActivationFunctionType Activation() const {
    return activation_;
  }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input1_shape_;
  std::vector<int32_t> input2_shape_;
  bool input1_static_ = false;
  bool input2_static_ = false;
  int32_t input1_zero_point_ = 0;
  int32_t input2_zero_point_ = 0;
  int32_t output_zero_point_ = 0;
  float input1_scale_ = 0.75f;
  float input2_scale_ = 1.0f;
  float output_scale_ = 1.75f;
  bool unsigned_ = false;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_BINARY_ELEMENTWISE_TESTER_H_
