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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_BINARY_ELEMENTWISE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_BINARY_ELEMENTWISE_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class BinaryElementwiseTester {
 public:
  BinaryElementwiseTester() = default;
  BinaryElementwiseTester(const BinaryElementwiseTester&) = delete;
  BinaryElementwiseTester& operator=(const BinaryElementwiseTester&) = delete;

  inline BinaryElementwiseTester& Input1Shape(
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

  inline BinaryElementwiseTester& Input2Shape(
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

  inline BinaryElementwiseTester& Input1Static(bool is_static) {
    input1_static_ = is_static;
    return *this;
  }

  inline bool Input1Static() const { return input1_static_; }

  inline BinaryElementwiseTester& Input2Static(bool is_static) {
    input2_static_ = is_static;
    return *this;
  }

  inline bool Input2Static() const { return input2_static_; }

  inline BinaryElementwiseTester& FP16Weights() {
    fp16_weights_ = true;
    return *this;
  }

  inline bool FP16Weights() const { return fp16_weights_; }

  inline BinaryElementwiseTester& SparseWeights() {
    sparse_weights_ = true;
    return *this;
  }

  inline bool SparseWeights() const { return sparse_weights_; }

  inline BinaryElementwiseTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline BinaryElementwiseTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline BinaryElementwiseTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline BinaryElementwiseTester& TanhActivation() {
    activation_ = ::tflite::ActivationFunctionType_TANH;
    return *this;
  }

  inline BinaryElementwiseTester& SignBitActivation() {
    activation_ = ::tflite::ActivationFunctionType_SIGN_BIT;
    return *this;
  }

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
  bool fp16_weights_ = false;
  bool sparse_weights_ = false;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_BINARY_ELEMENTWISE_TESTER_H_
