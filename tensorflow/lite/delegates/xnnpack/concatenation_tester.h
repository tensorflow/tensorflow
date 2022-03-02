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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_CONCATENATION_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_CONCATENATION_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class ConcatenationTester {
 public:
  ConcatenationTester() = default;
  ConcatenationTester(const ConcatenationTester&) = delete;
  ConcatenationTester& operator=(const ConcatenationTester&) = delete;

  inline ConcatenationTester& Axis(int axis) {
    axis_ = axis;
    return *this;
  }

  inline ConcatenationTester& Input1Shape(const std::vector<int32_t>& shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input1_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const int Axis() const { return axis_; }

  inline const std::vector<int32_t>& Input1Shape() const {
    return input1_shape_;
  }

  inline ConcatenationTester& Input2Shape(const std::vector<int32_t>& shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input2_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& Input2Shape() const {
    return input2_shape_;
  }

  std::vector<int32_t> OutputShape() const {
    std::vector<int32_t> output_shape = Input1Shape();
    int concat_axis = Axis() < 0 ? Axis() + Input1Shape().size() : Axis();
    output_shape[concat_axis] += Input2Shape()[concat_axis];
    return output_shape;
  }

  template <typename T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;
  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  int axis_;
  std::vector<int32_t> input1_shape_;
  std::vector<int32_t> input2_shape_;
  std::vector<int32_t> output_shape_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_CONCATENATION_TESTER_H_
