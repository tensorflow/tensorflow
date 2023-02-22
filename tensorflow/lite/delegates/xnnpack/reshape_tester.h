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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_RESHAPE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_RESHAPE_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class ReshapeTester {
 public:
  ReshapeTester() = default;
  ReshapeTester(const ReshapeTester&) = delete;
  ReshapeTester& operator=(const ReshapeTester&) = delete;

  inline ReshapeTester& InputShape(const std::vector<int32_t>& input_shape) {
    for (int32_t input_dim : input_shape) {
      EXPECT_GT(input_dim, 0);
    }
    input_shape_ = std::vector<int32_t>(input_shape.begin(), input_shape.end());
    input_size_ = ReshapeTester::ComputeSize(input_shape);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline ReshapeTester& OutputShape(const std::vector<int32_t>& output_shape) {
    for (int32_t output_dim : output_shape) {
      EXPECT_GT(output_dim, 0);
    }
    output_shape_ =
        std::vector<int32_t>(output_shape.begin(), output_shape.end());
    output_size_ = ReshapeTester::ComputeSize(output_shape);
    return *this;
  }

  inline const std::vector<int32_t>& OutputShape() const {
    return output_shape_;
  }

  inline int32_t InputSize() const { return input_size_; }

  inline int32_t OutputSize() const { return output_size_; }

  inline ReshapeTester& OutputShapeAsInput(bool shape_as_input) {
    shape_as_input_ = shape_as_input;
    return *this;
  }

  inline bool OutputShapeAsInput() const { return shape_as_input_; }

  template <class T>
  void Test(TensorType tensor_type, Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> output_shape_;
  int32_t input_size_ = 1;
  int32_t output_size_ = 1;
  bool shape_as_input_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_RESHAPE_TESTER_H_
