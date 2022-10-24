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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_REDUCE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_REDUCE_TESTER_H_

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class QuantizedReduceTester {
 public:
  QuantizedReduceTester() = default;
  QuantizedReduceTester(const QuantizedReduceTester&) = delete;
  QuantizedReduceTester& operator=(const QuantizedReduceTester&) = delete;

  inline QuantizedReduceTester& InputShape(
      std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input_size_ = QuantizedReduceTester::ComputeSize(input_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline int32_t InputSize() const { return input_size_; }

  inline QuantizedReduceTester& Axes(std::initializer_list<int32_t> axes) {
    for (auto it = axes.begin(); it != axes.end(); ++it) {
      EXPECT_GE(*it, 0);
    }
    axes_ = std::vector<int32_t>(axes.begin(), axes.end());
    return *this;
  }

  inline const std::vector<int32_t>& Axes() const { return axes_; }

  inline QuantizedReduceTester& KeepDims(bool keep_dims) {
    keep_dims_ = keep_dims;
    return *this;
  }

  inline bool KeepDims() const { return keep_dims_; }

  inline std::vector<int32_t> OutputShape() const {
    std::vector<int32_t> output_shape;
    output_shape.reserve(InputShape().size());
    std::unordered_set<int32_t> axes_set(Axes().cbegin(), Axes().cend());
    for (int32_t i = 0; i < InputShape().size(); i++) {
      if (axes_set.count(i) != 0) {
        if (KeepDims()) {
          output_shape.push_back(1);
        }
      } else {
        output_shape.push_back(InputShape()[i]);
      }
    }
    return output_shape;
  }

  inline int32_t OutputSize() const {
    int32_t output_size = 1;
    std::unordered_set<int32_t> axes_set(Axes().cbegin(), Axes().cend());
    for (int32_t i = 0; i < InputShape().size(); i++) {
      if (axes_set.count(i) == 0) {
        output_size *= InputShape()[i];
      }
    }
    return output_size;
  }

  inline QuantizedReduceTester& InputZeroPoint(int32_t input_zero_point) {
    input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int32_t InputZeroPoint() const { return input_zero_point_; }

  inline QuantizedReduceTester& OutputZeroPoint(int32_t output_zero_point) {
    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const { return output_zero_point_; }

  inline QuantizedReduceTester& InputScale(float input_scale) {
    input_scale_ = input_scale;
    return *this;
  }

  inline float InputScale() const { return input_scale_; }

  inline QuantizedReduceTester& OutputScale(float output_scale) {
    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const { return output_scale_; }

  inline QuantizedReduceTester& Unsigned(bool is_unsigned) {
    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const { return unsigned_; }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(tflite::BuiltinOperator reduce_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator reduce_op) const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> axes_;
  int32_t input_size_;
  bool keep_dims_ = true;
  int32_t input_zero_point_ = 1;
  int32_t output_zero_point_ = 2;
  float input_scale_ = 1.25f;
  float output_scale_ = 0.75f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_REDUCE_TESTER_H_
