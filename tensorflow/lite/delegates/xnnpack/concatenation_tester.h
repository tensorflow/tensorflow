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
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

// Creates a new shape with the same dimensions as `shape`, except for the axis
// dimension, which will have the value `size`.
std::vector<int32_t> SameShapeDifferentAxis(std::vector<int32_t> shape,
                                            int axis, int32_t size);

class ConcatenationTester {
 public:
  ConcatenationTester() = default;
  ConcatenationTester(const ConcatenationTester&) = delete;
  ConcatenationTester& operator=(const ConcatenationTester&) = delete;

  inline ConcatenationTester& Axis(int axis) {
    axis_ = axis;
    return *this;
  }

  inline const int Axis() const { return axis_; }

  inline ConcatenationTester& InputShapes(
      const std::initializer_list<std::vector<int32_t>> shapes) {
    for (auto shape : shapes) {
      for (auto it = shape.begin(); it != shape.end(); ++it) {
        EXPECT_GT(*it, 0);
      }
    }
    input_shapes_ = shapes;
    input_scales_.resize(shapes.size(), 1);
    output_scale_ = 1.f;
    input_zero_points_.resize(shapes.size(), 0);
    output_zero_point_ = 0;
    return *this;
  }

  inline std::vector<int32_t> InputShape(size_t i) const {
    return input_shapes_[i];
  }

  inline ConcatenationTester& InputScales(const std::vector<float> scales) {
    input_scales_ = scales;
    return *this;
  }

  inline float InputScale(size_t i) const { return input_scales_[i]; }

  inline ConcatenationTester& InputZeroPoint(
      const std::vector<int32_t> zero_points) {
    input_zero_points_ = zero_points;
    return *this;
  }

  inline float InputZeroPoint(size_t i) const { return input_zero_points_[i]; }

  inline ConcatenationTester& OutputScale(float scale) {
    output_scale_ = scale;
    return *this;
  }

  inline float OutputScale() const { return output_scale_; }

  inline ConcatenationTester& OutputZeroPoint(int32_t zero_point) {
    output_zero_point_ = zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const { return output_scale_; }

  inline size_t NumInputs() const { return input_shapes_.size(); }

  std::vector<int32_t> OutputShape() const {
    std::vector<int32_t> output_shape = InputShape(0);
    int concat_axis = Axis() < 0 ? Axis() + output_shape.size() : Axis();
    size_t axis_dim_size = 0;
    for (size_t i = 0; i < NumInputs(); i++) {
      axis_dim_size += InputShape(i)[concat_axis];
    }
    output_shape[concat_axis] = axis_dim_size;
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
  std::vector<int32_t> output_shape_;
  std::vector<std::vector<int32_t>> input_shapes_;
  std::vector<float> input_scales_;
  float output_scale_;
  std::vector<int32_t> input_zero_points_;
  int32_t output_zero_point_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_CONCATENATION_TESTER_H_
