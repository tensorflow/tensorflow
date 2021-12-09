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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZE_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class QuantizeTester {
 public:
  QuantizeTester() = default;
  QuantizeTester(const QuantizeTester&) = delete;
  QuantizeTester& operator=(const QuantizeTester&) = delete;

  inline QuantizeTester& Shape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    size_ = QuantizeTester::ComputeSize(shape_);
    return *this;
  }

  const std::vector<int32_t>& Shape() const { return shape_; }

  int32_t Size() const { return size_; }

  inline QuantizeTester& OutputZeroPoint(int32_t output_zero_point) {
    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const { return output_zero_point_; }

  inline QuantizeTester& OutputScale(float output_scale) {
    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const { return output_scale_; }

  inline QuantizeTester& Unsigned(bool is_unsigned) {
    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const { return unsigned_; }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> shape_;
  int32_t size_;
  int32_t output_zero_point_ = 0;
  float output_scale_ = 1.0f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZE_TESTER_H_
