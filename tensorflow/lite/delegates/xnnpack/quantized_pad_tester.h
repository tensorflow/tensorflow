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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_PAD_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_PAD_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class QuantizedPadTester {
 public:
  QuantizedPadTester() = default;
  QuantizedPadTester(const QuantizedPadTester&) = delete;
  QuantizedPadTester& operator=(const QuantizedPadTester&) = delete;

  inline QuantizedPadTester& InputShape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline QuantizedPadTester& InputPrePaddings(
      std::initializer_list<int32_t> paddings) {
    for (auto it = paddings.begin(); it != paddings.end(); ++it) {
      EXPECT_GE(*it, 0);
    }
    input_pre_paddings_ =
        std::vector<int32_t>(paddings.begin(), paddings.end());
    return *this;
  }

  inline const std::vector<int32_t> InputPrePaddings() const {
    return input_pre_paddings_;
  }

  inline QuantizedPadTester& InputPostPaddings(
      std::initializer_list<int32_t> paddings) {
    for (auto it = paddings.begin(); it != paddings.end(); ++it) {
      EXPECT_GE(*it, 0);
    }
    input_post_paddings_ =
        std::vector<int32_t>(paddings.begin(), paddings.end());
    return *this;
  }

  inline const std::vector<int32_t> InputPostPaddings() const {
    return input_post_paddings_;
  }

  std::vector<int32_t> OutputShape() const;

  inline QuantizedPadTester& ZeroPoint(int32_t zero_point) {
    zero_point_ = zero_point;
    return *this;
  }

  inline int32_t ZeroPoint() const { return zero_point_; }

  inline QuantizedPadTester& Scale(float scale) {
    scale_ = scale;
    return *this;
  }

  inline float Scale() const { return scale_; }

  inline QuantizedPadTester& Unsigned(bool is_unsigned) {
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

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> input_pre_paddings_;
  std::vector<int32_t> input_post_paddings_;
  int32_t zero_point_ = 7;
  float scale_ = 0.8f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_PAD_TESTER_H_
