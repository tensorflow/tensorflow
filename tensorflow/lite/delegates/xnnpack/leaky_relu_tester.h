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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_LEAKY_RELU_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_LEAKY_RELU_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class LeakyReluTester {
 public:
  LeakyReluTester() = default;
  LeakyReluTester(const LeakyReluTester&) = delete;
  LeakyReluTester& operator=(const LeakyReluTester&) = delete;

  inline LeakyReluTester& Shape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    size_ = LeakyReluTester::ComputeSize(shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Shape() const { return shape_; }

  inline int32_t Size() const { return size_; }

  inline LeakyReluTester& NegativeSlope(float negative_slope) {
    negative_slope_ = negative_slope;
    return *this;
  }

  inline float NegativeSlope() const { return negative_slope_; }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> shape_;
  int32_t size_;
  float negative_slope_ = 0.5f;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_LEAKY_RELU_TESTER_H_
