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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_PAD_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_PAD_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace xnnpack {

class PadTester {
 public:
  PadTester() = default;
  PadTester(const PadTester&) = delete;
  PadTester& operator=(const PadTester&) = delete;

  inline PadTester& InputShape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline PadTester& InputPrePaddings(std::initializer_list<int32_t> paddings) {
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

  inline PadTester& InputPostPaddings(std::initializer_list<int32_t> paddings) {
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

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> input_pre_paddings_;
  std::vector<int32_t> input_post_paddings_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_PAD_TESTER_H_
