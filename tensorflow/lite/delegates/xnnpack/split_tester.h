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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_SPLIT_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_SPLIT_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class SplitTester {
 public:
  SplitTester() = default;
  SplitTester(const SplitTester&) = delete;
  SplitTester& operator=(const SplitTester&) = delete;

  inline SplitTester& SplitDimension(int32_t split_dim) {
    split_dim_ = split_dim;
    return *this;
  }

  inline SplitTester& InputShape(const std::vector<int32_t>& shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  int32_t SplitDimension() const { return split_dim_; }

  inline SplitTester& NumSplits(int num_splits) {
    num_splits_ = num_splits;
    return *this;
  }

  inline const int NumSplits() const { return num_splits_; }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  std::vector<int32_t> OutputShape() const {
    std::vector<int32_t> output_shape = InputShape();
    int32_t split_dim = SplitDimension();
    split_dim += split_dim < 0 ? InputShape().size() : 0;
    EXPECT_LE(0, split_dim);
    EXPECT_EQ(0, output_shape[split_dim] % NumSplits());
    output_shape[split_dim] /= NumSplits();
    return output_shape;
  }

  template <typename T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;
  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  int32_t split_dim_;
  int num_splits_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_SPLIT_TESTER_H_
