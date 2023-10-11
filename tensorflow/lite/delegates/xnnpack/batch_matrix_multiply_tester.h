/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_BATCH_MATRIX_MULTIPLY_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_BATCH_MATRIX_MULTIPLY_TESTER_H_

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class BatchMatrixMultiplyTester {
 public:
  enum class WeightsType {
    kFP32,
  };

  BatchMatrixMultiplyTester() = default;
  BatchMatrixMultiplyTester(const BatchMatrixMultiplyTester&) = delete;
  BatchMatrixMultiplyTester& operator=(const BatchMatrixMultiplyTester&) =
      delete;

  inline BatchMatrixMultiplyTester& Input1Shape(
      std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input1_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input1_size_ = ComputeSize(input1_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Input1Shape() const {
    return input1_shape_;
  }

  inline BatchMatrixMultiplyTester& Input2Shape(
      std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input2_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input2_size_ = ComputeSize(input2_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& Input2Shape() const {
    return input2_shape_;
  }

  inline int32_t Input1Size() const { return input1_size_; }

  inline int32_t Input2Size() const { return input2_size_; }

  std::vector<int32_t> OutputShape() const;

  inline BatchMatrixMultiplyTester& AdjX(bool adj_x) {
    adj_x_ = adj_x;
    return *this;
  }

  inline bool AdjX() const { return adj_x_; }

  inline BatchMatrixMultiplyTester& AdjY(bool adj_y) {
    adj_y_ = adj_y;
    return *this;
  }

  inline bool AdjY() const { return adj_y_; }

  inline BatchMatrixMultiplyTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline WeightsType WeightsType() const { return weights_type_; }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input1_shape_;
  std::vector<int32_t> input2_shape_;
  int32_t input1_size_ = 1;
  int32_t input2_size_ = 1;
  bool adj_x_ = false;
  bool adj_y_ = false;
  enum WeightsType weights_type_ { WeightsType::kFP32 };
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_BATCH_MATRIX_MULTIPLY_TESTER_H_
