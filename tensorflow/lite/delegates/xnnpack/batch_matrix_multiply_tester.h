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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

class BatchMatrixMultiplyTester {
 public:
  enum class WeightsType {
    kFP32,
  };

  enum QuantizationType {
    kNone,
    kChannel,
    kTensor,
  };

  BatchMatrixMultiplyTester() = default;
  BatchMatrixMultiplyTester(const BatchMatrixMultiplyTester&) = delete;
  BatchMatrixMultiplyTester& operator=(const BatchMatrixMultiplyTester&) =
      delete;

  BatchMatrixMultiplyTester& InputADims(std::initializer_list<int32_t> shape) {
    EXPECT_THAT(shape, testing::Each(testing::Gt(0)));
    input_a_dims_ = std::vector<int32_t>(shape.begin(), shape.end());
    input1_size_ = ComputeSize(input_a_dims_);
    return *this;
  }

  const std::vector<int32_t>& InputADims() const { return input_a_dims_; }

  BatchMatrixMultiplyTester& InputBDims(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_b_dims_ = std::vector<int32_t>(shape.begin(), shape.end());
    input2_size_ = ComputeSize(input_b_dims_);
    return *this;
  }

  const std::vector<int32_t>& InputBDims() const { return input_b_dims_; }

  BatchMatrixMultiplyTester& InputBQuant(QuantizationType quantization) {
    quant_b_ = quantization;
    return *this;
  }
  QuantizationType InputBQuant() const { return quant_b_; }

  int32_t Input1Size() const { return input1_size_; }

  int32_t Input2Size() const { return input2_size_; }

  std::vector<int32_t> OutputShape() const;

  BatchMatrixMultiplyTester& TransposeB(bool adj_y) {
    transpose_b_ = adj_y;
    return *this;
  }

  bool TransposeB() const { return transpose_b_; }

  BatchMatrixMultiplyTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  WeightsType WeightsType() const { return weights_type_; }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_a_dims_;
  std::vector<int32_t> input_b_dims_;
  QuantizationType quant_b_ = kNone;
  int32_t input1_size_ = 1;
  int32_t input2_size_ = 1;
  bool transpose_b_ = false;
  enum WeightsType weights_type_ { WeightsType::kFP32 };
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_BATCH_MATRIX_MULTIPLY_TESTER_H_
