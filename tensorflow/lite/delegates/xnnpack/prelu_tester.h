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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_PRELU_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_PRELU_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

class PreluTester {
 public:
  PreluTester() = default;
  PreluTester(const PreluTester&) = delete;
  PreluTester& operator=(const PreluTester&) = delete;

  inline PreluTester& InputShape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline PreluTester& SlopeShape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    slope_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& SlopeShape() const { return slope_shape_; }

  inline const std::vector<int32_t>& OutputShape() const {
    return InputShape();
  }

  inline PreluTester& FP16Weights() {
    fp16_weights_ = true;
    return *this;
  }

  inline bool FP16Weights() const { return fp16_weights_; }

  inline PreluTester& INT8Weights() {
    int8_weights_ = true;
    return *this;
  }

  inline bool INT8Weights() const { return int8_weights_; }

  inline PreluTester& INT8ChannelWiseWeights() {
    int8_channel_wise_weights_ = true;
    return *this;
  }

  inline bool INT8ChannelWiseWeights() const {
    return int8_channel_wise_weights_;
  }

  inline PreluTester& SparseWeights() {
    sparse_weights_ = true;
    return *this;
  }

  inline bool SparseWeights() const { return sparse_weights_; }

  inline PreluTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> slope_shape_;
  bool fp16_weights_ = false;
  bool int8_weights_ = false;
  bool int8_channel_wise_weights_ = false;
  bool sparse_weights_ = false;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_PRELU_TESTER_H_
