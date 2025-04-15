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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_FULLY_CONNECTED_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_FULLY_CONNECTED_TESTER_H_

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class FullyConnectedTester {
 public:
  enum class WeightsType {
    kFP32,
    kFP16,
    kTensorWiseQuantizedInt8,
    kChannelWiseQuantizedInt8,
    kDynamic,
  };
  enum class BiasType {
    kNone,
    kFP32,
    kFP16,
    kDynamic,
  };

  FullyConnectedTester() = default;
  FullyConnectedTester(const FullyConnectedTester&) = delete;
  FullyConnectedTester& operator=(const FullyConnectedTester&) = delete;

  inline FullyConnectedTester& InputShape(
      std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input_size_ = ComputeSize(input_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline int32_t InputSize() const { return input_size_; }

  inline FullyConnectedTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline FullyConnectedTester& OutputChannels(int32_t output_channels) {
    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const { return output_channels_; }

  std::vector<int32_t> OutputShape() const;

  inline FullyConnectedTester& KeepDims(bool keep_dims) {
    keep_dims_ = keep_dims;
    return *this;
  }

  inline bool KeepDims() const { return keep_dims_; }

  inline FullyConnectedTester& FP16Weights() {
    weights_type_ = WeightsType::kFP16;
    bias_type_ = BiasType::kFP16;
    return *this;
  }

  inline FullyConnectedTester& TensorWiseQuantizedInt8Weights() {
    weights_type_ = WeightsType::kTensorWiseQuantizedInt8;
    // Bias is stored in FP32 even when filter is quantized to INT8
    bias_type_ = BiasType::kFP32;
    return *this;
  }

  inline FullyConnectedTester& ChannelWiseQuantizedInt8Weights() {
    weights_type_ = WeightsType::kChannelWiseQuantizedInt8;
    // Bias is stored in FP32 even when filter is quantized to INT8
    bias_type_ = BiasType::kFP32;
    return *this;
  }

  inline FullyConnectedTester& DynamicWeights() {
    weights_type_ = WeightsType::kDynamic;
    bias_type_ = BiasType::kFP32;
    return *this;
  }

  inline FullyConnectedTester& NoBias() {
    bias_type_ = BiasType::kNone;
    return *this;
  }

  inline FullyConnectedTester& DynamicBias() {
    bias_type_ = BiasType::kDynamic;
    return *this;
  }

  inline FullyConnectedTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline FullyConnectedTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline FullyConnectedTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline FullyConnectedTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline bool HasBias() const { return bias_type_ != BiasType::kNone; }

  inline WeightsType WeightsType() const { return weights_type_; }

  inline BiasType BiasType() const { return bias_type_; }

  inline ::tflite::ActivationFunctionType Activation() const {
    return activation_;
  }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  int32_t input_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;
  bool keep_dims_ = false;
  enum WeightsType weights_type_ { WeightsType::kFP32 };
  enum BiasType bias_type_ { BiasType::kFP32 };
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_FULLY_CONNECTED_TESTER_H_
