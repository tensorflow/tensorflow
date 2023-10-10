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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_CONV_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_CONV_TESTER_H_

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class TransposeConvTester {
 public:
  enum class WeightsType {
    kFP32,
    kFP16,
    kTensorWiseQuantizedInt8,
    kChannelWiseQuantizedInt8,
  };
  enum class BiasType {
    kNone,
    kFP32,
    kFP16,
  };

  TransposeConvTester() = default;
  TransposeConvTester(const TransposeConvTester&) = delete;
  TransposeConvTester& operator=(const TransposeConvTester&) = delete;

  inline TransposeConvTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline TransposeConvTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline TransposeConvTester& OutputChannels(int32_t output_channels) {
    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const { return output_channels_; }

  inline TransposeConvTester& OutputHeight(int32_t output_height) {
    EXPECT_GT(output_height, 0);
    output_height_ = output_height;
    return *this;
  }

  inline int32_t OutputHeight() const { return output_height_; }

  inline TransposeConvTester& OutputWidth(int32_t output_width) {
    EXPECT_GT(output_width, 0);
    output_width_ = output_width;
    return *this;
  }

  inline int32_t OutputWidth() const { return output_width_; }

  inline TransposeConvTester& KernelHeight(int32_t kernel_height) {
    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  inline int32_t KernelHeight() const { return kernel_height_; }

  inline TransposeConvTester& KernelWidth(int32_t kernel_width) {
    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  inline int32_t KernelWidth() const { return kernel_width_; }

  inline TransposeConvTester& StrideHeight(int32_t stride_height) {
    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const { return stride_height_; }

  inline TransposeConvTester& StrideWidth(int32_t stride_width) {
    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const { return stride_width_; }

  inline TransposeConvTester& FP16Weights() {
    weights_type_ = WeightsType::kFP16;
    bias_type_ = BiasType::kFP16;
    return *this;
  }

  inline TransposeConvTester& TensorWiseQuantizedInt8Weights() {
    weights_type_ = WeightsType::kTensorWiseQuantizedInt8;
    // Bias is stored in FP32 even when filter is quantized to INT8
    bias_type_ = BiasType::kFP32;
    return *this;
  }

  inline TransposeConvTester& ChannelWiseQuantizedInt8Weights() {
    weights_type_ = WeightsType::kChannelWiseQuantizedInt8;
    // Bias is stored in FP32 even when filter is quantized to INT8
    bias_type_ = BiasType::kFP32;
    return *this;
  }

  inline TransposeConvTester& SparseWeights() {
    sparse_weights_ = true;
    return *this;
  }

  inline bool SparseWeights() const { return sparse_weights_; }

  inline TransposeConvTester& SamePadding() {
    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline TransposeConvTester& ValidPadding() {
    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline ::tflite::Padding Padding() const { return padding_; }

  inline int32_t InputWidth() const {
    return ComputeInputSize(OutputWidth(), KernelWidth(), StrideWidth());
  }

  inline int32_t InputHeight() const {
    return ComputeInputSize(OutputHeight(), KernelHeight(), StrideHeight());
  }

  inline int32_t PaddingWidth() const {
    return ComputePadding(OutputWidth(), KernelWidth(), StrideWidth());
  }

  inline int32_t PaddingHeight() const {
    return ComputePadding(OutputHeight(), KernelHeight(), StrideHeight());
  }

  inline TransposeConvTester& NoBias() {
    bias_type_ = BiasType::kNone;
    return *this;
  }

  inline TransposeConvTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

 private:
  int32_t ComputeInputSize(int32_t output_size, int32_t kernel_size,
                           int32_t stride) const {
    // Roughly follows TFLite's `ComputeOutSize`.
    switch (padding_) {
      case ::tflite::Padding_VALID:
        return (output_size + stride - kernel_size) / stride;
        break;
      case ::tflite::Padding_SAME:
        return (output_size + stride - 1) / stride;
        break;
      default:
        assert(false);
    }
  }

  int32_t ComputePadding(int32_t output_size, int32_t kernel_size,
                         int32_t stride) const {
    // Roughly follows TFLite's `ComputePaddingWithOffset`.
    if (padding_ == ::tflite::Padding_VALID) {
      return 0;
    }
    assert(padding_ == ::tflite::Padding_SAME);
    const int32_t input_size =
        ComputeInputSize(output_size, kernel_size, stride);
    return (output_size - 1) * stride + kernel_size - input_size;
  }

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline bool HasBias() const { return bias_type_ != BiasType::kNone; }

  inline WeightsType WeightsType() const { return weights_type_; }

  inline BiasType BiasType() const { return bias_type_; }

  int32_t batch_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;
  int32_t output_height_ = 1;
  int32_t output_width_ = 1;
  int32_t kernel_height_ = 1;
  int32_t kernel_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  enum WeightsType weights_type_ { WeightsType::kFP32 };
  enum BiasType bias_type_ { BiasType::kFP32 };
  bool sparse_weights_ = false;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_CONV_TESTER_H_
