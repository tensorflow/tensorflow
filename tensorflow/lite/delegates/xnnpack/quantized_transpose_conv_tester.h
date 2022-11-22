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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_TRANSPOSE_CONV_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_TRANSPOSE_CONV_TESTER_H_

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

class Interpreter;

namespace xnnpack {

class QuantizedTransposeConvTester {
 public:
  explicit QuantizedTransposeConvTester() = default;
  QuantizedTransposeConvTester(const QuantizedTransposeConvTester&) = delete;
  QuantizedTransposeConvTester& operator=(const QuantizedTransposeConvTester&) =
      delete;

  inline QuantizedTransposeConvTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline QuantizedTransposeConvTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline QuantizedTransposeConvTester& OutputChannels(int32_t output_channels) {
    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const { return output_channels_; }

  inline QuantizedTransposeConvTester& OutputHeight(int32_t output_height) {
    EXPECT_GT(output_height, 0);
    output_height_ = output_height;
    return *this;
  }

  inline int32_t OutputHeight() const { return output_height_; }

  inline QuantizedTransposeConvTester& OutputWidth(int32_t output_width) {
    EXPECT_GT(output_width, 0);
    output_width_ = output_width;
    return *this;
  }

  inline int32_t OutputWidth() const { return output_width_; }

  inline QuantizedTransposeConvTester& KernelHeight(int32_t kernel_height) {
    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  inline int32_t KernelHeight() const { return kernel_height_; }

  inline QuantizedTransposeConvTester& KernelWidth(int32_t kernel_width) {
    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  inline int32_t KernelWidth() const { return kernel_width_; }

  inline QuantizedTransposeConvTester& StrideHeight(int32_t stride_height) {
    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const { return stride_height_; }

  inline QuantizedTransposeConvTester& StrideWidth(int32_t stride_width) {
    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const { return stride_width_; }

  inline QuantizedTransposeConvTester& SparseWeights() {
    sparse_weights_ = true;
    return *this;
  }

  inline bool SparseWeights() const { return sparse_weights_; }

  inline QuantizedTransposeConvTester& SamePadding() {
    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline QuantizedTransposeConvTester& ValidPadding() {
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

  inline bool UseBias() const { return use_bias_; }

  inline QuantizedTransposeConvTester& WithBias(bool use_bias = true) {
    use_bias_ = use_bias;
    return *this;
  }

  inline QuantizedTransposeConvTester& NoBias() { return WithBias(false); }

  inline QuantizedTransposeConvTester& Unsigned(bool is_unsigned) {
    unsigned_ = is_unsigned;
    return *this;
  }

  inline QuantizedTransposeConvTester& Signed(bool is_signed = true) {
    return Unsigned(!is_signed);
  }

  inline bool Unsigned() const { return unsigned_; }

  inline QuantizedTransposeConvTester& WeightsCache(
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

  template <typename WeightType>
  void EnsureOutputsClose(const Interpreter* default_interpreter,
                          const Interpreter* delegate_interpreter) const;

 private:
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
  bool unsigned_ = true;
  bool use_bias_ = true;
  bool sparse_weights_ = false;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_TRANSPOSE_CONV_TESTER_H_
