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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_DEPTHWISE_CONV_2D_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_DEPTHWISE_CONV_2D_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class DepthwiseConv2DTester {
 public:
  DepthwiseConv2DTester() = default;
  DepthwiseConv2DTester(const DepthwiseConv2DTester&) = delete;
  DepthwiseConv2DTester& operator=(const DepthwiseConv2DTester&) = delete;

  inline DepthwiseConv2DTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline DepthwiseConv2DTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline DepthwiseConv2DTester& DepthMultiplier(int32_t depth_multiplier) {
    EXPECT_GT(depth_multiplier, 0);
    depth_multiplier_ = depth_multiplier;
    return *this;
  }

  inline int32_t DepthMultiplier() const { return depth_multiplier_; }

  inline int32_t OutputChannels() const {
    return DepthMultiplier() * InputChannels();
  }

  inline DepthwiseConv2DTester& InputHeight(int32_t input_height) {
    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const { return input_height_; }

  inline DepthwiseConv2DTester& InputWidth(int32_t input_width) {
    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const { return input_width_; }

  inline int32_t OutputWidth() const {
    if (Padding() == ::tflite::Padding_SAME) {
      EXPECT_GE(InputWidth(), 1);
      return (InputWidth() - 1) / StrideWidth() + 1;
    } else {
      EXPECT_GE(InputWidth(), DilatedKernelWidth());
      return 1 + (InputWidth() - DilatedKernelWidth()) / StrideWidth();
    }
  }

  inline int32_t OutputHeight() const {
    if (Padding() == ::tflite::Padding_SAME) {
      EXPECT_GE(InputHeight(), 1);
      return (InputHeight() - 1) / StrideHeight() + 1;
    } else {
      EXPECT_GE(InputHeight(), DilatedKernelHeight());
      return 1 + (InputHeight() - DilatedKernelHeight()) / StrideHeight();
    }
  }

  inline DepthwiseConv2DTester& KernelHeight(int32_t kernel_height) {
    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  inline int32_t KernelHeight() const { return kernel_height_; }

  inline DepthwiseConv2DTester& KernelWidth(int32_t kernel_width) {
    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  inline int32_t KernelWidth() const { return kernel_width_; }

  inline DepthwiseConv2DTester& StrideHeight(int32_t stride_height) {
    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const { return stride_height_; }

  inline DepthwiseConv2DTester& StrideWidth(int32_t stride_width) {
    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const { return stride_width_; }

  inline DepthwiseConv2DTester& DilationHeight(int32_t dilation_height) {
    EXPECT_GT(dilation_height, 0);
    dilation_height_ = dilation_height;
    return *this;
  }

  inline int32_t DilationHeight() const { return dilation_height_; }

  inline DepthwiseConv2DTester& DilationWidth(int32_t dilation_width) {
    EXPECT_GT(dilation_width, 0);
    dilation_width_ = dilation_width;
    return *this;
  }

  inline int32_t DilationWidth() const { return dilation_width_; }

  inline int32_t DilatedKernelHeight() const {
    return (KernelHeight() - 1) * DilationHeight() + 1;
  }

  inline int32_t DilatedKernelWidth() const {
    return (KernelWidth() - 1) * DilationWidth() + 1;
  }

  inline DepthwiseConv2DTester& FP16Weights() {
    fp16_weights_ = true;
    return *this;
  }

  inline bool FP16Weights() const { return fp16_weights_; }

  inline DepthwiseConv2DTester& INT8Weights() {
    int8_weights_ = true;
    return *this;
  }

  inline bool INT8Weights() const { return int8_weights_; }

  inline DepthwiseConv2DTester& SparseWeights() {
    sparse_weights_ = true;
    return *this;
  }

  inline bool SparseWeights() const { return sparse_weights_; }

  inline DepthwiseConv2DTester& SamePadding() {
    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline DepthwiseConv2DTester& ValidPadding() {
    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline DepthwiseConv2DTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline DepthwiseConv2DTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline DepthwiseConv2DTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline DepthwiseConv2DTester& TanhActivation() {
    activation_ = ::tflite::ActivationFunctionType_TANH;
    return *this;
  }

  inline DepthwiseConv2DTester& SignBitActivation() {
    activation_ = ::tflite::ActivationFunctionType_SIGN_BIT;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline ::tflite::Padding Padding() const { return padding_; }

  inline ::tflite::ActivationFunctionType Activation() const {
    return activation_;
  }

  int32_t batch_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t depth_multiplier_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t kernel_height_ = 1;
  int32_t kernel_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  int32_t dilation_height_ = 1;
  int32_t dilation_width_ = 1;
  bool fp16_weights_ = false;
  bool int8_weights_ = false;
  bool sparse_weights_ = false;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_DEPTHWISE_CONV_2D_TESTER_H_
