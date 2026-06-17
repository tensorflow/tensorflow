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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_SPACE_TO_DEPTH_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_SPACE_TO_DEPTH_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite::xnnpack {

class SpaceToDepthTester {
 public:
  SpaceToDepthTester() = default;
  SpaceToDepthTester(const SpaceToDepthTester&) = delete;
  SpaceToDepthTester& operator=(const SpaceToDepthTester&) = delete;

  inline SpaceToDepthTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t NumInputElements() const {
    return batch_size_ * input_height_ * input_width_ * input_channels_;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline int32_t OutputChannels() const {
    return InputChannels() * BlockSize() * BlockSize();
  }

  inline SpaceToDepthTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline SpaceToDepthTester& InputHeight(int32_t input_height) {
    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const { return input_height_; }

  inline SpaceToDepthTester& InputWidth(int32_t input_width) {
    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const { return input_width_; }

  inline int32_t OutputWidth() const { return InputWidth() / BlockSize(); }

  inline int32_t OutputHeight() const { return InputHeight() / BlockSize(); }

  inline SpaceToDepthTester& BlockSize(int32_t block_size) {
    EXPECT_GT(block_size, 1);
    block_size_ = block_size;
    return *this;
  }

  inline int32_t BlockSize() const { return block_size_; }

  template <class T>
  void Test(TensorType tensor_type, Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  int32_t batch_size_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t input_channels_ = 1;
  int32_t block_size_ = 2;
};

}  // namespace tflite::xnnpack

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_SPACE_TO_DEPTH_TESTER_H_
