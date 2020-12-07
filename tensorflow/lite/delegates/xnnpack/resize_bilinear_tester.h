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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_RESIZE_BILINEAR_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_RESIZE_BILINEAR_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class ResizeBilinearTester {
 public:
  ResizeBilinearTester() = default;
  ResizeBilinearTester(const ResizeBilinearTester&) = delete;
  ResizeBilinearTester& operator=(const ResizeBilinearTester&) = delete;

  inline ResizeBilinearTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline ResizeBilinearTester& Channels(int32_t channels) {
    EXPECT_GT(channels, 0);
    channels_ = channels;
    return *this;
  }

  inline int32_t Channels() const { return channels_; }

  inline ResizeBilinearTester& InputHeight(int32_t input_height) {
    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const { return input_height_; }

  inline ResizeBilinearTester& InputWidth(int32_t input_width) {
    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const { return input_width_; }

  inline ResizeBilinearTester& OutputHeight(int32_t output_height) {
    EXPECT_GT(output_height, 0);
    output_height_ = output_height;
    return *this;
  }

  inline int32_t OutputHeight() const { return output_height_; }

  inline ResizeBilinearTester& OutputWidth(int32_t output_width) {
    EXPECT_GT(output_width, 0);
    output_width_ = output_width;
    return *this;
  }

  inline int32_t OutputWidth() const { return output_width_; }

  ResizeBilinearTester& AlignCorners(bool align_corners) {
    align_corners_ = align_corners;
    return *this;
  }

  bool AlignCorners() const { return align_corners_; }

  ResizeBilinearTester& HalfPixelCenters(bool half_pixel_centers) {
    half_pixel_centers_ = half_pixel_centers;
    return *this;
  }

  bool HalfPixelCenters() const { return half_pixel_centers_; }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  int32_t batch_size_ = 1;
  int32_t channels_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t output_height_ = 1;
  int32_t output_width_ = 1;
  bool align_corners_ = false;
  bool half_pixel_centers_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_RESIZE_BILINEAR_TESTER_H_
