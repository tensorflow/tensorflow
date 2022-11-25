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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_DECODE_JPEG_REGISTER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_DECODE_JPEG_REGISTER_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// DECODE_JPEG can be used to decode a batch of JPEG images on Android.
// TODO(b/172544567): Support iOS.
// TODO(b/172544567): Support greyscale images.
// Expects single 1D input of the shape {num_images} and type string.
// Single output containing the decoded images with shape {num_images, height,
// width, channels}. All input images are required to have 3 channels. The
// decoded images can have 3 or 4 channels depending on the shape of the
// target model input. This op will add an alpha channel value of 255 (fully
// opaque) if the target model accepts input images with 4 channels. This op
// will eventually be included in mainline Tflite as a built-in/custom op once
// it supports both Android and iOS.
TfLiteRegistration* Register_DECODE_JPEG();

struct OpData {
  // Number of images to decode.
  int32_t num_images;
  // All images should have the same height and width.
  // Height of images after decoding.
  int32_t height;
  // Width of images after decoding.
  int32_t width;
  // Number of channels to be decoded. Accepted values are 3 (RGB) and 4 (RGBA).
  int32_t channels;
};

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_DECODE_JPEG_REGISTER_H_
