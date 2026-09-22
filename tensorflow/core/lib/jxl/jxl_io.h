/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// Functions to read images in JPEG XL format.
//
// Example usage:
//
//   int width, height, channels;
//   if (!DecodeHeader(encoded, &width, &height, &channels)) {
//     // Handle error.
//   }
//
//   uint8_t* image = new uint8_t[width * height * channels];
//   if (!DecodeImage(encoded, channels, image, width * height * channels)) {
//     // Handle error.
//   }
//
//   // Do something with the decoded image.
//   delete[] image;

#ifndef TENSORFLOW_CORE_LIB_JXL_JXL_IO_H_
#define TENSORFLOW_CORE_LIB_JXL_JXL_IO_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace jxl {

// Returns true if the given string starts with a JXL header.
bool HasJxlHeader(absl::string_view encoded);

// Decode JXL header and get image dimensions and number of channels.
// Optionally returns the bit depth per sample (e.g. 8, 16) if bit_depth !=
// nullptr.
bool DecodeHeader(absl::string_view encoded, int* width, int* height,
                  int* channels, int* bit_depth = nullptr);

// Decode JXL image into uint8 pixels pointer.
bool DecodeImage(absl::string_view encoded, int channels, uint8_t* output,
                 size_t output_size_bytes);

// Decode JXL image with specified channel_bits (8, 16, or 32) into output
// pointer.
bool DecodeImage(absl::string_view encoded, int channels, int channel_bits,
                 void* output, size_t output_size_bytes);

// Decode JXL image into uint16 pixels pointer.
bool DecodeImage16(absl::string_view encoded, int channels, uint16_t* output,
                   size_t output_size_bytes);

// Decode JXL image into float16 (half) pixels pointer (nominal range
// [0.0, 1.0]).
bool DecodeImageFloat16(absl::string_view encoded, int channels, void* output,
                        size_t output_size_bytes);

// Decode JXL image into float pixels pointer (nominal range [0.0, 1.0]).
bool DecodeImageFloat(absl::string_view encoded, int channels, float* output,
                      size_t output_size_bytes);

// Maps a JPEG-style quality factor to a Butteraugli distance, using libjxl's
// own mapping (the same one used by `cjxl -q`). quality 100 maps to distance
// 0.0 (lossless), quality 90 maps to distance 1.0 (visually lossless), and
// quality 0 maps to distance 25.0.
float DistanceFromQuality(float quality);

// Encode an image to JPEG XL format.
// Supports channel_bits 8, 16 (uint16 or float16), or 32 (float), channels 1
// (gray), 3 (rgb), 4 (rgba). Set is_float = true for float16 (channel_bits 16)
// or float32 (channel_bits 32).
//
// distance is the Butteraugli distance in [0.0, 25.0]. 0.0 selects lossless
// modular encoding; any positive value selects lossy VarDCT encoding, where
// 1.0 is visually lossless and larger values compress more. Values in
// (0.0, 0.05) are clamped up to 0.05 by libjxl. Use DistanceFromQuality() to
// convert from a JPEG-style quality factor.
//
// effort can be 1 (fastest) to 9 (slowest/best compression), default is 7.
//
template <typename T>
bool WriteImageToBuffer(const void* image_data, int width, int height,
                        int channels, int channel_bits, float distance,
                        int effort, T* output, bool is_float = false);

extern template bool WriteImageToBuffer<std::string>(
    const void* image_data, int width, int height, int channels,
    int channel_bits, float distance, int effort, std::string* output,
    bool is_float);

extern template bool WriteImageToBuffer<tstring>(const void* image_data,
                                                 int width, int height,
                                                 int channels, int channel_bits,
                                                 float distance, int effort,
                                                 tstring* output,
                                                 bool is_float);

}  // namespace jxl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_JXL_JXL_IO_H_
