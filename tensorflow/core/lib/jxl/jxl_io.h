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

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace jxl {

// Returns true if the given string starts with a JXL header.
bool HasJxlHeader(absl::string_view encoded);

// Decode JXL header and get image dimensions and number of channels.
bool DecodeHeader(absl::string_view encoded, int* width, int* height,
                  int* channels);

// Decode JXL image into pixels pointer.
bool DecodeImage(absl::string_view encoded, int channels, uint8_t* output,
                 size_t output_size);

}  // namespace jxl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_JXL_JXL_IO_H_
