/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_PNG_FAST_PNG_H_
#define TENSORFLOW_CORE_LIB_PNG_FAST_PNG_H_

#include <string>

#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace png {

// Attempts SIMD-accelerated lossless PNG encoding for metadata-free images.
// Returns false without modifying `png_string` when fast encoding is
// unavailable or unsupported for the given image parameters, allowing callers
// to fall through to libpng.
bool TryFastEncodePng(const void* image, int width, int height, int row_bytes,
                      int num_channels, int channel_bits, int compression,
                      std::string* png_string);
bool TryFastEncodePng(const void* image, int width, int height, int row_bytes,
                      int num_channels, int channel_bits, int compression,
                      tstring* png_string);

}  // namespace png
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_PNG_FAST_PNG_H_
