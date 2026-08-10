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

#include <cstdint>
#include <functional>
#include <string>

#include "absl/strings/string_view.h"
#include "third_party/sandboxed_api/annotations.h"

// From header file: third_party/tensorflow/core/lib/webp/webp_io.h

namespace tensorflow {
namespace webp {

bool DecodeWebPHeader(absl::string_view webp_string, int* width SANDBOX_OUT_PTR,
                      int* height SANDBOX_OUT_PTR,
                      int* channels SANDBOX_OUT_PTR,
                      bool* has_animation SANDBOX_OUT_PTR);

bool DecodeWebPImage(absl::string_view webp_string,
                     uint8_t* output SANDBOX_OUT_PTR
                         SANDBOX_ELEM_SIZED_BY(width * height * channels),
                     int width, int height, int channels, bool use_threads);

SANDBOX_ALIAS_CALLBACK_RETURN(allocate_output)
uint8_t* DecodeWebPAnimation(
    absl::string_view webp_string,
    const std::function<uint8_t*(int num_frames, int width, int height,
                                 int channels)>& allocate_output SANDBOX_OUT_PTR
        SANDBOX_BYTE_SIZED_BY(num_frames * width * height * channels),
    std::string* error_string SANDBOX_OUT_PTR, bool expand_animations,
    bool use_threads);

}  // namespace webp
}  // namespace tensorflow
