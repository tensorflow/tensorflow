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

// Functions to read images in WebP format.
//
// First call DecodeWebPHeader with an input WebP image as a string_view, to get
// the width, height, channels, and whether or not the WebP file is an animation
// (animation decoding is currently not supported beyond the first frame). Then
// call DecodeWebP with an appropriately sized output buffer to hold the decoded
// images as either RGB or RGBA (based on channels)
//
//
// int width, height, channels;
// bool has_animation;
// DecodeWebPHeader(input_bytes, &width, &height, &channels, &has_animation);
//
// if (has_animation) { DecideIfYouWantFrame0(); }
//
// uint8_t* output_bytes = new uint8_t[width * height * channels];
// DecodeWebPImage(input_bytes, output_bytes, width, height, channels);
//

#ifndef TENSORFLOW_CORE_LIB_WEBP_WEBP_IO_H_
#define TENSORFLOW_CORE_LIB_WEBP_WEBP_IO_H_

#include <functional>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace webp {

// Given an input encoded in WebP as `webp_string`, extract the width, height,
// number of channels, and whether or not the file is an animation. Return false
// on failure or true for success.
bool DecodeWebPHeader(absl::string_view webp_string, int* width, int* height,
                      int* channels, bool* has_animation);

// Decode the first image from `webp_string` into the output buffer
// `output`. `output` is assumed to be width * height * channels *
// sizeof(uint8_t) or larger.
bool DecodeWebPImage(absl::string_view webp_string, uint8_t* output, int width,
                     int height, int channels);

// Decode a sequence of images in the animation from `webp_string` into a
// dynamically allocated output buffer via `allocate_output`. `allocate_output`
// takes the arguments as (num_frames, width, height, channels). The channels is
// (currently) always 4 (RGBA).
//
// Note: Decoding a WebP animation, even to get the number of frames, reads the
// entire image into memory, hence this callback mechanism.
uint8_t* DecodeWebPAnimation(
    absl::string_view webp_string,
    const std::function<uint8_t*(int, int, int, int)>& allocate_output,
    std::string* error_string, bool expand_animations);

}  // namespace webp
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_WEBP_WEBP_IO_H_
