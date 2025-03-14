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

#include "tensorflow/core/lib/webp/webp_io.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/libwebp/src/webp/decode.h"
#include "third_party/libwebp/src/webp/demux.h"
#include "third_party/libwebp/src/webp/mux_types.h"

namespace tensorflow {
namespace webp {

bool DecodeWebPHeader(absl::string_view webp_string, int* width, int* height,
                      int* channels, bool* has_animation) {
  const uint8_t* input_data =
      reinterpret_cast<const uint8_t*>(webp_string.data());
  const size_t input_size = webp_string.size();

  WebPBitstreamFeatures features;
  if (WebPGetFeatures(input_data, input_size, &features) != VP8_STATUS_OK) {
    return false;
  }

  *width = features.width;
  *height = features.height;
  *channels = features.has_alpha ? 4 : 3;
  *has_animation = features.has_animation;
  return true;
}

bool DecodeWebPImage(absl::string_view webp_string, uint8_t* output, int width,
                     int height, int channels) {
  const uint8_t* input_data =
      reinterpret_cast<const uint8_t*>(webp_string.data());
  const size_t input_size = webp_string.size();
  const int row_stride = width * channels * sizeof(uint8_t);
  const size_t output_size = height * row_stride;

  switch (channels) {
    case 3:
      return ::WebPDecodeRGBInto(input_data, input_size, output, output_size,
                                 row_stride) != nullptr;
    case 4:
      return ::WebPDecodeRGBAInto(input_data, input_size, output, output_size,
                                  row_stride) != nullptr;
    default:
      // Invalid number of channels.
      return false;
  }
}

uint8_t* DecodeWebPAnimation(
    absl::string_view webp_string,
    const std::function<uint8_t*(int, int, int, int)>& allocate_output,
    std::string* error_string, bool expand_animations) {
  WebPData webp_data = {reinterpret_cast<const uint8_t*>(webp_string.data()),
                        webp_string.size()};

  // Use the default decoder options, which is single-threaded RGBA decode.
  WebPAnimDecoder* decoder = WebPAnimDecoderNew(&webp_data, nullptr);
  if (decoder == nullptr) {
    *error_string = "failed to decode WebP Animation";
    return nullptr;
  }

  const auto cleanup =
      absl::MakeCleanup([decoder] { WebPAnimDecoderDelete(decoder); });

  WebPAnimInfo info;
  if (!WebPAnimDecoderGetInfo(decoder, &info)) {
    *error_string = "failed to get WebP Animation Info";
    return nullptr;
  }

  const uint32_t width = info.canvas_width;
  const uint32_t height = info.canvas_height;
  // If we only want the first frame, expand_animations will be false.
  const uint32_t num_frames = (expand_animations) ? info.frame_count : 1;
  const uint32_t num_channels = 4; /* libwebp only supports RGBA animations */
  const size_t bytes_per_frame = width * height * num_channels;

  uint8_t* output = allocate_output(num_frames, width, height, num_channels);
  if (output == nullptr) {
    *error_string = "failed to allocate output for WebP Animation";
    return nullptr;
  }

  size_t frame = 0;
  while (WebPAnimDecoderHasMoreFrames(decoder)) {
    uint8_t* buffer;
    int timestamp_dummy;
    if (!WebPAnimDecoderGetNext(decoder, &buffer, &timestamp_dummy)) {
      *error_string = absl::StrCat("failed to decode frame: ", frame);
      return nullptr;
    }

    // Copy buffer (owned by decoder) into our output.
    uint8_t* frame_output = output + frame * bytes_per_frame;
    memcpy(frame_output, buffer, bytes_per_frame);

    // Move on to the next frame.
    frame++;

    // Exit early, if we only want to grab the first frame.
    if (!expand_animations) break;
  }

  // We should have gotten all the frames in num_frames.
  if (frame != num_frames) {
    *error_string =
        absl::StrCat("only read ", frame, " of ", num_frames, " frames");
    return nullptr;
  }

  return output;
}

}  // namespace webp
}  // namespace tensorflow
