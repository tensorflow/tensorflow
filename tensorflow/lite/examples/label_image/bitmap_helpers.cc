/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"

#include <unistd.h>  // NOLINT(build/include_order)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "tensorflow/lite/examples/label_image/label_image.h"
#include "tensorflow/lite/examples/label_image/log.h"

namespace tflite {
namespace label_image {
namespace {

constexpr size_t kBmpHeaderMinSize = 30;

uint16_t ReadLe16(const std::vector<uint8_t>& bytes, size_t offset) {
  return static_cast<uint16_t>(bytes[offset]) |
         static_cast<uint16_t>(bytes[offset + 1]) << 8;
}

int32_t ReadLe32(const std::vector<uint8_t>& bytes, size_t offset) {
  const uint32_t value = static_cast<uint32_t>(bytes[offset]) |
                         static_cast<uint32_t>(bytes[offset + 1]) << 8 |
                         static_cast<uint32_t>(bytes[offset + 2]) << 16 |
                         static_cast<uint32_t>(bytes[offset + 3]) << 24;
  return static_cast<int32_t>(value);
}

bool ValidateBmpAndGetPixelOffset(const std::vector<uint8_t>& img_bytes,
                                  int* width, int* height, int* channels,
                                  int* row_size, size_t* pixel_offset) {
  *width = 0;
  *height = 0;
  *channels = 0;
  *row_size = 0;
  *pixel_offset = 0;

  if (img_bytes.size() < kBmpHeaderMinSize || img_bytes[0] != 'B' ||
      img_bytes[1] != 'M') {
    LOG(ERROR) << "Invalid BMP header";
    return false;
  }

  const int32_t parsed_pixel_offset = ReadLe32(img_bytes, 10);
  const int32_t parsed_width = ReadLe32(img_bytes, 18);
  const int32_t parsed_height = ReadLe32(img_bytes, 22);
  const uint16_t bpp = ReadLe16(img_bytes, 28);

  if (parsed_pixel_offset < 0 ||
      static_cast<size_t>(parsed_pixel_offset) > img_bytes.size()) {
    LOG(ERROR) << "BMP pixel data offset is outside the file";
    return false;
  }
  if (parsed_width <= 0 || parsed_height == 0 || parsed_height == std::numeric_limits<int>::min()) {
    LOG(ERROR) << "Invalid BMP dimensions";
    return false;
  }
  if (bpp != 8 && bpp != 24 && bpp != 32) {
    LOG(ERROR) << "Unsupported BMP bits per pixel: " << bpp;
    return false;
  }

  const int parsed_channels = bpp / 8;
  const int64_t abs_height = parsed_height < 0 ? -static_cast<int64_t>(parsed_height)
                                               : static_cast<int64_t>(parsed_height);
  const int64_t bits_per_row = static_cast<int64_t>(bpp) * parsed_width;
  if (bits_per_row > (std::numeric_limits<int>::max() - 31)) {
    LOG(ERROR) << "BMP row size overflow";
    return false;
  }
  const int parsed_row_size = static_cast<int>((bits_per_row + 31) / 32 * 4);

  const size_t pixel_bytes = img_bytes.size() - static_cast<size_t>(parsed_pixel_offset);
  if (abs_height > 0 &&
      static_cast<uint64_t>(parsed_row_size) >
          std::numeric_limits<uint64_t>::max() /
              static_cast<uint64_t>(abs_height)) {
    LOG(ERROR) << "BMP pixel data size overflow";
    return false;
  }
  const uint64_t required_pixel_bytes =
      static_cast<uint64_t>(parsed_row_size) * static_cast<uint64_t>(abs_height);
  if (required_pixel_bytes > pixel_bytes) {
    LOG(ERROR) << "BMP pixel data is shorter than the declared dimensions";
    return false;
  }

  *width = parsed_width;
  *height = parsed_height;
  *channels = parsed_channels;
  *row_size = parsed_row_size;
  *pixel_offset = static_cast<size_t>(parsed_pixel_offset);
  return true;
}

}  // namespace

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }
  return output;
}

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels, Settings* s) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input_bmp_name << " not found";
    exit(-1);
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  if (s->verbose) LOG(INFO) << "len: " << len;

  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
  int row_size = 0;
  size_t header_size = 0;
  if (!ValidateBmpAndGetPixelOffset(img_bytes, width, height, channels,
                                    &row_size, &header_size)) {
    return std::vector<uint8_t>();
  }

  if (s->verbose)
    LOG(INFO) << "width, height, channels: " << *width << ", " << *height
              << ", " << *channels;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = img_bytes.data() + header_size;
  const int abs_height =
      static_cast<int>(*height < 0 ? -static_cast<int64_t>(*height) : *height);
  return decode_bmp(bmp_pixels, row_size, *width, abs_height, *channels,
                    top_down);
}

}  // namespace label_image
}  // namespace tflite
