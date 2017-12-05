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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <unistd.h>

#define LOG(x) std::cerr

namespace tflite {
namespace label_image {

extern bool verbose;
extern int input_floating;
extern float input_mean;
extern float input_std;
extern std::string input_bmp_name;

uint8_t* decode_bmp(const uint8_t* input, const int row_size,
                    uint8_t* const output, const int width, const int height,
                    const int channels, bool top_down) {
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

uint8_t* read_bmp(const std::string& input_bmp_name, int& width, int& height,
                  int& channels) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input_bmp_name << " not found\n";
    exit(-1);
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  if (verbose) LOG(INFO) << "len: " << len << "\n";

  const uint8_t* img_bytes = new uint8_t[len];
  file.seekg(0, std::ios::beg);
  file.read((char*)img_bytes, len);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t*>(img_bytes + 10));
  width = *(reinterpret_cast<const int32_t*>(img_bytes + 18));
  height = *(reinterpret_cast<const int32_t*>(img_bytes + 22));
  const int32_t bpp = *(reinterpret_cast<const int32_t*>(img_bytes + 28));
  channels = bpp / 8;

  if (verbose)
    LOG(INFO) << "width, height, channels: " << width << ", " << height << ", "
              << channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * channels * width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (height < 0);

  // Decode image, allocating tensor once the image size is known
  uint8_t* output = new uint8_t[abs(height) * width * channels];
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, output, width, abs(height), channels,
                    top_down);
}

template <class T>
void downsize(T* out, uint8_t* in, int image_height, int image_width,
              int image_channels, int wanted_height, int wanted_width,
              int wanted_channels) {
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    uint8_t* in_row = in + (in_y * image_width * image_channels);
    T* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      uint8_t* in_pixel = in_row + (in_x * image_channels);
      T* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        if (input_floating)
          out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
        else
          out_pixel[c] = in_pixel[c];
      }
    }
  }
}

// explicit instantiation so that we can use them otherwhere
template void downsize<uint8_t>(unsigned char*, unsigned char*, int, int, int,
                                int, int, int);
template void downsize<float>(float*, unsigned char*, int, int, int, int, int,
                              int);

}  // label_image
}  // tflite
