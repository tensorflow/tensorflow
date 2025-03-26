/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Functions to read and write images in PNG format.
//
// The advantage over image/codec/png{enc,dec}ocder.h is that this library
// supports both 8 and 16 bit images.
//
// The decoding routine accepts binary image data as a StringPiece.  These are
// implicitly constructed from strings or char* so they're completely
// transparent to the caller.  They're also very cheap to construct so this
// doesn't introduce any additional overhead.
//
// The primary benefit of StringPieces being, in this case, that APIs already
// returning StringPieces (e.g., Bigtable Scanner) or Cords (e.g., IOBuffer;
// only when they're flat, though) or protocol buffer fields typed to either of
// these can be decoded without copying the data into a C++ string.

#ifndef TENSORFLOW_CORE_LIB_PNG_PNG_IO_H_
#define TENSORFLOW_CORE_LIB_PNG_PNG_IO_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "tensorflow/core/platform/png.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace png {

// Handy container for decoding information and struct pointers
struct DecodeContext {
  const uint8* data;
  int data_left;
  png_structp png_ptr;
  png_infop info_ptr;
  png_uint_32 width, height;
  int num_passes;
  int color_type;
  int bit_depth;
  int channels;
  bool need_to_synthesize_16;
  bool error_condition;
  DecodeContext() : png_ptr(nullptr), info_ptr(nullptr) {}
};

bool DecodeHeader(absl::string_view png_string, int* width, int* height,
                  int* components, int* channel_bit_depth,
                  std::vector<std::pair<std::string, std::string> >* metadata);

// Sample usage for reading PNG:
//
// string png_string;  /* fill with input PNG format data */
// DecodeContext context;
// CHECK(CommonInitDecode(png_string, 3 /*RGB*/, 8 /*uint8*/, &context));
// char* image_buffer = new char[3*context.width*context.height];
// CHECK(CommonFinishDecode(absl::bit_cast<png_byte*>(image_buffer),
//       3*context.width /*stride*/, &context));
//
// desired_channels may be 0 to detected it from the input.

bool CommonInitDecode(absl::string_view png_string, int desired_channels,
                      int desired_channel_bits, DecodeContext* context);

bool CommonFinishDecode(png_bytep data, int row_bytes, DecodeContext* context);

// Normally called automatically from CommonFinishDecode.  If CommonInitDecode
// is called but not CommonFinishDecode, call this to clean up.  Safe to call
// extra times.
void CommonFreeDecode(DecodeContext* context);

// Sample usage for writing PNG:
//
// uint16* image_buffer = new uint16[width*height];  /* fill with pixels */
// string png_string;
// CHECK(WriteImageToBuffer(image_buffer, width, height, 2*width /*stride*/,
//       1 /*gray*/, 16 /*uint16*/, &png_string, NULL));
//
// compression is in [-1,9], where 0 is fast and weak compression, 9 is slow
// and strong, and -1 is the zlib default.

template <typename T>
bool WriteImageToBuffer(
    const void* image, int width, int height, int row_bytes, int num_channels,
    int channel_bits, int compression, T* png_string,
    const std::vector<std::pair<std::string, std::string> >* metadata);

// Explicit instantiations defined in png_io.cc.
extern template bool WriteImageToBuffer<std::string>(
    const void* image, int width, int height, int row_bytes, int num_channels,
    int channel_bits, int compression, std::string* png_string,
    const std::vector<std::pair<std::string, std::string> >* metadata);
extern template bool WriteImageToBuffer<tstring>(
    const void* image, int width, int height, int row_bytes, int num_channels,
    int channel_bits, int compression, tstring* png_string,
    const std::vector<std::pair<std::string, std::string> >* metadata);

}  // namespace png
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_PNG_PNG_IO_H_
