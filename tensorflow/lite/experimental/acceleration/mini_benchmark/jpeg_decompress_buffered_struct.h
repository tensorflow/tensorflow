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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_DECOMPRESS_BUFFERED_STRUCT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_DECOMPRESS_BUFFERED_STRUCT_H_

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// May provide an extra buffer of characters beyond the `jpeg_decompress_struct`
// for some builds of Libjpeg Dynamic Library on Android that expect a larger
// struct than we were compiled with. Zeroes out any allocated bytes beyond
// sizeof(jpeg_decompress_struct). This class is exclusively used by
// decode_jpeg.cc to resize `jpeg_decompress_struct`. This is to fix a struct
// mismatch problem. See go/libjpeg-android for more details.
class JpegDecompressBufferedStruct {
 public:
  explicit JpegDecompressBufferedStruct(std::size_t expected_size)
      : resized_size_(std::max(sizeof(jpeg_decompress_struct), expected_size)),
        buffer_(reinterpret_cast<char*>(malloc(resized_size_))) {
    // Note: Malloc guarantees alignment for 8 bytes. Hence, using malloc
    // instead of aligned_alloc.
    // https://www.gnu.org/software/libc/manual/html_node/Aligned-Memory-Blocks.html
    // alignof(jpeg_decompress_struct) is 8 bytes both on 32 and 64 bit.
    // It's safe to align the buffered struct as
    // alignof(jpeg_decompress_struct). This is because we only access the
    // `jpeg_common_fields` fields of `jpeg_decompress_struct`, all of which are
    // pointers. The alignment of these pointer fields is 8 and 4 bytes for 64
    // bit and 32 bit platforms respectively. Since
    // alignof(jpeg_decompress_struct) is 8 bytes on both platforms, accessing
    // these fields shouldn't be a problem.
    // Zero out any excess bytes. Zero-initialization is safe for the bytes
    // beyond sizeof(jpeg_decompress_struct) because both the dynamic library
    // and the implementation in decode_jpeg.cc limit their access only to
    // `jpeg_common_fields` in `jpeg_decompress_struct`.
    while (--expected_size >= sizeof(jpeg_decompress_struct)) {
      buffer_[expected_size] = 0;
    }
  }
  ~JpegDecompressBufferedStruct() { std::free(buffer_); }
  JpegDecompressBufferedStruct(const JpegDecompressBufferedStruct&) = delete;
  JpegDecompressBufferedStruct& operator=(const JpegDecompressBufferedStruct&) =
      delete;
  jpeg_decompress_struct* get() const {
    return reinterpret_cast<jpeg_decompress_struct*>(buffer_);
  }
  int const size() { return resized_size_; }
  const char* buffer() { return buffer_; }

 private:
  int resized_size_;
  char* const buffer_;
};

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_JPEG_DECOMPRESS_BUFFERED_STRUCT_H_
