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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_HANDLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_HANDLE_H_

#include <stddef.h>
#include <stdio.h>

#include <memory>
#include <string>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// This class offers a handle to Libjpeg shared object library on Android.
// It offers pointers to functions in Libjpeg that are required for decoding
// JPEG images.
// TODO(b/172544567): Support Apple.
class LibjpegHandle {
 public:
  // Factory for creating an initialised instance of LibjpegHandle.
  // Loads the libjpeg dynamic library and gets handle to all the functions
  // required for decompressing JPEGs. Returns an initialised instance of
  // LibjpegHandle if successful, else nullptr. Stores initialisation status in
  // `status`.
  static std::unique_ptr<LibjpegHandle> Create(Status& status);
  // Closes the dynamic library loaded in libjpeg_.
  ~LibjpegHandle();
  LibjpegHandle(LibjpegHandle const&) = delete;
  LibjpegHandle& operator=(const LibjpegHandle&) = delete;
  LibjpegHandle(LibjpegHandle&& LibjpegHandle) = delete;
  LibjpegHandle& operator=(LibjpegHandle&& other) = delete;
  // Based on our analysis of Android devices in the ODML lab, it is reasonable
  // to expect 62 (6b) as the version of libjpeg on all Android devices from SDK
  // 22 onwards.
  static const int kLibjpegVersion = 62;
  // Definitions of the functions below can be found in
  // third_party/libjpeg_turbo/src/jpeglib.h
  struct jpeg_error_mgr* (*jpeg_std_error_)(struct jpeg_error_mgr*);
  void (*jpeg_destroy_decompress_)(j_decompress_ptr);
  void (*jpeg_create_decompress_)(j_decompress_ptr, int, size_t);
  void (*jpeg_stdio_src_)(j_decompress_ptr, FILE*);
  int (*jpeg_read_header_)(j_decompress_ptr, boolean);
  boolean (*jpeg_start_decompress_)(j_decompress_ptr);
  unsigned int (*jpeg_read_scanlines_)(j_decompress_ptr, JSAMPARRAY,
                                       JDIMENSION);
  boolean (*jpeg_finish_decompress_)(j_decompress_ptr);

 private:
  LibjpegHandle() {}
  void* libjpeg_ = nullptr;
};

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_HANDLE_H_
