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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_handle.h"

#ifdef TFLITE_ACCELERATION_USE_SYSTEM_LIBJPEG
#include <dlfcn.h>
#endif  // TFLITE_ACCELERATION_USE_SYSTEM_LIBJPEG

#include <stddef.h>
#include <stdio.h>

#include <memory>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

std::unique_ptr<LibjpegHandle> LibjpegHandle::Create(Status &status) {
  std::unique_ptr<LibjpegHandle> handle(new LibjpegHandle());

#ifndef TFLITE_ACCELERATION_USE_SYSTEM_LIBJPEG
  // Use statically linked implementation unless otherwise configured.
  handle->jpeg_std_error_ = jpeg_std_error;
  handle->jpeg_destroy_decompress_ = jpeg_destroy_decompress;
  handle->jpeg_create_decompress_ = jpeg_CreateDecompress;
  handle->jpeg_stdio_src_ = jpeg_stdio_src;
  handle->jpeg_read_header_ = jpeg_read_header;
  handle->jpeg_start_decompress_ = jpeg_start_decompress;
  handle->jpeg_read_scanlines_ = jpeg_read_scanlines;
  handle->jpeg_finish_decompress_ = jpeg_finish_decompress;
  status = {kTfLiteOk, ""};
  return handle;
#else  // TFLITE_ACCELERATION_USE_SYSTEM_LIBJPEG

  if (!(handle->libjpeg_ = dlopen("libjpeg.so", RTLD_NOW | RTLD_LOCAL))) {
    status = {kTfLiteError, "Failed to load dynamic library."};
    return nullptr;
  }

  // On Android S, the system libjpeg symbols have been prefixed
  // with 'chromium_' to avoid collisions.
#define LOAD(variable, symbol_name)                                      \
  do {                                                                   \
    static_assert(                                                       \
        std::is_same<decltype(variable), decltype(&symbol_name)>::value, \
        "Mismatched types");                                             \
    void *symbol = dlsym(handle->libjpeg_, #symbol_name);                \
    if (!symbol) {                                                       \
      symbol = dlsym(handle->libjpeg_, "chromium_" #symbol_name);        \
    }                                                                    \
    if (!symbol) {                                                       \
      status = {kTfLiteError,                                            \
                "Failed to dynamically load the method: " #symbol_name}; \
      return nullptr;                                                    \
    }                                                                    \
    variable = reinterpret_cast<decltype(variable)>(symbol);             \
  } while (0)

  LOAD(handle->jpeg_std_error_, jpeg_std_error);
  LOAD(handle->jpeg_destroy_decompress_, jpeg_destroy_decompress);
  LOAD(handle->jpeg_create_decompress_, jpeg_CreateDecompress);
  LOAD(handle->jpeg_stdio_src_, jpeg_stdio_src);
  LOAD(handle->jpeg_read_header_, jpeg_read_header);
  LOAD(handle->jpeg_start_decompress_, jpeg_start_decompress);
  LOAD(handle->jpeg_read_scanlines_, jpeg_read_scanlines);
  LOAD(handle->jpeg_finish_decompress_, jpeg_finish_decompress);
#undef LOAD
  status = {kTfLiteOk, ""};
  return handle;
#endif  // !TFLITE_ACCELERATION_USE_SYSTEM_LIBJPEG
}

LibjpegHandle::~LibjpegHandle() {
#ifdef TFLITE_ACCELERATION_USE_SYSTEM_LIBJPEG
  if (libjpeg_) {
    dlclose(libjpeg_);
  }
#endif  // TFLITE_ACCELERATION_USE_SYSTEM_LIBJPEG
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
