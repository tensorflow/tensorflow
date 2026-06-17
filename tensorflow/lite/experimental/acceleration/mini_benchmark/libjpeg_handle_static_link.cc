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

#include <memory>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

std::unique_ptr<LibjpegHandle> LibjpegHandle::Create(Status &status) {
  std::unique_ptr<LibjpegHandle> handle(new LibjpegHandle());

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
}

LibjpegHandle::~LibjpegHandle() = default;

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
