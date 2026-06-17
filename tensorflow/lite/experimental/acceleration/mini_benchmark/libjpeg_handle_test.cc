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

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
namespace {

TEST(LibjpegHandleTest, LoadingSucceeds) {
  Status status;
  std::unique_ptr<LibjpegHandle> handle = LibjpegHandle::Create(status);
  EXPECT_TRUE(handle != nullptr);
  EXPECT_EQ(status.error_message, "");
  EXPECT_EQ(status.code, kTfLiteOk);
}

}  // namespace
}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
