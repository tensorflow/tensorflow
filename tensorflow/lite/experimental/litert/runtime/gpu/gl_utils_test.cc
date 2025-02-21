// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/gpu/gl_utils.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"

namespace {

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};

TEST(GlUtilsTest, AllocateSsbo) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices";
#endif

  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
  ABSL_CHECK_OK(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env));

  GLuint gl_buffer;
  size_t bytes = 1024;
  ASSERT_EQ(litert::AllocateSsbo(&gl_buffer, bytes), kLiteRtStatusOk);
}

TEST(GlUtilsTest, AllocateSsboBackedByAhwb) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices";
#endif

  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
  ABSL_CHECK_OK(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env));

  GLuint gl_buffer;
  size_t bytes = sizeof(kTensorData);

  const litert::RankedTensorType kTensorType(::kTensorType);
  auto tensor_buffer = litert::TensorBuffer::CreateManaged(
      kLiteRtTensorBufferTypeAhwb, kTensorType, bytes);

  auto ahwb_buffer = tensor_buffer->GetAhwb();
  ASSERT_TRUE(ahwb_buffer);

  ASSERT_EQ(
      litert::AllocateSsboBackedByAhwb(&gl_buffer, bytes, ahwb_buffer.Value()),
      kLiteRtStatusOk);
}

}  // namespace
