// Copyright 2025 Google LLC.
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

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_buffer.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

#if LITERT_HAS_AHWB_SUPPORT
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#endif  // LITERT_HAS_AHWB_SUPPORT

namespace litert {
namespace internal {
namespace {

constexpr const float kTensorData[] = {10, 20, 30, 40};

TEST(Buffer, GlBufferAlloc) {
  if (!GlBuffer::IsSupported()) {
    GTEST_SKIP() << "OpenGL buffers are not supported on this platform";
  }
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  auto buffer = GlBuffer::Alloc(4 * sizeof(float));
  ASSERT_TRUE(buffer);

  // Test lock and unlock.
  LITERT_ASSERT_OK_AND_ASSIGN(float* data, buffer->Lock<float>());
  EXPECT_NE(data, nullptr);
  LITERT_ASSERT_OK(buffer->Unlock<float>());
}

#if LITERT_HAS_AHWB_SUPPORT
TEST(Buffer, GlBufferAllocFromAhwb) {
  if (!GlBuffer::IsSupported()) {
    GTEST_SKIP() << "OpenGL buffers are not supported on this platform";
  }
  // TODO(gcarranza): Incorporate this into LiteRT environment.
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  LITERT_ASSERT_OK_AND_ASSIGN(AhwbBuffer ahwb_buffer,
                              AhwbBuffer::Alloc(4 * sizeof(float)));
  // Write to AHWB.
  LITERT_ASSERT_OK_AND_ASSIGN(
      void* ahwb_host_data,
      litert::internal::AhwbBuffer::Lock(ahwb_buffer.ahwb));
  std::memcpy(ahwb_host_data, kTensorData, sizeof(kTensorData));
  LITERT_ASSERT_OK(litert::internal::AhwbBuffer::Unlock(ahwb_buffer.ahwb));

  // Create GL buffer from AHWB.
  LITERT_ASSERT_OK_AND_ASSIGN(GlBuffer gl_buffer,
                              GlBuffer::AllocFromAhwbBuffer(ahwb_buffer));

  // Read from GL buffer backed by AHWB.
  LITERT_ASSERT_OK_AND_ASSIGN(float* gl_host_data, gl_buffer.Lock<float>());
  ASSERT_NE(gl_host_data, nullptr);
  EXPECT_EQ(std::memcmp(gl_host_data, kTensorData, sizeof(kTensorData)), 0);
  LITERT_EXPECT_OK(gl_buffer.Unlock<float>());
}
#endif  // LITERT_HAS_AHWB_SUPPORT

}  // namespace
}  // namespace internal
}  // namespace litert

#endif  // LITERT_HAS_OPENGL_SUPPORT
