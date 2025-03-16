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

TEST(Buffer, GlBufferAlloc) {
  if (!GlBuffer::IsSupported()) {
    GTEST_SKIP() << "OpenGL buffers are not supported on this platform";
  }
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  auto buffer = GlBuffer::Alloc(4 * sizeof(float));
  ASSERT_TRUE(buffer);
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
  LITERT_ASSERT_OK_AND_ASSIGN(GlBuffer gl_buffer,
                              GlBuffer::AllocFromAhwbBuffer(ahwb_buffer));
  // TODO(gcarranza): Add test to verify buffer content is the same.
}
#endif  // LITERT_HAS_AHWB_SUPPORT

}  // namespace
}  // namespace internal
}  // namespace litert

#endif  // LITERT_HAS_OPENGL_SUPPORT
