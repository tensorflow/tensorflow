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

using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

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
  // Write to AHWB on CPU.
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

TEST(Buffer, NegativeFenceAhwbRead) {
  LITERT_ASSERT_OK_AND_ASSIGN(AhwbBuffer ahwb_buffer,
                              AhwbBuffer::Alloc(4 * sizeof(float)));

  LiteRtEventT event;
  LITERT_ASSERT_OK_AND_ASSIGN(int fence_fd, event.GetSyncFenceFd());
  ASSERT_EQ(fence_fd, -1);
  // Since fence is -1, there should be no wait on fence.
  LITERT_ASSERT_OK_AND_ASSIGN(void* ahwb_host_data,
                              AhwbBuffer::Lock(ahwb_buffer.ahwb, &event));
  ASSERT_TRUE(ahwb_host_data != nullptr);
  LITERT_ASSERT_OK(AhwbBuffer::Unlock(ahwb_buffer.ahwb));
}

// Utility function to fill the GPU buffer.
void FillGlBuffer(GLuint id, std::size_t size) {
  std::string shader_source = R"( #version 310 es
    precision highp float;
    layout(local_size_x = 1, local_size_y = 1) in;
    layout(std430, binding = 0) buffer Output {float elements[];} output_data;
    void main() {
      uint v = gl_GlobalInvocationID.x * 2u;
      output_data.elements[v] = float(v) / 10.0;
      output_data.elements[v + 1u] = float(v + 1u) / 10.0;
    })";
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  const GLchar* sources[] = {shader_source.c_str()};
  glShaderSource(shader, 1, sources, nullptr);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glCompileShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  GLuint to_buffer_program = glCreateProgram();
  glAttachShader(to_buffer_program, shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glLinkProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, id);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glUseProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDispatchCompute(size / 2, 1, 1);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
}

TEST(Buffer, GpuWriteAhwbRead) {
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  LITERT_ASSERT_OK_AND_ASSIGN(AhwbBuffer ahwb_buffer,
                              AhwbBuffer::Alloc(4 * sizeof(float)));
  // Write to AHWB on CPU.
  LITERT_ASSERT_OK_AND_ASSIGN(
      void* ahwb_host_data,
      litert::internal::AhwbBuffer::Lock(ahwb_buffer.ahwb));
  std::memcpy(ahwb_host_data, kTensorData, sizeof(kTensorData));
  LITERT_ASSERT_OK(litert::internal::AhwbBuffer::Unlock(ahwb_buffer.ahwb));

  // Create GL buffer from AHWB.
  LITERT_ASSERT_OK_AND_ASSIGN(GlBuffer gl_buffer,
                              GlBuffer::AllocFromAhwbBuffer(ahwb_buffer));

  // Schedule GPU write to GL buffer.
  FillGlBuffer(gl_buffer.id(), 4);

  // Create EGL sync and fence before AHWB read.
  LITERT_ASSERT_OK_AND_ASSIGN(int native_fence,
                              GlBuffer::CreateEglSyncAndFence());

  // Wrap native fence in LiteRT event.
  LiteRtEventT gpu_write_event = {.fd = native_fence, .owns_fd = true};

  // Read from AHWB on CPU, waiting for GPU write to complete.
  LITERT_ASSERT_OK_AND_ASSIGN(
      void* ahwb_host_data_after_write_data,
      AhwbBuffer::Lock(ahwb_buffer.ahwb, &gpu_write_event));
  ASSERT_NE(ahwb_host_data_after_write_data, nullptr);
  auto ahwb_host_data_after_write = absl::MakeSpan(
      reinterpret_cast<float*>(ahwb_host_data_after_write_data), 4);
  // Check that the data is the same as the GPU write.
  std::vector<float> expected_data = {0.0f, 0.1f, 0.2f, 0.3f};
  EXPECT_THAT(ahwb_host_data_after_write,
              Pointwise(FloatNear(1e-5), expected_data));
  LITERT_ASSERT_OK(AhwbBuffer::Unlock(ahwb_buffer.ahwb));
}

#endif  // LITERT_HAS_AHWB_SUPPORT

}  // namespace
}  // namespace internal
}  // namespace litert

#endif  // LITERT_HAS_OPENGL_SUPPORT
