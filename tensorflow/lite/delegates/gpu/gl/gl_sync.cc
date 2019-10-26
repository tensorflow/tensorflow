/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/gl_sync.h"

#ifdef __ARM_ACLE
#include <arm_acle.h>
#endif  // __ARM_ACLE

#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

namespace tflite {
namespace gpu {
namespace gl {

Status GlSyncWait() {
  GlSync sync;
  RETURN_IF_ERROR(GlSync::NewSync(&sync));
  // Flush sync and loop afterwards without it.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  while (true) {
    switch (status) {
      case GL_TIMEOUT_EXPIRED:
        break;
      case GL_CONDITION_SATISFIED:
      case GL_ALREADY_SIGNALED:
        return OkStatus();
      case GL_WAIT_FAILED:
        return GetOpenGlErrors();
    }
    status = glClientWaitSync(sync.sync(), 0, /* timeout ns = */ 10000000);
  }
  return OkStatus();
}

Status GlActiveSyncWait() {
  GlSync sync;
  RETURN_IF_ERROR(GlSync::NewSync(&sync));
  // Since creating a Sync object is itself a GL command it *must* be flushed.
  // Otherwise glGetSynciv may never succeed. Perform a flush with
  // glClientWaitSync call.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  switch (status) {
    case GL_TIMEOUT_EXPIRED:
      break;
    case GL_CONDITION_SATISFIED:
    case GL_ALREADY_SIGNALED:
      return OkStatus();
    case GL_WAIT_FAILED:
      return GetOpenGlErrors();
  }

  // Start active loop.
  GLint result = GL_UNSIGNALED;
  while (true) {
    glGetSynciv(sync.sync(), GL_SYNC_STATUS, sizeof(GLint), nullptr, &result);
    if (result == GL_SIGNALED) {
      return OkStatus();
    }
#ifdef __ARM_ACLE
    // Try to save CPU power by yielding CPU to another thread.
    __yield();
#endif
  }
}

Status GlShaderSync::NewSync(GlShaderSync* gl_sync) {
  GlShaderSync sync;
  RETURN_IF_ERROR(CreatePersistentBuffer(sizeof(int), &sync.flag_buffer_));
  static const std::string* kCode = new std::string(R"(#version 310 es
  layout(local_size_x = 1, local_size_y = 1) in;
  layout(std430) buffer;
  layout(binding = 0) buffer Output {
    int elements[];
  } output_data;
  void main() {
    output_data.elements[0] = 1;
  })");
  GlShader shader;
  RETURN_IF_ERROR(GlShader::CompileShader(GL_COMPUTE_SHADER, *kCode, &shader));
  RETURN_IF_ERROR(GlProgram::CreateWithShader(shader, &sync.flag_program_));
  *gl_sync = std::move(sync);
  return OkStatus();
}

// How it works: GPU writes a buffer and CPU checks the buffer value to be
// changed. The buffer is accessible for writing by GPU and reading by CPU
// simultaneously - persistent buffer or buffer across shild context can be used
// for that.
Status GlShaderSync::Wait() {
  if (!flag_buffer_.is_valid()) {
    return UnavailableError("GlShaderSync is not initialized.");
  }
  RETURN_IF_ERROR(flag_buffer_.BindToIndex(0));
  volatile int* flag_ptr_ = reinterpret_cast<int*>(flag_buffer_.data());
  *flag_ptr_ = 0;
  RETURN_IF_ERROR(flag_program_.Dispatch({1, 1, 1}));
  // glFlush must be called to upload GPU task. Adreno won't start executing
  // the task without glFlush.
  glFlush();
  // Wait for the value is being updated by the shader.
  while (*flag_ptr_ != 1) {
  }
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
