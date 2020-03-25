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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_SYNC_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_SYNC_H_

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// RAII wrapper for OpenGL GLsync object.
// See https://www.khronos.org/opengl/wiki/Sync_Object for more information.
//
// GlSync is moveable but not copyable.
class GlSync {
 public:
  static Status NewSync(GlSync* gl_sync) {
    GLsync sync;
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glFenceSync, &sync,
                                       GL_SYNC_GPU_COMMANDS_COMPLETE, 0));
    *gl_sync = GlSync(sync);
    return OkStatus();
  }

  // Creates invalid object.
  GlSync() : GlSync(nullptr) {}

  // Move-only
  GlSync(GlSync&& sync) : sync_(sync.sync_) { sync.sync_ = nullptr; }

  GlSync& operator=(GlSync&& sync) {
    if (this != &sync) {
      Invalidate();
      std::swap(sync_, sync.sync_);
    }
    return *this;
  }

  GlSync(const GlSync&) = delete;
  GlSync& operator=(const GlSync&) = delete;

  ~GlSync() { Invalidate(); }

  const GLsync sync() const { return sync_; }

 private:
  explicit GlSync(GLsync sync) : sync_(sync) {}

  void Invalidate() {
    if (sync_) {
      glDeleteSync(sync_);
      sync_ = nullptr;
    }
  }

  GLsync sync_;
};

// Waits until GPU is done with processing.
Status GlSyncWait();

// Waits until all commands are flushed and then performs active waiting by
// spinning a thread and checking sync status. It leads to shorter wait time
// (up to tens of ms) but consumes more CPU.
Status GlActiveSyncWait();

// CPU checks the value in the buffer that is going to be written by GPU. The
// persistent buffer is used for the simultaneous access to the buffer by GPU
// and CPU. The instance remains invalid if persistent buffer OpenGL extension
// is not supported by the device.
class GlShaderSync {
 public:
  static Status NewSync(GlShaderSync* gl_sync);
  GlShaderSync() {}
  Status Wait();

 private:
  GlProgram flag_program_;
  GlPersistentBuffer flag_buffer_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_SYNC_H_
