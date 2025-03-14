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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_BUFFER_H_

#include <cstddef>
#include <cstdlib>

#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_gl_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_AHWB_SUPPORT
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#endif  // LITERT_HAS_AHWB_SUPPORT

namespace litert::internal {

class GlBuffer {
 public:
#if LITERT_HAS_OPENGL_SUPPORT
  explicit GlBuffer(tflite::gpu::gl::GlBuffer&& tflite_gl_buffer
#if LITERT_HAS_AHWB_SUPPORT
                    ,
                    AHardwareBuffer* ahwb = nullptr
#endif  // LITERT_HAS_AHWB_SUPPORT
                    )
      : tflite_gl_buffer_(std::move(tflite_gl_buffer)),
        deallocator_(nullptr),
        size_bytes_(tflite_gl_buffer.bytes_size())
#if LITERT_HAS_AHWB_SUPPORT
        ,
        ahwb_(ahwb)
#endif  // LITERT_HAS_AHWB_SUPPORT
  {
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT

  GlBuffer(LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes,
           size_t offset, LiteRtGlBufferDeallocator deallocator);

  GlBuffer(GlBuffer&& other);

  ~GlBuffer();

  static bool IsSupported() { return true; }
  static Expected<GlBuffer> Alloc(size_t size_bytes);

#if LITERT_HAS_AHWB_SUPPORT
  static Expected<GlBuffer> AllocFromAhwbBuffer(AhwbBuffer& ahwb_buffer);
#endif  // LITERT_HAS_AHWB_SUPPORT

  template <typename T>
  Expected<T*> Lock();

  template <typename T>
  Expected<void> Unlock();

  LiteRtGLenum target() const;
  LiteRtGLuint id() const;
  size_t size_bytes() const;
  size_t offset() const;

  // Creates an EGL sync object on the GPU command queue and returns a native
  // fence associated with the sync object.
  // Note: This function assumes that all GL operations have been already added
  // to the GPU command queue.
  static Expected<int> CreateEglSyncAndFence();

 private:
  absl::Mutex mutex_;
#if LITERT_HAS_OPENGL_SUPPORT
  tflite::gpu::gl::GlBuffer tflite_gl_buffer_;
#endif  // LITERT_HAS_OPENGL_SUPPORT
  LiteRtGlBufferDeallocator deallocator_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_bytes_ = 0;
#if LITERT_HAS_AHWB_SUPPORT
  AHardwareBuffer* ahwb_ = nullptr;
#endif  // LITERT_HAS_AHWB_SUPPORT
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_BUFFER_H_
