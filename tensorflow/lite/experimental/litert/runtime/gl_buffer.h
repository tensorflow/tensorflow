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

#if LITERT_HAS_OPENGL_SUPPORT

#include <GLES3/gl31.h>
#include <GLES3/gl32.h>

#include <cstddef>
#include <cstdlib>

#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace internal {

class GlBuffer {
 public:
  explicit GlBuffer(tflite::gpu::gl::GlBuffer&& tflite_gl_buffer)
      : tflite_gl_buffer_(std::move(tflite_gl_buffer)),
        deallocator_(nullptr),
        size_(tflite_gl_buffer_.bytes_size()) {}

  GlBuffer(GLenum target, GLuint id, size_t bytes_size, size_t offset,
           LiteRtGlBufferDeallocator deallocator) {
    if (deallocator != nullptr) {
      tflite_gl_buffer_ = tflite::gpu::gl::GlBuffer(
          target, id, bytes_size, offset, /*has_ownership=*/false);
      deallocator_ = std::move(deallocator);
    } else {
      tflite_gl_buffer_ = tflite::gpu::gl::GlBuffer(
          target, id, bytes_size, offset, /*has_ownership=*/true);
      deallocator_ = nullptr;
    }
  }
  GlBuffer(GlBuffer&& other) {
    tflite_gl_buffer_ = std::move(other.tflite_gl_buffer_);
    deallocator_ = std::move(other.deallocator_);
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  ~GlBuffer() {
    if (deallocator_ != nullptr) {
      deallocator_(reinterpret_cast<void*>(tflite_gl_buffer_.id()));
    }
    if (data_ != nullptr) {
      free(data_);
    };
  }

  static bool IsSupported() { return true; }
  static Expected<GlBuffer> Alloc(size_t bytes_size);

  template <typename T>
  Expected<T*> Lock();

  template <typename T>
  Expected<void> Unlock();

  GLenum target() const { return tflite_gl_buffer_.target(); }
  GLuint id() const { return tflite_gl_buffer_.id(); }
  size_t bytes_size() const { return tflite_gl_buffer_.bytes_size(); }
  size_t offset() const { return tflite_gl_buffer_.offset(); }

 private:
  absl::Mutex mutex_;
  tflite::gpu::gl::GlBuffer tflite_gl_buffer_;
  LiteRtGlBufferDeallocator deallocator_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_ = 0;
};

}  // namespace internal
}  // namespace litert

#endif  // LITERT_HAS_OPENGL_SUPPORT

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_BUFFER_H_
