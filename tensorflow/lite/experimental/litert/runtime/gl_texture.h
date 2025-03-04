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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_TEXTURE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_TEXTURE_H_

#if LITERT_HAS_OPENGL_SUPPORT

#include <GLES3/gl31.h>
#include <GLES3/gl32.h>

#include <cstddef>

#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"

namespace litert {
namespace internal {

class GlTexture {
 public:
  // GlTexture() = default;

  GlTexture(GLenum target, GLuint id, GLenum format, size_t size_bytes,
            GLint layer, LiteRtGlTextureDeallocator deallocator);

  GlTexture(GlTexture&& other);

  ~GlTexture();

  GLenum target() const { return tflite_gl_texture_.target(); }
  GLuint id() const { return tflite_gl_texture_.id(); }
  GLenum format() const { return tflite_gl_texture_.format(); }
  size_t size_bytes() const { return tflite_gl_texture_.bytes_size(); }
  GLint layer() const { return tflite_gl_texture_.layer(); }

 private:
  absl::Mutex mutex_;
  tflite::gpu::gl::GlTexture tflite_gl_texture_;
  LiteRtGlTextureDeallocator deallocator_;
};

}  // namespace internal
}  // namespace litert

#endif  // LITERT_HAS_OPENGL_SUPPORT

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_TEXTURE_H_
