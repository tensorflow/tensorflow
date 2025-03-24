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

#include <cstddef>

#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_gl_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert::internal {

class GlTexture {
 public:
  GlTexture(LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
            size_t size_bytes, LiteRtGLint layer,
            LiteRtGlTextureDeallocator deallocator);

  GlTexture(GlTexture&& other);

  ~GlTexture();

  LiteRtGLenum target() const;
  LiteRtGLuint id() const;
  LiteRtGLenum format() const;
  size_t size_bytes() const;
  LiteRtGLint layer() const;

 private:
  absl::Mutex mutex_;
#if LITERT_HAS_OPENGL_SUPPORT
  tflite::gpu::gl::GlTexture tflite_gl_texture_;
#endif  // LITERT_HAS_OPENGL_SUPPORT
  LiteRtGlTextureDeallocator deallocator_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GL_TEXTURE_H_
