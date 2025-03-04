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

#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_texture.h"

namespace litert {
namespace internal {

GlTexture::GlTexture(GLenum target, GLuint id, GLenum format, size_t size_bytes,
                     GLint layer, LiteRtGlTextureDeallocator deallocator) {
  if (deallocator != nullptr) {
    tflite_gl_texture_ = tflite::gpu::gl::GlTexture(
        target, id, format, size_bytes, layer, /*has_ownership=*/false);
    deallocator_ = std::move(deallocator);
  } else {
    tflite_gl_texture_ = tflite::gpu::gl::GlTexture(
        target, id, format, size_bytes, layer, /*has_ownership=*/true);
    deallocator_ = nullptr;
  }
}

GlTexture::GlTexture(GlTexture&& other) {
  tflite_gl_texture_ = std::move(other.tflite_gl_texture_);
  deallocator_ = std::move(other.deallocator_);
}

GlTexture::~GlTexture() {
  if (deallocator_ != nullptr) {
    deallocator_(reinterpret_cast<void*>(tflite_gl_texture_.id()));
  }
}

}  // namespace internal
}  // namespace litert
#endif  // LITERT_HAS_OPENGL_SUPPORT
