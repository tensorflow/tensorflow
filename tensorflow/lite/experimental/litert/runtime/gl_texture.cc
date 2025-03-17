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

#include "tensorflow/lite/experimental/litert/runtime/gl_texture.h"

#include <cstddef>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_gl_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert {
namespace internal {

LiteRtGLenum GlTexture::target() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return tflite_gl_texture_.target();
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::target() is not supported");
  return 0;
}

LiteRtGLuint GlTexture::id() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return tflite_gl_texture_.id();
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::id() is not supported");
  return 0;
}

LiteRtGLenum GlTexture::format() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return tflite_gl_texture_.format();
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::format() is not supported");
  return 0;
}

size_t GlTexture::size_bytes() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return tflite_gl_texture_.bytes_size();
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::size_bytes() is not supported");
  return 0;
}

LiteRtGLint GlTexture::layer() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return tflite_gl_texture_.layer();
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::layer() is not supported");
  return 0;
#endif
}

GlTexture::GlTexture(LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
                     size_t size_bytes, LiteRtGLint layer,
                     LiteRtGlTextureDeallocator deallocator) {
#if LITERT_HAS_OPENGL_SUPPORT
  if (deallocator != nullptr) {
    tflite_gl_texture_ = tflite::gpu::gl::GlTexture(
        target, id, format, size_bytes, layer, /*has_ownership=*/false);
    deallocator_ = std::move(deallocator);
  } else {
    tflite_gl_texture_ = tflite::gpu::gl::GlTexture(
        target, id, format, size_bytes, layer, /*has_ownership=*/true);
    deallocator_ = nullptr;
  }
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::GlTexture() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

GlTexture::GlTexture(GlTexture&& other) {
#if LITERT_HAS_OPENGL_SUPPORT
  tflite_gl_texture_ = std::move(other.tflite_gl_texture_);
  deallocator_ = std::move(other.deallocator_);
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::GlTexture() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

GlTexture::~GlTexture() {
#if LITERT_HAS_OPENGL_SUPPORT
  if (deallocator_ != nullptr) {
    deallocator_(reinterpret_cast<void*>(tflite_gl_texture_.id()));
  }
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::~GlTexture() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

}  // namespace internal
}  // namespace litert
