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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_H_

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// Texture is an RAII wrapper for OpenGL texture object.
// See https://www.khronos.org/opengl/wiki/Texture for more information.
//
// Texture is moveable but not copyable.
class GlTexture {
 public:
  // Creates invalid texture.
  GlTexture()
      : GlTexture(GL_INVALID_ENUM, GL_INVALID_INDEX, GL_INVALID_ENUM, 0, 0,
                  false) {}

  GlTexture(GLenum target, GLuint id, GLenum format, size_t bytes_size,
            GLint layer, bool owned)
      : id_(id),
        target_(target),
        format_(format),
        bytes_size_(bytes_size),
        layer_(layer),
        owned_(owned) {}

  // Move-only
  GlTexture(GlTexture&& texture);
  GlTexture& operator=(GlTexture&& texture);
  GlTexture(const GlTexture&) = delete;
  GlTexture& operator=(const GlTexture&) = delete;

  ~GlTexture();

  // Binds a texture as an image to the given index.
  absl::Status BindAsReadonlyImage(uint32_t index) const;

  // Bind texture as an image for write access at given index.
  absl::Status BindAsWriteonlyImage(uint32_t index) const;

  // Bind texture as an image for read-write access at given index.
  absl::Status BindAsReadWriteImage(uint32_t index) const;

  // Binds a texture as a sampler to the given index.
  absl::Status BindAsSampler2D(uint32_t index) const;

  GLenum target() const { return target_; }

  GLuint id() const { return id_; }

  GLenum format() const { return format_; }

  GLint layer() const { return layer_; }

  bool is_valid() const { return id_ != GL_INVALID_INDEX; }

  size_t bytes_size() const { return bytes_size_; }

  // @return true if this object actually owns corresponding GL buffer
  //         and manages it's lifetime.
  bool has_ownership() const { return owned_; }

 private:
  void Invalidate();

  absl::Status BindImage(uint32_t index, GLenum access) const;

  GLuint id_;
  GLenum target_;
  GLenum format_;
  size_t bytes_size_;
  GLint layer_;
  bool owned_;
};

// Creates new 2D image texture that will be filled with float32 data once which
// will be used for reading.
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTexture(const uint2& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture);

// Creates new 2D image texture that will be filled with float16 data once which
// will be used for reading.
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTextureF16(const uint2& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture);

// Creates new 2D image texture that will be filled with uint8 data once which
// will be used for reading.
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTextureU8(const uint2& size,
                                          absl::Span<const uint8_t> data,
                                          GlTexture* gl_texture);

// Creates new 3D RGBA image texture that will be filled with float32 data once
// which will be used for reading.
//
// @param size defines 3D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTexture(const uint3& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture);

// Creates new 3D RGBA image texture that will be filled with float16 data once
// which will be used for reading.
//
// @param size defines 3D image texture size where each pixel is RGBA.
absl::Status CreateReadOnlyImageTextureF16(const uint3& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture);

// Creates new RGBA 2D image texture
//
// @param size defines 2D image texture size where each pixel is RGBA.
absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint2& size,
                                             GlTexture* gl_texture);

// Creates new RGBA 3D image texture
//
// @param size defines 3D image texture size where each pixel is RGBA.
absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint3& size,
                                             GlTexture* gl_texture);

namespace gl_texture_internal {

// RAII for creating and/or owning texture id.
class TextureId {
 public:
  TextureId() : id_(GL_INVALID_INDEX) {
    TFLITE_GPU_CALL_GL(glGenTextures, 1 /* number of textures*/, &id_)
        .IgnoreError();
  }

  explicit TextureId(GLuint id) : id_(id) {}

  ~TextureId() {
    if (id_ != GL_INVALID_INDEX) {
      TFLITE_GPU_CALL_GL(glDeleteTextures, 1, &id_).IgnoreError();
    }
  }

  GLuint id() const { return id_; }

  GLuint Release() {
    GLuint id = GL_INVALID_INDEX;
    std::swap(id, id_);
    return id;
  }

 private:
  GLuint id_;
};

// RAII for binding and unbinding a texture.
class TextureBinder {
 public:
  TextureBinder(GLenum target, GLuint id) : target_(target) {
    TFLITE_GPU_CALL_GL(glBindTexture, target_, id).IgnoreError();
  }

  ~TextureBinder() {
    TFLITE_GPU_CALL_GL(glBindTexture, target_, 0).IgnoreError();
  }

 private:
  const GLenum target_;
};

}  // namespace gl_texture_internal
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_H_
