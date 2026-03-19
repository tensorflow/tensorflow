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

#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture_helper.h"

namespace tflite {
namespace gpu {
namespace gl {

GlTexture::GlTexture(GlTexture&& texture)
    : GlTexture(texture.target_, texture.id_, texture.format_,
                texture.bytes_size_, texture.layer_, texture.owned_) {
  texture.owned_ = false;
}

GlTexture& GlTexture::operator=(GlTexture&& texture) {
  if (this != &texture) {
    Invalidate();

    target_ = texture.target_;
    format_ = texture.format_;
    bytes_size_ = texture.bytes_size_;
    layer_ = texture.layer_;
    owned_ = texture.owned_;
    id_ = texture.id_;
    texture.owned_ = false;
  }
  return *this;
}

GlTexture::~GlTexture() {
  Invalidate();
}

void GlTexture::Invalidate() {
  if (owned_ && id_ != GL_INVALID_INDEX) {
    TFLITE_GPU_CALL_GL(glDeleteTextures, 1, &id_).IgnoreError();
    id_ = GL_INVALID_INDEX;
  }
}

absl::Status GlTexture::BindImage(uint32_t index, GLenum access) const {
  return TFLITE_GPU_CALL_GL(glBindImageTexture, index, id_, /* level = */ 0,
                            /* layered = */ GL_TRUE, layer_, access, format_);
}

absl::Status GlTexture::BindAsReadonlyImage(uint32_t index) const {
  return BindImage(index, GL_READ_ONLY);
}

absl::Status GlTexture::BindAsWriteonlyImage(uint32_t index) const {
  return BindImage(index, GL_WRITE_ONLY);
}

absl::Status GlTexture::BindAsReadWriteImage(uint32_t index) const {
  return BindImage(index, GL_READ_WRITE);
}

absl::Status GlTexture::BindAsSampler2D(uint32_t index) const {
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glActiveTexture, GL_TEXTURE0 + index));
  return TFLITE_GPU_CALL_GL(glBindTexture, GL_TEXTURE_2D, id_);
}

namespace {

absl::Status SetTextureWrapAndFilter(GLenum target, GLenum texture_format) {
  if (texture_format == GL_RGBA32F) {
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_S, GL_REPEAT));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_T, GL_REPEAT));
    if (target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_3D) {
      RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                         GL_TEXTURE_WRAP_R, GL_REPEAT));
    }
    // Texture filtering is not available for GL_RGBA32F, hence explicitly
    // specifying GL_NEAREST param for texture (Otherwise, we can end up
    // sampling some incorrect values from texture.)
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  } else if (texture_format == GL_RGBA16F) {
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_S, GL_REPEAT));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_WRAP_T, GL_REPEAT));
    if (target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_3D) {
      RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                         GL_TEXTURE_WRAP_R, GL_REPEAT));
    }
    // Texture filtering is available for GL_RGBA16F, specifying that
    // explicitly improves quality for some operations like texture upscaling
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexParameteri, target,
                                       GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  }
  return absl::OkStatus();
}

absl::Status CreateReadOnlyRgba2dImageTexture(DataType data_type,
                                              const uint2& size,
                                              const void* data,
                                              size_t byte_size,
                                              GlTexture* gl_texture) {
  if (byte_size != /* RGBA=*/4 * SizeOf(data_type) * size.x * size.y) {
    return absl::InvalidArgumentError(
        "Creating image texture failed. Source data size is not matching "
        "expected dimensions.");
  }
  const GLenum kTarget = GL_TEXTURE_2D;
  const bool normalized = data_type == DataType::UINT8;
  GLenum internal_format = ToTextureInternalFormat(data_type, normalized);
  GLenum format = ToTextureFormat(data_type, normalized);
  GLenum type = ToTextureDataType(data_type);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage2D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexSubImage2D, kTarget, /* level = */ 0,
                                     0, 0, size.x, size.y, format, type, data));
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size, 0,
                          /*owned=*/true);
  return absl::OkStatus();
}

absl::Status CreateReadOnlyRgba3dImageTexture(DataType data_type,
                                              const uint3& size,
                                              const void* data,
                                              size_t byte_size,
                                              GlTexture* gl_texture) {
  if (byte_size != /* RGBA=*/4 * SizeOf(data_type) * size.x * size.y * size.z) {
    return absl::InvalidArgumentError(
        "Creating image texture failed. Source data is larger than dimensions "
        "product.");
  }
  const GLenum kTarget = GL_TEXTURE_2D_ARRAY;
  const bool normalized = data_type == DataType::UINT8;
  GLenum internal_format = ToTextureInternalFormat(data_type, normalized);
  GLenum format = ToTextureFormat(data_type, normalized);
  GLenum type = ToTextureDataType(data_type);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage3D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y, size.z));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexSubImage3D, kTarget, /* level = */ 0,
                                     0, 0, 0, size.x, size.y, size.z, format,
                                     type, data));
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size, 0,
                          /*owned=*/true);
  return absl::OkStatus();
}

}  // namespace

absl::Status CreateReadOnlyImageTexture(const uint2& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture) {
  return CreateReadOnlyRgba2dImageTexture(DataType::FLOAT32, size, data.data(),
                                          data.size() * sizeof(float),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTexture(const uint3& size,
                                        absl::Span<const float> data,
                                        GlTexture* gl_texture) {
  return CreateReadOnlyRgba3dImageTexture(DataType::FLOAT32, size, data.data(),
                                          data.size() * sizeof(float),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTextureU8(const uint2& size,
                                          absl::Span<const uint8_t> data,
                                          GlTexture* gl_texture) {
  return CreateReadOnlyRgba2dImageTexture(DataType::UINT8, size, data.data(),
                                          data.size() * sizeof(uint8_t),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTextureF16(const uint2& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture) {
  return CreateReadOnlyRgba2dImageTexture(DataType::FLOAT16, size, data.data(),
                                          data.size() * sizeof(uint16_t),
                                          gl_texture);
}

absl::Status CreateReadOnlyImageTextureF16(const uint3& size,
                                           absl::Span<const uint16_t> data,
                                           GlTexture* gl_texture) {
  return CreateReadOnlyRgba3dImageTexture(DataType::FLOAT16, size, data.data(),
                                          data.size() * sizeof(uint16_t),
                                          gl_texture);
}

absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint2& size,
                                             GlTexture* gl_texture) {
  const GLenum kTarget = GL_TEXTURE_2D;
  const bool normalized = data_type == DataType::UINT8;
  const GLenum internal_format = ToTextureInternalFormat(data_type, normalized);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage2D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y));
  size_t byte_size = /* RGBA = */ 4 * SizeOf(data_type) * size.x * size.y;
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size,
                          /* layer = */ 0,
                          /* owned = */ true);
  return absl::OkStatus();
}

absl::Status CreateReadWriteRgbaImageTexture(DataType data_type,
                                             const uint3& size,
                                             GlTexture* gl_texture) {
  const GLenum kTarget = GL_TEXTURE_2D_ARRAY;
  const bool normalized = data_type == DataType::UINT8;
  GLenum internal_format = ToTextureInternalFormat(data_type, normalized);
  gl_texture_internal::TextureId id;
  gl_texture_internal::TextureBinder binder(kTarget, id.id());
  RETURN_IF_ERROR(SetTextureWrapAndFilter(kTarget, internal_format));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glTexStorage3D, kTarget,
                                     /* num_levels = */ 1, internal_format,
                                     size.x, size.y, size.z));
  size_t byte_size =
      /* RGBA = */ 4 * SizeOf(data_type) * size.x * size.y * size.z;
  *gl_texture = GlTexture(kTarget, id.Release(), internal_format, byte_size,
                          /* layer = */ 0,
                          /* owned = */ true);
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
