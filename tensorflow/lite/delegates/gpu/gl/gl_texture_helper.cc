/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/gl_texture_helper.h"

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

GLenum ToTextureFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::INT8:
    case DataType::UINT8:
      return normalized ? GL_RGBA : GL_RGBA_INTEGER;
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::INT16:
    case DataType::INT32:
      return GL_RGBA_INTEGER;
    case DataType::FLOAT16:
    case DataType::FLOAT32:
      return GL_RGBA;
    default:
      return 0;
  }
}

GLenum ToTextureInternalFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::UINT8:
      return normalized ? GL_RGBA8 : GL_RGBA8UI;
    case DataType::INT8:
      return normalized ? GL_RGBA8_SNORM : GL_RGBA8I;
    case DataType::UINT16:
      return GL_RGBA16UI;
    case DataType::UINT32:
      return GL_RGBA32UI;
    case DataType::INT16:
      return GL_RGBA16I;
    case DataType::INT32:
      return GL_RGBA32I;
    case DataType::FLOAT16:
      return GL_RGBA16F;
    case DataType::FLOAT32:
      return GL_RGBA32F;
    default:
      return 0;
  }
}

GLenum ToTextureDataType(DataType type) {
  switch (type) {
    case DataType::UINT8:
      return GL_UNSIGNED_BYTE;
    case DataType::INT8:
      return GL_BYTE;
    case DataType::UINT16:
      return GL_UNSIGNED_SHORT;
    case DataType::UINT32:
      return GL_UNSIGNED_INT;
    case DataType::INT16:
      return GL_SHORT;
    case DataType::INT32:
      return GL_INT;
    case DataType::FLOAT16:
      return GL_HALF_FLOAT;
    case DataType::FLOAT32:
      return GL_FLOAT;
    default:
      return 0;
  }
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
