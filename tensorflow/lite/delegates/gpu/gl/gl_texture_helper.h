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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_HELPER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_HELPER_H_

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// From https://www.khronos.org/opengl/wiki/Normalized_Integer
// A Normalized Integer is an integer which is used to store a decimal floating
// point number. When formats use such an integer, OpenGL will automatically
// convert them to/from floating point values as needed. This allows normalized
// integers to be treated equivalently with floating-point values, acting as a
// form of compression.
GLenum ToTextureFormat(DataType type, bool normalized = false);

GLenum ToTextureInternalFormat(DataType type, bool normalized = false);

GLenum ToTextureDataType(DataType type);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_TEXTURE_HELPER_H_
