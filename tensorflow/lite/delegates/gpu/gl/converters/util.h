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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_CONVERTERS_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_CONVERTERS_UTIL_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {

inline std::string GetShaderHeader(const uint3& localsize) {
  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

inline uint32_t BytesForPHWC4(const BHWC& shape) {
  return shape.b * shape.h * shape.w * AlignByN(shape.c, 4) * sizeof(float);
}

inline uint32_t BytesForBHWC(const BHWC& shape) {
  return shape.DimensionsProduct() * sizeof(float);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_CONVERTERS_UTIL_H_
