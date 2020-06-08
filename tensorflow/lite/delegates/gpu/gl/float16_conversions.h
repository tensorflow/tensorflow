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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_FLOAT16_CONVERSIONS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_FLOAT16_CONVERSIONS_H_

#include "tensorflow/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// If an object is float32, converts it to float16 representation.
bool MaybeConvertToFloat16(Object* object);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_FLOAT16_CONVERSIONS_H_
