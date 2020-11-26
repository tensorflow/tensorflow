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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_VARIABLE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_VARIABLE_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {

struct Variable {
  using ValueType =
      absl::variant<int32_t, int2, int4, uint32_t, uint4, float, float2, float4,
                    std::vector<int2>, std::vector<float4>>;

  std::string name;
  ValueType value;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_VARIABLE_H_
