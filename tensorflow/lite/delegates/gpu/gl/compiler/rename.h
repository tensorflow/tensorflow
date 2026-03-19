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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_RENAME_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_RENAME_H_

#include <functional>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

// Functor takes old name and returns new name.
using NameFunctor = std::function<std::string(absl::string_view name)>;

// Rewrites source code, objects and parameters with the new names supplied
// by the given functor.
absl::Status Rename(const NameFunctor& name_func, GeneratedCode* code);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_RENAME_H_
