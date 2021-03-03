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

#include "tensorflow/lite/delegates/gpu/gl/kernels/custom_registry.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace tflite {
namespace gpu {
namespace gl {

void RegisterCustomOps(
    absl::flat_hash_map<std::string, std::vector<std::unique_ptr<NodeShader>>>*
        shaders) {}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
