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

#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

unsigned int GetOptimalSize(unsigned int grid_size) {
  if (grid_size % 8 == 0 || grid_size % 8 >= 4 || grid_size >= 16) {
    return 8;
  }
  if (grid_size % 4 == 0 || grid_size % 4 >= 2 || grid_size >= 8) {
    return 4;
  }
  if (grid_size % 2 == 0 || grid_size >= 4) {
    return 2;
  }
  return 1;
}

}  // namespace

uint3 GetWorkGroupSizeForGrid(const uint3& grid_size) {
  unsigned int x_size = GetOptimalSize(grid_size.x);
  unsigned int y_size = GetOptimalSize(grid_size.y);
  unsigned int z_size = std::max(1u, 32u / (x_size * y_size));
  return {x_size, y_size, z_size};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
