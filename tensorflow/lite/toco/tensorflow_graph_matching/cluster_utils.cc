/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"

#include <cstdint>

namespace toco {

void Transpose2DTensor(const float* tensor, int64_t row, int64_t col,
                       float* transposed_tensor) {
  for (int64_t r = 0; r < row; ++r) {
    for (int64_t c = 0; c < col; ++c) {
      transposed_tensor[c * row + r] = tensor[r * col + c];
    }
  }
}

}  // end namespace toco
