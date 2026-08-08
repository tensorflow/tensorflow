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
#ifndef TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_CLUSTER_UTILS_H_
#define TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_CLUSTER_UTILS_H_

#include <cstdint>
#include <string>

#include "absl/strings/match.h"

namespace toco {

using ::absl::StrContains;

// Transpose a 2D tensor of size row * col pointed by "tensor" and return the
// results in "transposed_tensor". "transposed_tensor" must be pre-allocated
// by the same size as "tensor".
void Transpose2DTensor(const float* tensor, int64_t row, int64_t col,
                       float* transposed_tensor);

}  // end namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_CLUSTER_UTILS_H_
