/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/culprit_finder/model_metadata_lib.h"

#include <vector>

namespace tflite {

namespace tooling {

std::vector<int> ModelMetadata::GetOutputTensorsOfNode(int node_id) {
  std::vector<int> output_tensors;
  output_tensors.reserve(node_index_to_node_proto_[node_id].outputs_size());

  for (int output_tensor_id : node_index_to_node_proto_[node_id].outputs()) {
    output_tensors.push_back(output_tensor_id);
  }
  return output_tensors;
}
}  // namespace tooling
}  // namespace tflite
