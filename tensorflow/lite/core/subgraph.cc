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

#include "tensorflow/lite/core/subgraph.h"

namespace tflite {

Subgraph::~Subgraph() {
  for (auto& node_and_reg : nodes_and_registration_) {
    TfLiteNode& node = node_and_reg.first;
    TfLiteIntArrayFree(node.inputs);
    TfLiteIntArrayFree(node.outputs);
    TfLiteIntArrayFree(node.temporaries);
    if (node.builtin_data) free(node.builtin_data);
    OpFree(node_and_reg.second, node.user_data);
    node.builtin_data = nullptr;
  }
}

}  // namespace tflite
