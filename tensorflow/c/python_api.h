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

#ifndef THIRD_PARTY_TENSORFLOW_C_PYTHON_API_H_
#define THIRD_PARTY_TENSORFLOW_C_PYTHON_API_H_

#include "tensorflow/c/c_api.h"

// These functions can be removed without notice. They exist to facilitate some
// refactoring of graph construction code in the Python API.

namespace tensorflow {

void AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input);

// Changes an attr value in the node_def Protocol Buffer and sets a status upon
// completion.
void SetAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
             TF_Buffer* attr_value_proto, TF_Status* status);

void SetRequestedDevice(TF_Graph* graph, TF_Operation* op, const char* device);

void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst,
                TF_Status* status);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_C_PYTHON_API_H_
