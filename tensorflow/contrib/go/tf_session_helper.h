/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_GO_CLIENT_TF_SESSION_HELPER_H_
#define TENSORFLOW_GO_CLIENT_TF_SESSION_HELPER_H_

#include <string>
#include <vector>

#include "tensorflow/core/public/tensor_c_api.h"

namespace tensorflow {

void TF_Run_wrapper(TF_Session* session,
                   // Input tensors
                   std::vector<std::string> input_tensor_names, std::vector<TF_Tensor*> inputs,
                   // Output tensors
                   std::vector<std::string> output_tensor_names, std::vector<TF_Tensor*> &outputs,
                   // Target nodes
                   std::vector<std::string> target_node_names,
                   // Output status
                   TF_Status* out_status);

}  // namespace tensorflow

#endif  // TENSORFLOW_GO_CLIENT_TF_SESSION_HELPER_H_
