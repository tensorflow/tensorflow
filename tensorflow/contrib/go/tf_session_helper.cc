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

// We define the PY_ARRAY_UNIQUE_SYMBOL in this .cc file and provide an
// ImportNumpy function to populate it.
#define TF_IMPORT_NUMPY

#include "tf_session_helper.h"

namespace tensorflow {

void TF_Run_wrapper(TF_Session* session,
                   // Input tensors
                   std::vector<std::string> input_tensor_names, std::vector<TF_Tensor*> inputs,
                   // Output tensors
                   std::vector<std::string> output_tensor_names, std::vector<TF_Tensor*> &outputs,
                   // Target nodes
                   std::vector<std::string> target_node_names,
                   // Output status
                   TF_Status* out_status) {
 
  std::vector<const char*> cstring_input_tensor_names;
  std::vector<const char*> cstring_output_tensor_names;
  std::vector<const char*> cstring_target_node_names;
  for (auto& input_name: input_tensor_names) {
    cstring_input_tensor_names.push_back(input_name.c_str());
  }
  for (auto& out_name: output_tensor_names) {
    cstring_output_tensor_names.push_back(out_name.c_str());
  }
  for (auto& target_name: target_node_names) {
    cstring_target_node_names.push_back(target_name.c_str());
  }
  
  outputs = std::vector<TF_Tensor*>(output_tensor_names.size());

  const TF_Buffer* run_options = TF_NewBuffer();
  TF_Buffer* run_outputs = TF_NewBuffer();

  // TODO: Add run_options
  TF_Run(session, run_options, cstring_input_tensor_names.data(), inputs.data(), input_tensor_names.size(),
         const_cast<const char**>(cstring_output_tensor_names.data()), outputs.data(),
         output_tensor_names.size(), cstring_target_node_names.data(),
         target_node_names.size(), run_outputs, out_status);

}

}  // namespace tensorflow
