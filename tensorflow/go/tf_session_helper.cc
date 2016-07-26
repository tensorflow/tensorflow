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
  for(auto i = 0; i < input_tensor_names.size(); ++i)
    cstring_input_tensor_names.push_back(const_cast<char*>(input_tensor_names[i].c_str()));
  for(auto i = 0; i < output_tensor_names.size(); ++i)
    cstring_output_tensor_names.push_back(const_cast<char*>(output_tensor_names[i].c_str()));
  for(auto i = 0; i < target_node_names.size(); ++i)
    cstring_target_node_names.push_back(const_cast<char*>(target_node_names[i].c_str()));

  
  outputs = std::vector<TF_Tensor*>(output_tensor_names.size());

  TF_Run(session, cstring_input_tensor_names.data(), inputs.data(), input_tensor_names.size(),
         const_cast<const char**>(cstring_output_tensor_names.data()), outputs.data(),
         output_tensor_names.size(), cstring_target_node_names.data(),
         target_node_names.size(), out_status);
}

}  // namespace tensorflow
