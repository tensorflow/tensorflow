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

#include "tensorflow/core/graph/while_context.h"

namespace tensorflow {

WhileContext::WhileContext(StringPiece frame_name,
                           std::vector<Node*> enter_nodes,
                           std::vector<Node*> exit_nodes,
                           OutputTensor cond_output,
                           std::vector<OutputTensor> body_inputs,
                           std::vector<OutputTensor> body_outputs)
    : frame_name_(frame_name),
      enter_nodes_(std::move(enter_nodes)),
      exit_nodes_(std::move(exit_nodes)),
      cond_output_(cond_output),
      body_inputs_(std::move(body_inputs)),
      body_outputs_(std::move(body_outputs)) {
  const size_t num_loop_vars = enter_nodes_.size();
  DCHECK_EQ(exit_nodes_.size(), num_loop_vars);
  DCHECK_EQ(body_inputs_.size(), num_loop_vars);
  DCHECK_EQ(body_outputs_.size(), num_loop_vars);
}

}  // namespace tensorflow
