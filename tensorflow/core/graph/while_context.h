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

#ifndef TENSORFLOW_CORE_GRAPH_WHILE_CONTEXT_H_
#define TENSORFLOW_CORE_GRAPH_WHILE_CONTEXT_H_

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Information about a while loop. Every user-defined while loop has an
// associated WhileContext, i.e., there is a WhileContext for every execution
// frame. Created with the while loop and used during gradient
// construction. Note that the gradient graph of while loop contains while loops
// itself, but these do not generate separate WhileContexts.
//
// TODO(skyewm): this is currently insufficient to handle nested loops and
// conditionals (and possibly other requirements). This may change a lot in the
// future to support these features.
//
// TODO(skyewm): de/serialize in MetaGraphDef so imported while loops will be
// differentiable. Figure out backwards compatibility story.
class WhileContext {
 public:
  WhileContext(absl::string_view frame_name, std::vector<Node*> enter_nodes,
               std::vector<Node*> exit_nodes, OutputTensor cond_output,
               std::vector<OutputTensor> body_inputs,
               std::vector<OutputTensor> body_outputs);

  const string& frame_name() const { return frame_name_; }
  const std::vector<Node*>& enter_nodes() const { return enter_nodes_; }
  const std::vector<Node*>& exit_nodes() const { return exit_nodes_; }
  const OutputTensor& cond_output() const { return cond_output_; }
  const std::vector<OutputTensor>& body_inputs() const { return body_inputs_; }
  const std::vector<OutputTensor>& body_outputs() const {
    return body_outputs_;
  }

 private:
  // Each user-defined while loop defines a new execution frame, which is
  // uniquely identified by its frame name. Frames are used by the executor to
  // manage the iterations of a loop. See the FrameState comment in
  // core/common_runtime/executor.cc for more details.
  const string frame_name_;

  // The enter nodes defining the input loop variables to the while loop. This
  // vector defines the order of the loop variables.
  const std::vector<Node*> enter_nodes_;

  // The exit nodes defining the outputs of the while loop. These are in loop
  // variable order.
  const std::vector<Node*> exit_nodes_;

  // The boolean output of the loop predicate.
  const OutputTensor cond_output_;

  // The inputs and outputs to the loop body.
  const std::vector<OutputTensor> body_inputs_;
  const std::vector<OutputTensor> body_outputs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_WHILE_CONTEXT_H_
