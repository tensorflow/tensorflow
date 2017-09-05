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

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Sets the device field of ops in the graph.
Status SetDevice(const GraphDef& input_graph_def,
                 const TransformFuncContext& context,
                 GraphDef* output_graph_def) {
  string new_device;
  TF_RETURN_IF_ERROR(context.GetOneStringParameter("device", "", &new_device));
  bool if_default;
  TF_RETURN_IF_ERROR(
      context.GetOneBoolParameter("if_default", false, &if_default));

  output_graph_def->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    if (!if_default || (node.device().empty())) {
      new_node->set_device(new_device);
    }
  }

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("set_device", SetDevice);

}  // namespace graph_transforms
}  // namespace tensorflow
