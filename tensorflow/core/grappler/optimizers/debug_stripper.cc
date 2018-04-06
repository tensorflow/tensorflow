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

#include "tensorflow/core/grappler/optimizers/debug_stripper.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

Status DebugStripper::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* output) {
  *output = item.graph;
  for (NodeDef& node : *output->mutable_node()) {
    if (IsAssert(node)) {
      // Convert this node into a no-op.
      node.set_op("NoOp");
      node.clear_attr();
      // Convert all its inputs into control dependency, which will then
      // be optimized away by dependency optimizer.
      for (string& inp : *node.mutable_input()) {
        if (!IsControlInput(inp)) {
          inp = AsControlDependency(inp);
        }
      }
    } else if (IsCheckNumerics(node)) {
      // Replace with Identity op which will be pruned later.
      node.set_op("Identity");
      node.mutable_attr()->erase("message");
    }
  }
  return Status::OK();
}

void DebugStripper::Feedback(Cluster* cluster, const GrapplerItem& item,
                             const GraphDef& optimize_output, double result) {
  // Takes no feedback.
}

}  // end namespace grappler
}  // end namespace tensorflow
