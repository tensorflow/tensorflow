/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status RenameAttribute(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  if (!context.params.count("old_attribute_name") ||
      (context.params.at("old_attribute_name").size() != 1) ||
      !context.params.count("new_attribute_name") ||
      (context.params.at("new_attribute_name").size() != 1)) {
    return errors::InvalidArgument(
        "remove_nodes expects exactly one 'old_attribute_name' and one "
        "'new_attribute_name' argument, e.g. "
        "remove_attribute(old_attribute_name=foo, new_attribute_name=bar)");
  }

  string op_name;
  if (context.params.count("op_name")) {
    op_name = context.params.at("op_name")[0];
  } else {
    op_name = "*";
  }

  const string old_attribute_name = context.params.at("old_attribute_name")[0];
  const string new_attribute_name = context.params.at("new_attribute_name")[0];
  output_graph_def->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    new_node->CopyFrom(node);
    if (((op_name == "*") || (op_name == node.op())) &&
        (node.attr().count(old_attribute_name))) {
      AttrValue attribute_value = node.attr().at(old_attribute_name);
      new_node->mutable_attr()->erase(old_attribute_name);
      new_node->mutable_attr()->insert({new_attribute_name, attribute_value});
    }
  }

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("rename_attribute", RenameAttribute);

}  // namespace graph_transforms
}  // namespace tensorflow
