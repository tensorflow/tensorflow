/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/grappler/utils/functions.h"

#include <unordered_map>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

std::unique_ptr<GrapplerItem> GrapplerItemFromFunctionDef(
    const FunctionDef& func,
    const std::unordered_map<string, AttrValue>& func_attr,
    const FunctionDefLibrary& library) {
  if (func.signature().name().empty()) {
    LOG(ERROR) << "function name must be specified.";
    return nullptr;
  }
  std::unique_ptr<GrapplerItem> new_item(new GrapplerItem());
  new_item->id = func.signature().name();

  std::unordered_map<string, string> port_map;

  // Add the function inputs as placeholder
  for (const auto& inp : func.signature().input_arg()) {
    NodeDef* ph = new_item->graph.add_node();
    ph->set_name(inp.name());
    ph->set_op("Placeholder");
    if (inp.type() != DT_INVALID) {
      (*ph->mutable_attr())["T"].set_type(inp.type());
    } else {
      auto it = func_attr.find(inp.type_attr());
      if (it == func_attr.end()) {
        LOG(ERROR) << "Unknown type attribute " << inp.type_attr()
                   << " for function input " << inp.name();
        return nullptr;
      } else {
        (*ph->mutable_attr())["T"] = it->second;
      }
    }
    port_map[inp.name()] = inp.name();
  }

  // Add the function body to the graph.
  FunctionLibraryDefinition func_def(OpRegistry::Global(), library);

  for (const NodeDef& node : func.node_def()) {
    NodeDef* new_node = new_item->graph.add_node();
    *new_node = node;
    // Replace the placeholder attribute values with the specified value.
    for (auto& attr : *new_node->mutable_attr()) {
      const string& ph_name = attr.second.placeholder();
      auto it = func_attr.find(ph_name);
      if (it != func_attr.end()) {
        attr.second = it->second;
      }
    }

    // Functions use a custom format to encode connectivity. Map these custom
    // strings to regular ones.
    const OpRegistrationData* registration;
    Status status = func_def.LookUp(node.op(), &registration);
    if (!status.ok()) {
      LOG(ERROR) << "Op " << node.op() << " not registered: " << status;
      return nullptr;
    }

    tensorflow::NameRangeMap inputs;
    tensorflow::NameRangeMap outputs;
    status = tensorflow::NameRangesForNode(node, registration->op_def, &inputs,
                                           &outputs);
    if (!status.ok()) {
      LOG(ERROR) << "Op " << node.op() << " invalid: " << status;
      return nullptr;
    }
    for (const auto& name_range : outputs) {
      string port_prefix =
          strings::StrCat(node.name(), ":", name_range.first, ":");
      int index_start = name_range.second.first;
      int index_end = name_range.second.second;
      for (int i = index_start; i < index_end; ++i) {
        string port_id = strings::StrCat(port_prefix, i - index_start);
        string port_name = strings::StrCat(node.name(), ":", i);
        port_map[port_id] = port_name;
      }
    }
  }

  for (auto& node : *new_item->graph.mutable_node()) {
    // Rewrite the inputs to use the normal naming convention.
    for (int i = 0; i < node.input_size(); ++i) {
      const string& input = node.input(i);
      if (IsControlInput(input)) {
        // No need to remap control dependencies.
        continue;
      } else {
        auto it = port_map.find(input);
        if (it == port_map.end()) {
          LOG(ERROR) << "Unknown input: " << input;
          return nullptr;
        }
        node.set_input(i, it->second);
      }
    }
  }

  // Add the function outputs to the list of fetch nodes, taking into account
  // the output mapping if any.
  for (const auto& out : func.signature().output_arg()) {
    auto it = func.ret().find(out.name());
    if (it != func.ret().end()) {
      auto it2 = port_map.find(it->second);
      if (it2 == port_map.end()) {
        LOG(ERROR) << "Unknown output mapping: " << it->first << " to "
                   << it->second;
        return nullptr;
      } else {
        new_item->fetch.emplace_back(it2->second);
      }
    } else {
      new_item->fetch.emplace_back(out.name());
    }
  }
  // Add the function inputs to the list of feeds.
  for (const auto& inp : func.signature().input_arg()) {
    new_item->feed.emplace_back(inp.name(), Tensor());
  }

  return new_item;
}

}  // end namespace grappler
}  // end namespace tensorflow
