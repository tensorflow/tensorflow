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

#include "tensorflow/compiler/tf2tensorrt/utils/funcdef_to_graphdef.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/logging.h"

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace tensorrt {

string AppendIdToNodeName(const Node* n) {
  if (absl::StartsWith(n->name(), IONamePrefixes::kInputPHNameLower)) {
    return strings::StrCat(IONamePrefixes::kInputPHName, n->id());
  } else if (absl::StartsWith(n->name(), IONamePrefixes::kOutputPHNameLower)) {
    return strings::StrCat(IONamePrefixes::kOutputPHName, n->id());
  }
  return strings::StrCat("n", n->id());
}

void ToGraphDefWithIOPrefix(const Graph* g, GraphDef* gdef) {
  // This is the same function as in function.cc. However, it uses the
  // name mapping above, which retains IO prefixes (IONamePrefixes::kInputPHName etc)
  gtl::InlinedVector<const Edge*, 4> inputs;
  gdef->Clear();
  *gdef->mutable_versions() = g->versions();

  std::vector<Node*> start_nodes;
  for (Node* n : g->nodes()) {
    if (n->out_edges().empty()) {
      start_nodes.push_back(n);
    }
  }

  ReverseDFSFrom(*g, start_nodes, nullptr, [gdef, &inputs](Node* n) {
    if (!n->IsOp()) return;
    NodeDef* ndef = gdef->add_node();
    ndef->set_name(AppendIdToNodeName(n));
    ndef->set_op(n->type_string());
    for (const auto& attr : n->attrs()) {
      (*ndef->mutable_attr())[attr.first] = attr.second;
    }

    if (!n->assigned_device_name().empty()) {
      ndef->set_device(n->assigned_device_name());
    } else {
      ndef->set_device(n->requested_device());
    }

    inputs.clear();
    inputs.resize(n->num_inputs());
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        inputs.push_back(e);
      } else {
        if (inputs[e->dst_input()] == nullptr) {
          inputs[e->dst_input()] = e;
        } else {
          LOG(WARNING) << "Malformed graph node. multiple input edges: "
                       << n->DebugString();
        }
      }
    }
    // node->name() is merely NodeDef::name, which are not guaranteed
    // to be unique and stable after optimization rewrites. Therefore,
    // we use "n<node id> or <io prefix><node_id>" instead.
    for (const Edge* e : inputs) {
      if (e == nullptr) {
        ndef->add_input("unknown");
        continue;
      }
      const string srcname = AppendIdToNodeName(e->src());
      if (!e->src()->IsOp()) {
      } else if (e->IsControlEdge()) {
        ndef->add_input(strings::StrCat("^", srcname));
      } else if (e->src_output() == 0) {
        ndef->add_input(srcname);
      } else {
        ndef->add_input(strings::StrCat(srcname, ":", e->src_output()));
      }
    }
  });
}

Status FunctionDefToGraphDef(FunctionLibraryRuntime::Handle handle,
                             FunctionLibraryRuntime* flib_runtime,
                             GraphDef* graph_def,
                             std::vector<int>* input_node_ids,
                             std::vector<int>* output_node_ids) {
  const FunctionLibraryDefinition* flib_def =
      flib_runtime->GetFunctionLibraryDefinition();
  const FunctionBody* fbody;
  fbody = flib_runtime->GetFunctionBody(handle);
  if (!fbody) {
    return errors::Internal(
        "Function body is null when converting from FuncDef to GraphDef.");
  }
  std::unique_ptr<Graph> graph(new Graph(flib_def));

  CopyGraph(*fbody->graph, graph.get());

  for (Node* n : graph->nodes()) {
    auto id = n->id();
    if (n->IsArg()) {
      VLOG(2) << "Arg Node id used for unique naming is " << id;
      input_node_ids->push_back(id);
    }
    if (n->IsRetval()) {
      VLOG(2) << "Retval Node id used for unique naming is " << id;
      output_node_ids->push_back(id);
    }
  }

  ToGraphDefWithIOPrefix(graph.release(), graph_def);

  if VLOG_IS_ON(2) {
    for (const auto node_def : graph_def->node()) {
      VLOG(2) << "Node name after FunctionDefToGraphDef: " << node_def.name();
    }
  }

  return Status::OK();
}
}
}
