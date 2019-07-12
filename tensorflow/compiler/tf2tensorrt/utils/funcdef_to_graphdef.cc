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
//#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/ascii.h"

namespace tensorflow {
namespace tensorrt {

const char* const kInputPHName = "TensorRTInputPH_";
const char* const kOutputPHName = "TensorRTOutputPH_";
const char* const kInputPHNameLower = "tensorrtinputph_";
const char* const kOutputPHNameLower = "tensorrtoutputph_";

string NewNameWithIOPrefix(const Node* n) {
  if (absl::StartsWith(n->name(), kInputPHNameLower)){
    return strings::StrCat(kInputPHName, n->id());
  }
  else if (absl::StartsWith(n->name(), kOutputPHNameLower)) {
    return strings::StrCat(kOutputPHName, n->id());
  }
  return strings::StrCat("n", n->id());
}

void ToGraphDefWithIOPrefix(const Graph* g, GraphDef* gdef) {
  // This is the same function as in function.cc. However, it uses the
  // NewName mapping above, which retains IO prefixes (kInputPHName etc)
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
    ndef->set_name(NewNameWithIOPrefix(n));
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
      const string srcname = NewNameWithIOPrefix(e->src());
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
  const FunctionLibraryDefinition* flib_def = flib_runtime->GetFunctionLibraryDefinition();
  const FunctionBody* fbody;
  VLOG(0) << "Getting Function Body \n";
  VLOG(0) << "HANDLE" << handle;
  fbody = flib_runtime->GetFunctionBody(handle);
  //TF_RET_CHECK(*fbody)
  std::unique_ptr<Graph> graph(new Graph(flib_def));
    
  CopyGraph(*fbody->graph, graph.get());

  // Copied from compiler/xla/compile_xla.cc : 
  /*
  OptimizerOptions opts;
  opts.set_opt_level(OptimizerOptions::L0);
  opts.set_do_common_subexpression_elimination(false);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  auto cf_consider_fn = [](const Node* n) {
    for (const auto& output_arg : n->op_def().output_arg()) {
      if (output_arg.type() == DT_VARIANT) {
        return false;
      }
    }
    return true;
  };
  GraphOptimizer::Options graph_optimizer_options;
  graph_optimizer_options.cf_consider_fn = cf_consider_fn;
  
  */
  //optimizer.Optimize(flib_runtime, flib_runtime->env(),
  //                   /*device=*/nullptr, &graph, graph_optimizer_options);
   
  for (Node* n : graph->nodes()) {
    auto id = n->id();
    if (n->IsArg()) {
      VLOG(1) << "Arg Node id " << id;
      input_node_ids->push_back(id);
    }
    if (n->IsRetval()) {
      VLOG(1) << "Retval Node id " << id;
      output_node_ids->push_back(id);
    }
  }
  
  ToGraphDefWithIOPrefix(graph.release(), graph_def);

  for (const auto node_def : graph_def->node()) {
    string node_name = node_def.name();
    VLOG(0) << "NODENAME AFTER FROM FUNCDEF " << node_name << ", op=" << node_def.op();
  }
  VLOG(0) << "Finished converting \n";

  return Status::OK();

}

}
}
