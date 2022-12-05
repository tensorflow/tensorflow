/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/function_def_utils.h"

#include <vector>

#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

Status FunctionDefToBodyHelper(
    const FunctionDef& fdef, const AttrSlice& attrs,
    const FunctionLibraryDefinition* const lib_def,
    const std::function<Status(const string&, const OpDef**)>& get_func_sig,
    std::unique_ptr<FunctionBody>* fbody) {
  // Instantiates the function template into a graph def.
  InstantiationResult result;
  TF_RETURN_IF_ERROR(InstantiateFunction(fdef, attrs, get_func_sig, &result));

  auto graph = std::make_unique<Graph>(lib_def);

  auto construction_context_iter = fdef.attr().find("_construction_context");
  if (construction_context_iter != fdef.attr().end()) {
    if (construction_context_iter->second.s() == "kEagerRuntime") {
      graph->SetConstructionContext(ConstructionContext::kEagerRuntime);
    } else {
      DCHECK(false) << "Unknown _construction_context attribute: "
                    << construction_context_iter->second.s();
    }
  }

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;
  TF_RETURN_IF_ERROR(ConvertNodeDefsToGraph(opts, result.nodes, graph.get()));

  const StackTracesMap& stack_traces =
      lib_def->GetStackTraces(fdef.signature().name());
  for (Node* n : graph->nodes()) {
    if (n) {
      auto it = stack_traces.find(n->name());
      if (it != stack_traces.end()) {
        n->SetStackTrace(it->second);
      }
    }
  }

  // Call BuildControlFlowInfo to validate that this function body has
  // well-formed control flow.
  std::vector<ControlFlowInfo> dummy;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph.get(), &dummy));

  *fbody = std::make_unique<FunctionBody>(fdef, result.arg_types,
                                           result.ret_types, graph.release());
  return OkStatus();
}

Status FunctionDefToBodyHelper(const FunctionDef& fdef, const AttrSlice& attrs,
                               const FunctionLibraryDefinition* lib_def,
                               std::unique_ptr<FunctionBody>* fbody) {
  const auto get_func_sig = [&lib_def](const string& op, const OpDef** sig) {
    return lib_def->LookUpOpDef(op, sig);
  };
  return FunctionDefToBodyHelper(fdef, attrs, lib_def, get_func_sig, fbody);
}

}  // end namespace tensorflow
