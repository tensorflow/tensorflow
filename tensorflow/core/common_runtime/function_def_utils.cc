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

#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/hash.h"

namespace tensorflow {

absl::Status FunctionDefToBodyHelper(
    core::RefCountPtr<FunctionRecord>&& record, const AttrSlice& attrs,
    const FunctionLibraryDefinition* const lib_def,
    const std::function<absl::Status(const string&, const OpDef**)>&
        get_func_sig,
    std::unique_ptr<FunctionBody>* fbody) {
  // Instantiates the function template into a graph def.
  InstantiationResult result;
  TF_RETURN_IF_ERROR(
      InstantiateFunction(record->fdef(), attrs, get_func_sig, &result));

  auto graph = std::make_unique<Graph>(lib_def);

  auto construction_context_iter =
      record->fdef().attr().find("_construction_context");
  if (construction_context_iter != record->fdef().attr().end()) {
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

  TF_RETURN_IF_ERROR(ConvertNodeDefsToGraph(opts, result.nodes, graph.get(),
                                            /*debug_info=*/nullptr));

  const StackTracesMap* stack_traces =
      lib_def->GetStackTraces(record->fdef().signature().name());
  if (stack_traces) {
    for (Node* n : graph->nodes()) {
      if (n) {
        auto it = stack_traces->find(n->name());
        if (it != stack_traces->end()) {
          n->SetStackTrace(it->second);
        }
      }
    }
  }

  // Call BuildControlFlowInfo to validate that this function body has
  // well-formed control flow.
  std::vector<ControlFlowInfo> dummy;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph.get(), &dummy));

  *fbody = std::make_unique<FunctionBody>(std::move(record), result.arg_types,
                                          result.ret_types, graph.release());
  return absl::OkStatus();
}

absl::Status FunctionDefToBodyHelper(core::RefCountPtr<FunctionRecord>&& record,
                                     const AttrSlice& attrs,
                                     const FunctionLibraryDefinition* lib_def,
                                     std::unique_ptr<FunctionBody>* fbody) {
  const auto get_func_sig = [&lib_def](const string& op, const OpDef** sig) {
    return lib_def->LookUpOpDef(op, sig);
  };
  return FunctionDefToBodyHelper(std::move(record), attrs, lib_def,
                                 get_func_sig, fbody);
}

absl::Status FunctionDefToBodyHelper(const FunctionDef& fdef,
                                     const AttrSlice& attrs,
                                     const FunctionLibraryDefinition* lib_def,
                                     std::unique_ptr<FunctionBody>* fbody) {
  core::RefCountPtr<FunctionRecord> record(
      new FunctionRecord(FunctionDef(fdef), {}, true));
  const auto get_func_sig = [&lib_def](const string& op, const OpDef** sig) {
    return lib_def->LookUpOpDef(op, sig);
  };
  return FunctionDefToBodyHelper(std::move(record), attrs, lib_def,
                                 get_func_sig, fbody);
}

namespace {
bool PrunableStatefulNode(const Node* n) {
  // This set contains ops that are marked as "stateful" in their op
  // registration, but can be pruned from a function graph if nothing depends
  // on them. Typically, these are operations that are "impure" but have no
  // side effects. For example, "ResourceGather" reads from a resource variable
  // and can produce different results on each invocation (due to variable
  // updates) but it does not itself modify the variable.
  // TODO(b/341721055): Consolidate this set with other side effect modeling.
  static const absl::flat_hash_set<string>* prunable_stateful_ops =
      new absl::flat_hash_set<string>{
          FunctionLibraryDefinition::kArgOp,
          "ResourceGather",
          "ResourceGatherNd",
      };
  return prunable_stateful_ops->contains(n->type_string());
}
}  // namespace

// TODO(ezhulenev, skyewm): Function body should not have special treatment of
// stateful ops, graph should encode nodes that must execute with `control_ret`
// and `control_output`.
void PruneFunctionBody(const FunctionDef& fdef, Graph* g,
                       absl::Span<Node*> additional_root_nodes) {
  VLOG(2) << "Pruning function body: function_name=" << fdef.signature().name()
          << " #nodes = " << g->num_nodes();

  // `control_ret` nodes must be always executed.
  absl::flat_hash_set<absl::string_view, tsl::StringPieceHasher>
      control_ret_nodes;
  for (const auto& control_ret : fdef.control_ret()) {
    control_ret_nodes.insert(control_ret.second);
  }

  std::unordered_set<const Node*> nodes;
  for (auto n : additional_root_nodes) {
    nodes.insert(n);
  }
  for (auto n : g->nodes()) {
    // NOTE(mrry): "_Retval" nodes are stateful, and so will be added
    // to the seed set of `nodes`. "_Arg" nodes are also stateful, but we
    // specifically exclude them as seeds, to avoid unconditionally executing
    // unused argument nodes (e.g. in a function like `lambda x, y: y`).
    // TODO(mrry): Investigate whether the `n->IsControlFlow()` test is
    // still needed. It would be preferable to prune entire loops and/or
    // conditionals if they are not used in the graph.
    if (n->IsControlFlow() ||
        (n->op_def().is_stateful() && !PrunableStatefulNode(n)) ||
        (control_ret_nodes.find(n->name()) != control_ret_nodes.end())) {
      nodes.insert(n);
    }
  }
  bool changed = PruneForReverseReachability(g, std::move(nodes));
  if (changed) {
    VLOG(2) << "Pruned function body and changed: function_name="
            << fdef.signature().name() << " #nodes = " << g->num_nodes();
    FixupSourceAndSinkEdges(g);
  }
}

}  // end namespace tensorflow
