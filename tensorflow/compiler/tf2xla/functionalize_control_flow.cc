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

#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"

#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/functionalize_cond.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/compiler/tf2xla/functionalize_while.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

Status FunctionalizeControlFlow(const FunctionLibraryDefinition* lookup_library,
                                Graph* graph,
                                FunctionLibraryDefinition* library) {
  VLOG(2) << "FunctionalizeControlFlow (initial): "
          << dump_graph::DumpGraphToFile("functionalize_initial", *graph,
                                         library);

  // Functionalize and remove while loops from graph.
  TF_RETURN_IF_ERROR(FunctionalizeWhileLoop(lookup_library, graph, library));

  // FunctionalizeControlFlow is invoked for every function, so the loops's
  // bodies and conditionals that were extracted into functions will be handled
  // in successive invocations.
  TF_RETURN_IF_ERROR(FunctionalizeCond(graph, library));

  VLOG(2) << "FunctionalizeControlFlow (final): "
          << dump_graph::DumpGraphToFile("functionalize_final", *graph,
                                         library);

  return Status::OK();
}

// Transformation that converts TensorFlow's graph control flow constructs into
// functional equivalents.
Status FunctionalizeControlFlow(Graph* graph,
                                FunctionLibraryDefinition* library) {
  return FunctionalizeControlFlow(/*lookup_library=*/nullptr, graph, library);
}

Status FunctionalizeControlFlowForFunction(
    const string& func_name, const string& new_func_name,
    const protobuf::Map<string, tensorflow::AttrValue>& attrs,
    FunctionLibraryDefinition* fld, FunctionLibraryRuntime* flr,
    std::map<string, string>* canonicalized_name_to_new_name) {
  // Convert the function to Graph.
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(flr->Instantiate(func_name, AttrSlice(&attrs), &handle));
  Status ret_status = Status::OK();
  auto cleanup_handle = gtl::MakeCleanup([&]() {
    auto s = flr->ReleaseHandle(handle);
    if (!s.ok()) {
      ret_status.Update(s);
    }
  });
  const FunctionBody* body = flr->GetFunctionBody(handle);

  // Call graph optimizer. The most important optimization we need is constant
  // folding, which will replace ops like Shape/BroadcastGradientArgs with
  // constant shape input. Without this optimization, those ops might become
  // dynamic input for then/else body function and XLA will complain that input
  // is not compile time constant. We enable function inlining as well, because
  // otherwise we won't be able to infer shape for any node depending on
  // function call nodes.
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile(
        absl::StrCat("functionalize_control_flow_before_opt_", func_name),
        *body->graph, fld);
  }
  // Optimizer accepts std::unique_ptr<Graph>* as input and might change
  // underlying pointer, thus we create a new Graph and copy from body->graph.
  std::unique_ptr<Graph> optimized_graph(new Graph(fld));
  CopyGraph(*body->graph, optimized_graph.get());
  OptimizerOptions opts;
  opts.set_opt_level(OptimizerOptions::L0);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  auto cf_consider_fn = [](const Node* n) {
    // Skip SymbolicGradient op when doing constant folding.
    // Enabling SymbolicGradient op in constant folding requires
    // flr->device() to be non-null, and here we have not constructed
    // proper Device object yet (it will be constructed in XlaCompiler).
    return n->type_string() != FunctionLibraryDefinition::kGradientOp;
  };
  optimizer.Optimize(flr, flr->env(),
                     /*device=*/nullptr, &optimized_graph,
                     /*shape_map=*/nullptr, /*cse_consider_fn=*/nullptr,
                     cf_consider_fn);
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile(
        absl::StrCat("functionalize_control_flow_after_opt_", func_name),
        *optimized_graph, fld);
  }

  // If any node has associated functions, functionalize them first.
  // Gather nodes with associated functions first, because rewriting those nodes
  // might involve node deletion/addition. Avoid modifying nodes while iterating
  // it.
  std::vector<std::pair<Node*, std::vector<AssociatedFunctionInfo>>>
      nodes_to_associated_functions;
  for (auto* n : optimized_graph->nodes()) {
    auto associated_functions = GetAssociatedFunctions(*n, flr);
    if (!associated_functions.empty()) {
      nodes_to_associated_functions.push_back({n, associated_functions});
    }
  }
  for (auto iter : nodes_to_associated_functions) {
    Node* n = iter.first;
    auto associated_functions = iter.second;
    for (auto& associated_function : associated_functions) {
      string name = associated_function.func_name();
      string canonicalized_name =
          Canonicalize(name, AttrSlice(&associated_function.attrs()));
      auto iter = canonicalized_name_to_new_name->find(canonicalized_name);
      string new_name;
      if (iter != canonicalized_name_to_new_name->end()) {
        // If we already functionalized this function, skip functionalization
        // but still rewrite the node.
        new_name = iter->second;
      } else {
        if (associated_function.type() ==
            AssociatedFunctionInfo::AssociatedFunctionType::kSymbolicGradient) {
          // For SymbolicGradient, `name` is always "SymbolicGradient",
          // which is not very informative. Use node name instead.
          new_name = fld->UniqueFunctionName(absl::StrCat(n->name(), "_f15n_"));
        } else {
          new_name = fld->UniqueFunctionName(absl::StrCat(name, "_f15n_"));
        }
        TF_RETURN_IF_ERROR(FunctionalizeControlFlowForFunction(
            name, new_name, associated_function.attrs(), fld, flr,
            canonicalized_name_to_new_name));
        (*canonicalized_name_to_new_name)[canonicalized_name] = new_name;
      }
      // Notice that if "n" is a function call, RewriteAssociatedFunction() will
      // delete it and create a new node instead, making "n" an invalid pointer.
      // That's fine because in that case, associated_functions will only have
      // one member and the loop will only run once.
      TF_RETURN_IF_ERROR(RewriteAssociatedFunction(
          optimized_graph.get(), n, fld, associated_function, new_name));
    }
  }

  // Functionalize the function body.
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile(
        absl::StrCat("functionalize_control_flow_before_fdef_", func_name),
        *optimized_graph, fld);
  }
  TF_RETURN_IF_ERROR(FunctionalizeControlFlow(optimized_graph.get(), fld));
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile(
        absl::StrCat("functionalize_control_flow_after_fdef_", func_name),
        *optimized_graph, fld);
  }
  FunctionDef functionalized_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*optimized_graph, new_func_name,
                                        &functionalized_fdef));

  // Add rewritten FunctionDef into library.
  if (func_name == new_func_name) {
    VLOG(2) << "Replacing function " << func_name;
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(new_func_name, functionalized_fdef));
  } else {
    VLOG(2) << "Adding function " << new_func_name;
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(functionalized_fdef));
  }

  return ret_status;
}

Status FunctionalizeControlFlowPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile("functionalize_control_flow_before", *graph,
                                options.flib_def);
  }
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          /*device_mgr=*/nullptr, options.session_options->env,
          TF_GRAPH_DEF_VERSION, options.flib_def, OptimizerOptions()));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  // Find XLA compile ops and its corresponding FunctionDef.
  static std::map<string, string>* kNodeTypeToFunctionAttrMapping =
      new std::map<string, string>{
          {"TPUCompile", "function"},
          {"XlaLaunch", "function"},
      };
  std::map<string, string> canonicalized_name_to_new_name;
  for (Node* n : graph->nodes()) {
    auto it = kNodeTypeToFunctionAttrMapping->find(n->type_string());
    if (it == kNodeTypeToFunctionAttrMapping->end()) {
      continue;
    }
    const string func_attr = it->second;
    if (kNodeTypeToFunctionAttrMapping->find(n->type_string()) !=
        kNodeTypeToFunctionAttrMapping->end()) {
      NameAttrList func;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), func_attr, &func));
      VLOG(2) << "Graph has node " << n->type_string()
              << ". Corresponding function: " << func.name();
      string new_func_name = options.flib_def->UniqueFunctionName(
          absl::StrCat(func.name(), "_f15n_"));
      TF_RETURN_IF_ERROR(FunctionalizeControlFlowForFunction(
          func.name(), new_func_name, func.attr(), options.flib_def, flr,
          &canonicalized_name_to_new_name));
      n->ClearAttr(func_attr);
      func.set_name(new_func_name);
      n->AddAttr(func_attr, func);
    }
  }

  if (VLOG_IS_ON(4)) {
    dump_graph::DumpGraphToFile("functionalize_control_flow_after", *graph,
                                options.flib_def);
  }
  return Status::OK();
}

}  // namespace tensorflow
