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

#include "tensorflow/c/c_api.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "absl/strings/match.h"
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"  // NOLINT

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eval_const_tensor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"

// The implementation below is at the top level instead of the
// brain namespace because we are defining 'extern "C"' functions.
using tensorflow::AllocationDescription;
using tensorflow::DataType;
using tensorflow::ExtendSessionGraphHelper;
using tensorflow::Graph;
using tensorflow::GraphDef;
using tensorflow::mutex_lock;
using tensorflow::NameRangeMap;
using tensorflow::NameRangesForNode;
using tensorflow::NewSession;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::OpRegistry;
using tensorflow::OutputTensor;
using tensorflow::PartialTensorShape;
using tensorflow::RunMetadata;
using tensorflow::RunOptions;
using tensorflow::Session;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::TensorId;
using tensorflow::TensorShape;
using tensorflow::TensorShapeProto;
using tensorflow::ToTensorId;
using tensorflow::VersionDef;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;
using tensorflow::gtl::ArraySlice;
using tensorflow::strings::StrCat;

namespace {
#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
std::vector<tensorflow::Output> OutputsFromTFOutputs(TF_Output* tf_outputs,
                                                     int n) {
  std::vector<tensorflow::Output> outputs(n);
  for (int i = 0; i < n; ++i) {
    outputs[i] =
        tensorflow::Output(&tf_outputs[i].oper->node, tf_outputs[i].index);
  }
  return outputs;
}

void TFOutputsFromOutputs(const std::vector<tensorflow::Output>& outputs,
                          TF_Output* tf_outputs) {
  for (int i = 0; i < outputs.size(); i++) {
    tf_outputs[i].oper = ToOperation(outputs[i].node());
    tf_outputs[i].index = outputs[i].index();
  }
}
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

}  // namespace

extern "C" {

// While loop functions -------------------------------------------------------

namespace {

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

// Creates a placeholder representing an input to the cond or body graph.
// TODO(skyewm): remove these from final graph
bool CreateInput(const TF_Output& parent_input, TF_Graph* g, const char* name,
                 TF_Output* input, TF_Status* status) {
  TF_OperationDescription* desc = TF_NewOperation(g, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", TF_OperationOutputType(parent_input));
  // TODO(skyewm): set placeholder shape
  TF_Operation* oper = TF_FinishOperation(desc, status);
  if (!status->status.ok()) return false;
  *input = {oper, 0};
  return true;
}

// Copies `src_graph` into `dst_graph`. Any node in `src_graph` with input
// `src_inputs[i]` will have that input replaced with `dst_inputs[i]`.  `prefix`
// will be prepended to copied node names. `control_deps` are nodes in
// `dst_graph` that the copied `src_graph` nodes will have control dependencies
// on. `return_nodes` are nodes in `src_graph`, and the new corresponding nodes
// in `dst_graph` will be returned. `return_nodes` must be non-null.
Status CopyGraph(Graph* src_graph, Graph* dst_graph,
                 tensorflow::ShapeRefiner* dst_refiner,
                 const TF_Output* src_inputs,
                 const std::vector<tensorflow::Output>& dst_inputs,
                 const string& prefix,
                 const std::vector<tensorflow::Operation>& control_deps,
                 const TF_Output* nodes_to_return, int nreturn_nodes,
                 std::vector<tensorflow::Output>* return_nodes) {
  DCHECK(return_nodes != nullptr);
  GraphDef gdef;
  src_graph->ToGraphDef(&gdef);

  tensorflow::ImportGraphDefOptions opts;
  opts.prefix = prefix;

  for (int i = 0; i < dst_inputs.size(); ++i) {
    opts.input_map[ToTensorId(src_inputs[i])] =
        TensorId(dst_inputs[i].node()->name(), dst_inputs[i].index());
  }
  opts.skip_mapped_nodes = true;

  for (const tensorflow::Operation& op : control_deps) {
    opts.control_dependencies.push_back(op.node()->name());
  }

  for (int i = 0; i < nreturn_nodes; ++i) {
    opts.return_tensors.push_back(ToTensorId(nodes_to_return[i]));
  }

  // TODO(skyewm): change to OutputTensor
  tensorflow::ImportGraphDefResults results;
  TF_RETURN_IF_ERROR(
      ImportGraphDef(opts, gdef, dst_graph, dst_refiner, &results));

  for (const auto& pair : results.return_tensors) {
    return_nodes->emplace_back(pair.first, pair.second);
  }
  return Status::OK();
}

bool ValidateConstWhileParams(const TF_WhileParams& params, TF_Status* s) {
  if (params.cond_graph == nullptr || params.body_graph == nullptr ||
      params.cond_graph->parent == nullptr ||
      params.cond_graph->parent != params.body_graph->parent ||
      params.cond_graph->parent_inputs != params.body_graph->parent_inputs ||
      params.ninputs <= 0 || params.cond_inputs == nullptr ||
      params.body_inputs == nullptr || params.body_outputs == nullptr) {
    s->status = InvalidArgument(
        "TF_WhileParams must be created by successful TF_NewWhile() call");
    return false;
  }
  return true;
}

bool ValidateInputWhileParams(const TF_WhileParams& params, TF_Status* s) {
  if (params.cond_output.oper == nullptr) {
    s->status = InvalidArgument("TF_WhileParams `cond_output` field isn't set");
    return false;
  }
  for (int i = 0; i < params.ninputs; ++i) {
    if (params.body_outputs[i].oper == nullptr) {
      s->status = InvalidArgument("TF_WhileParams `body_outputs[", i, "]` ",
                                  "field isn't set");
      return false;
    }
  }
  if (params.name == nullptr) {
    s->status = InvalidArgument("TF_WhileParams `name` field is null");
    return false;
  }
  return true;
}

#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

void FreeWhileResources(const TF_WhileParams* params) {
  TF_DeleteGraph(params->cond_graph);
  TF_DeleteGraph(params->body_graph);
  delete[] params->cond_inputs;
  delete[] params->body_inputs;
  delete[] params->body_outputs;
}

TF_WhileParams EmptyWhileParams() {
  return {0,       nullptr, nullptr, {nullptr, 0},
          nullptr, nullptr, nullptr, nullptr};
}

}  // namespace

TF_WhileParams TF_NewWhile(TF_Graph* g, TF_Output* inputs, int ninputs,
                           TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Creating while loops is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
  return EmptyWhileParams();
#else
  if (ninputs == 0) {
    status->status =
        InvalidArgument("TF_NewWhile() must be passed at least one input");
    return EmptyWhileParams();
  }

  TF_Graph* cond_graph = TF_NewGraph();
  TF_Graph* body_graph = TF_NewGraph();
  cond_graph->parent = g;
  cond_graph->parent_inputs = inputs;
  body_graph->parent = g;
  body_graph->parent_inputs = inputs;

  TF_Output* cond_inputs = new TF_Output[ninputs];
  TF_Output cond_output = {nullptr, -1};
  TF_Output* body_inputs = new TF_Output[ninputs];
  TF_Output* body_outputs = new TF_Output[ninputs];
  for (int i = 0; i < ninputs; ++i) body_outputs[i] = {nullptr, -1};
  const char* name = nullptr;

  for (int i = 0; i < ninputs; ++i) {
    // TODO(skyewm): prefix names with underscore (requires some plumbing)
    if (!CreateInput(inputs[i], cond_graph, StrCat("cond_input", i).c_str(),
                     &cond_inputs[i], status)) {
      break;
    }
    if (!CreateInput(inputs[i], body_graph, StrCat("body_input", i).c_str(),
                     &body_inputs[i], status)) {
      break;
    }
  }

  TF_WhileParams params = {ninputs,    cond_graph,  cond_inputs,  cond_output,
                           body_graph, body_inputs, body_outputs, name};

  if (!status->status.ok()) {
    FreeWhileResources(&params);
    return EmptyWhileParams();
  }
  return params;
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
namespace {

// TODO(skyewm): make nodes in while loop unfetchable like in Python version
void TF_FinishWhileHelper(const TF_WhileParams* params, TF_Status* status,
                          TF_Output* outputs) {
  if (!ValidateInputWhileParams(*params, status)) return;

  TF_Graph* parent = params->cond_graph->parent;
  TF_Output* parent_inputs = params->cond_graph->parent_inputs;
  int num_loop_vars = params->ninputs;

  mutex_lock l(parent->mu);

  // 'cond_fn' copies the cond graph into the parent graph.
  tensorflow::ops::CondGraphBuilderFn cond_fn =
      [params, parent](const tensorflow::Scope& scope,
                       const std::vector<tensorflow::Output>& inputs,
                       tensorflow::Output* output) {
        DCHECK_EQ(scope.graph(), &parent->graph);
        std::vector<tensorflow::Output> cond_output;
        TF_RETURN_IF_ERROR(CopyGraph(
            &params->cond_graph->graph, &parent->graph, &parent->refiner,
            params->cond_inputs, inputs, scope.impl()->name(),
            scope.impl()->control_deps(), &params->cond_output,
            /* nreturn_nodes */ 1, &cond_output));
        *output = cond_output[0];
        return Status::OK();
      };

  // 'body_fn' copies the body graph into the parent graph.
  tensorflow::ops::BodyGraphBuilderFn body_fn =
      [params, parent, num_loop_vars](
          const tensorflow::Scope& scope,
          const std::vector<tensorflow::Output>& inputs,
          std::vector<tensorflow::Output>* outputs) {
        DCHECK_EQ(scope.graph(), &parent->graph);
        TF_RETURN_IF_ERROR(
            CopyGraph(&params->body_graph->graph, &parent->graph,
                      &parent->refiner, params->body_inputs, inputs,
                      scope.impl()->name(), scope.impl()->control_deps(),
                      params->body_outputs, num_loop_vars, outputs));
        return Status::OK();
      };

  // Create the while loop using an internal scope.
  tensorflow::Scope scope =
      NewInternalScope(&parent->graph, &status->status, &parent->refiner)
          .NewSubScope(params->name);

  const int first_new_node_id = parent->graph.num_node_ids();

  tensorflow::OutputList loop_outputs;
  status->status = tensorflow::ops::BuildWhileLoop(
      scope, OutputsFromTFOutputs(parent_inputs, num_loop_vars), cond_fn,
      body_fn, params->name, &loop_outputs);

  // Update name_map with newly-created ops.
  // TODO(skyewm): right now BuildWhileLoop() may alter the graph if it returns
  // a bad status. Once we fix this, we may want to return early instead of
  // executing the following code.
  for (int i = first_new_node_id; i < parent->graph.num_node_ids(); ++i) {
    Node* new_node = parent->graph.FindNodeId(i);
    if (new_node == nullptr) continue;
    parent->name_map[new_node->name()] = new_node;
  }

  // Populate 'outputs'.
  DCHECK_LE(loop_outputs.size(), num_loop_vars);
  for (int i = 0; i < loop_outputs.size(); ++i) {
    outputs[i] = {ToOperation(loop_outputs[i].node()), loop_outputs[i].index()};
  }
}

}  // namespace
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

void TF_FinishWhile(const TF_WhileParams* params, TF_Status* status,
                    TF_Output* outputs) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Creating while loops is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
#else
  // If it appears the caller created or modified `params`, don't free resources
  if (!ValidateConstWhileParams(*params, status)) return;
  TF_FinishWhileHelper(params, status, outputs);
  FreeWhileResources(params);
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

void TF_AbortWhile(const TF_WhileParams* params) { FreeWhileResources(params); }

void TF_AddGradients(TF_Graph* g, TF_Output* y, int ny, TF_Output* x, int nx,
                     TF_Output* dx, TF_Status* status, TF_Output* dy) {
  TF_AddGradientsWithPrefix(g, nullptr, y, ny, x, nx, dx, status, dy);
}

void TF_AddGradientsWithPrefix(TF_Graph* g, const char* prefix, TF_Output* y,
                               int ny, TF_Output* x, int nx, TF_Output* dx,
                               TF_Status* status, TF_Output* dy) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Adding gradients is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
#else
  std::vector<tensorflow::Output> y_arg = OutputsFromTFOutputs(y, ny);
  std::vector<tensorflow::Output> x_arg = OutputsFromTFOutputs(x, nx);
  std::vector<tensorflow::Output> dy_arg;

  {
    // We need to hold on to the lock while we have a scope that uses TF_Graph.
    mutex_lock graph_lock(g->mu);

    const int first_new_node_id = g->graph.num_node_ids();

    string prefix_cmp;
    const char* child_scope_name;
    if (prefix == nullptr) {
      child_scope_name = "gradients";
    } else {
      prefix_cmp = string(prefix) + "/";
      // The operation should fail if the provided name prefix has already been
      // used in this graph
      for (const auto& pair : g->name_map) {
        const string& name = pair.first;
        if ((name == prefix) || absl::StartsWith(name, prefix_cmp)) {
          status->status = InvalidArgument(
              "prefix [", prefix,
              "] conflicts with existing node in the graph named [", name, "]");
          return;
        }
      }
      child_scope_name = prefix;
    }
    tensorflow::Scope scope =
        NewInternalScope(&g->graph, &status->status, &g->refiner)
            .NewSubScope(child_scope_name);

    if (dx != nullptr) {
      std::vector<tensorflow::Output> dx_arg = OutputsFromTFOutputs(dx, ny);
      status->status =
          AddSymbolicGradients(scope, y_arg, x_arg, dx_arg, &dy_arg);
    } else {
      status->status = AddSymbolicGradients(scope, y_arg, x_arg, &dy_arg);
    }

    // Update g->name_map with the name_map from the scope, which will contain
    // the new gradient ops.
    for (int i = first_new_node_id; i < g->graph.num_node_ids(); ++i) {
      Node* n = g->graph.FindNodeId(i);
      if (n == nullptr) continue;

      // Adding the gradients to the graph can alter the prefix to prevent
      // name collisions only if this prefix has not been provided explicitly
      // by the user. If it was provided, assert that it remained intact.
      if (prefix != nullptr && !absl::StartsWith(n->name(), prefix_cmp)) {
        status->status = tensorflow::errors::Internal(
            "BUG: The gradients prefix have been unexpectedly altered when "
            "adding the nodes to the graph. This is a bug. Please file an "
            "issue at https://github.com/tensorflow/tensorflow/issues.");
        return;
      }
      // We have a convoluted scheme here: Using the C++ graph construction API
      // to add potentially many nodes to the graph without running the checks
      // (such as uniqueness of the names of nodes) we run with other functions
      // that add a node to the graph (like TF_FinishOperation).
      if (!g->name_map.insert(std::make_pair(n->name(), n)).second) {
        status->status = tensorflow::errors::Internal(
            "BUG: The API allowed construction of a graph with duplicate node "
            "names (",
            n->name(),
            "). This is a bug. Please file an issue at "
            "https://github.com/tensorflow/tensorflow/issues.");
      }
    }
  }

  // Unpack the results from grad_outputs_arg.
  TFOutputsFromOutputs(dy_arg, dy);
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

}  // end extern "C"
