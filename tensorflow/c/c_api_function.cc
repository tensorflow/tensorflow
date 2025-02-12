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

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "absl/strings/match.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_buffer_internal.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/base64.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/util/debug_data_dumper.h"

using tensorflow::errors::InvalidArgument;

namespace tensorflow {
namespace {

absl::Status ValidateNonRefOutput(const Node* node, int idx) {
  const DataType& dt = node->output_type(idx);
  return IsRefType(dt)
             ? InvalidArgument("Output ", idx, " of node '", node->name(),
                               "' has a reference type ", DataTypeString(dt))
             : absl::OkStatus();
}

// Converts `ninputs` and `inputs` into `inputs_tensors` and `input_nodes` and
// does various checks while doing so. `input_nodes` will contain the same
// information as input_tensors just in a different structure to make
// following processing easier. TODO(iga): Simplify this nested structure.
absl::Status ProcessInputs(
    const TF_Graph* fn_body, const char* fn_name, int ninputs,
    const TF_Output* inputs, std::vector<OutputTensor>* input_tensors,
    std::unordered_map<const Node*, std::vector<int>>* input_nodes)
    TF_EXCLUSIVE_LOCKS_REQUIRED(fn_body->mu) {
  input_tensors->reserve(ninputs);
  for (int i = 0; i < ninputs; ++i) {
    Node* node = inputs[i].oper ? &inputs[i].oper->node : nullptr;
    int idx = inputs[i].index;

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        fn_body->graph.IsValidOutputTensor(node, idx),
        "Encountered while processing input ", i, " into function '", fn_name,
        "'");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(ValidateNonRefOutput(node, idx),
                                    "Encountered while processing input ", i,
                                    " into function '", fn_name, "'");

    input_tensors->emplace_back(node, idx);

    const auto& iter = input_nodes->find(node);
    if (iter == input_nodes->end()) {
      input_nodes->insert({node, {idx}});
    } else {
      auto& indices = iter->second;
      if (std::find(indices.begin(), indices.end(), idx) != indices.end()) {
        return InvalidArgument("TF_Output ", node->name(), ":", idx,
                               " appears more than once in the input list");
      }
      indices.push_back(idx);
    }
  }
  return absl::OkStatus();
}

// Converts `noutputs` and `outputs` into `outputs_tensors` and does various
// checks while doing so.
absl::Status ProcessOutputs(const TF_Graph* fn_body, const char* fn_name,
                            int noutputs, const TF_Output* outputs,
                            std::vector<OutputTensor>* output_tensors)
    TF_EXCLUSIVE_LOCKS_REQUIRED(fn_body->mu) {
  output_tensors->reserve(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    Node* node = outputs[i].oper ? &outputs[i].oper->node : nullptr;
    int idx = outputs[i].index;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        fn_body->graph.IsValidOutputTensor(node, idx),
        "Encountered while processing output ", i, " from function '", fn_name,
        "'");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(ValidateNonRefOutput(node, idx),
                                    "Encountered while creating function '",
                                    fn_name, "'");
    output_tensors->emplace_back(node, idx);
  }
  return absl::OkStatus();
}

// Populates `body_nodes` with the nodes that will become function's body.
// Performs various checks.
absl::Status ComputeBodyNodes(
    const TF_Graph* fn_body, const char* fn_name, int num_opers,
    const TF_Operation* const* opers,
    const std::unordered_map<const Node*, std::vector<int>>& input_nodes,
    std::vector<const Node*>* body_nodes)
    TF_EXCLUSIVE_LOCKS_REQUIRED(fn_body->mu) {
  if (num_opers == -1) {
    for (const Node* node : fn_body->graph.op_nodes()) {
      const auto& iter = input_nodes.find(node);
      if (iter == input_nodes.end()) {
        // This node is not referenced in inputs. Add it to the body.
        body_nodes->push_back(node);
      } else {
        // This node is referenced in inputs. Currently, we place an
        // artificial restriction and require that when num_opers=-1, such
        // nodes must have a single output.
        if (node->num_outputs() != 1) {
          return InvalidArgument(
              "When `num_opers` is set to -1, nodes referenced in `inputs` "
              "must have a single output. Node ",
              node->name(), " has ", node->num_outputs(),
              " outputs. Encountered while creating function '", fn_name, "'");
        }
      }
    }
  } else {
    body_nodes->reserve(num_opers);
    for (int i = 0; i < num_opers; ++i) {
      const Node* node = &opers[i]->node;
      body_nodes->push_back(node);
    }
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace tensorflow

using tensorflow::Node;
using tensorflow::string;

TF_Function* TF_GraphToFunctionWithControlOutputs(
    const TF_Graph* fn_body, const char* fn_name,
    unsigned char append_hash_to_fn_name, int num_opers,
    const TF_Operation* const* opers, int ninputs, const TF_Output* inputs,
    int noutputs, const TF_Output* outputs, const char* const* output_names,
    int ncontrol_outputs, const TF_Operation* const* control_outputs,
    const char* const* control_output_names, const TF_FunctionOptions* opts,
    const char* description, TF_Status* status) {
  tensorflow::mutex_lock l(fn_body->mu);

  // Process inputs.
  std::vector<tensorflow::OutputTensor> input_tensors;
  std::unordered_map<const Node*, std::vector<int>> input_nodes;
  status->status = tensorflow::ProcessInputs(fn_body, fn_name, ninputs, inputs,
                                             &input_tensors, &input_nodes);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // Process outputs.
  std::vector<tensorflow::OutputTensor> output_tensors;
  status->status = tensorflow::ProcessOutputs(fn_body, fn_name, noutputs,
                                              outputs, &output_tensors);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // Process output names.
  std::vector<string> output_names_vec;
  if (output_names) {
    output_names_vec.reserve(noutputs);
    for (int i = 0; i < noutputs; ++i) {
      output_names_vec.push_back(string(output_names[i]));
    }
  }

  // Process control output names.
  std::vector<string> control_output_names_vec;
  if (control_output_names) {
    control_output_names_vec.reserve(ncontrol_outputs);
    for (int i = 0; i < ncontrol_outputs; ++i) {
      control_output_names_vec.push_back(string(control_output_names[i]));
    }
  }

  // Compute body nodes.
  std::vector<const Node*> body_nodes;
  status->status = tensorflow::ComputeBodyNodes(
      fn_body, fn_name, num_opers, opers, input_nodes, &body_nodes);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // Compute body nodes.
  std::vector<const Node*> control_output_nodes;
  control_output_nodes.reserve(ncontrol_outputs);
  for (int i = 0; i < ncontrol_outputs; ++i) {
    control_output_nodes.push_back(&control_outputs[i]->node);
  }

  // Do the actual function creation.
  DCHECK(append_hash_to_fn_name <= 1);
  tensorflow::FunctionDef fdef;
  status->status = tensorflow::GraphToFunctionDef(
      fn_body->graph, fn_name, append_hash_to_fn_name != 0,
      /*set_stateful_from_nodes=*/true,
      /*copy_placeholder_attrs_from_nodes=*/true, body_nodes, input_tensors,
      output_tensors, output_names_vec, control_output_nodes,
      control_output_names_vec, description, &fdef);
  if (TF_GetCode(status) != TF_OK) {
    return nullptr;
  }

  // Dump the op creation stacktraces for debugging purpose.
  DEBUG_DATA_DUMPER()->DumpOpCreationStackTraces(
      fn_name, kDebugGroupOpStacktrace, "initial", &fn_body->graph);

  tensorflow::StackTracesMap stack_traces;
  for (const Node* n : fn_body->graph.nodes()) {
    stack_traces[n->name()] = n->GetStackTrace();
  }

  TF_Function* tf_function = new TF_Function();
  tf_function->record = new tensorflow::FunctionRecord(
      std::move(fdef), std::move(stack_traces), false);

  return tf_function;
}

TF_Function* TF_GraphToFunction(const TF_Graph* fn_body, const char* fn_name,
                                unsigned char append_hash_to_fn_name,
                                int num_opers, const TF_Operation* const* opers,
                                int ninputs, const TF_Output* inputs,
                                int noutputs, const TF_Output* outputs,
                                const char* const* output_names,
                                const TF_FunctionOptions* opts,
                                const char* description, TF_Status* status) {
  return TF_GraphToFunctionWithControlOutputs(
      fn_body, fn_name, append_hash_to_fn_name, num_opers, opers, ninputs,
      inputs, noutputs, outputs, output_names, 0, nullptr, nullptr, opts,
      description, status);
}

const char* TF_FunctionName(TF_Function* func) {
  return func->record->fdef().signature().name().c_str();
}

void TF_GraphCopyFunction(TF_Graph* g, const TF_Function* func,
                          const TF_Function* grad, TF_Status* status) {
  if (func == nullptr) {
    status->status = InvalidArgument(
        "'func' argument to TF_GraphCopyFunction cannot be null");
    return;
  }

  tensorflow::mutex_lock l(g->mu);
  status->status = g->graph.AddFunctionDef(func->record->fdef(),
                                           func->record->stack_traces());
  if (TF_GetCode(status) != TF_OK) return;
  if (!grad) return;

  status->status = g->graph.AddFunctionDef(grad->record->fdef(),
                                           grad->record->stack_traces());
  if (TF_GetCode(status) != TF_OK) return;

  tensorflow::GradientDef gdef;
  gdef.set_function_name(func->record->fdef().signature().name());
  gdef.set_gradient_func(grad->record->fdef().signature().name());
  status->status = g->graph.AddGradientDef(std::move(gdef));
}

int TF_GraphNumFunctions(TF_Graph* g) {
  tensorflow::mutex_lock l(g->mu);
  return g->graph.flib_def().num_functions();
}

int TF_GraphGetFunctions(TF_Graph* g, TF_Function** funcs, int max_func,
                         TF_Status* status) {
  tensorflow::FunctionDefLibrary lib;
  {
    tensorflow::mutex_lock l(g->mu);
    lib = g->graph.flib_def().ToProto();
  }
  const auto len = std::min(max_func, static_cast<int>(lib.function_size()));
  for (int i = 0; i < len; ++i) {
    TF_Function* func = new TF_Function();
    func->record = new tensorflow::FunctionRecord(lib.function(i), {}, false);
    funcs[i] = func;
  }
  status->status = absl::OkStatus();
  return len;
}

void TF_FunctionToFunctionDef(TF_Function* func, TF_Buffer* output_func_def,
                              TF_Status* status) {
  status->status = MessageToBuffer(func->record->fdef(), output_func_def);
}

TF_Function* TF_FunctionImportFunctionDef(const void* proto, size_t proto_len,
                                          TF_Status* status) {
  tensorflow::FunctionDef fdef;
  bool success = fdef.ParseFromArray(proto, proto_len);
  if (!success) {
    status->status = InvalidArgument(
        "Invalid FunctionDef given to TF_FunctionImportFunctionDef");
    return nullptr;
  }

  TF_Function* func = new TF_Function();
  func->record = new tensorflow::FunctionRecord(std::move(fdef), {}, false);
  status->status = absl::OkStatus();
  return func;
}

void TF_FunctionSetAttrValueProto(TF_Function* func, const char* attr_name,
                                  const void* proto, size_t proto_len,
                                  TF_Status* status) {
  tensorflow::AttrValue attr_value;
  if (!attr_value.ParseFromArray(proto, proto_len)) {
    status->status = InvalidArgument(
        "Unparseable AttrValue proto passed to "
        "TF_FunctionSetAttrValueProto");
    return;
  }

  auto fdef_or = func->record->mutable_fdef();
  if (!fdef_or.ok()) {
    status->status = fdef_or.status();
    return;
  }

  (*(fdef_or.value()->mutable_attr()))[string(attr_name)] = attr_value;

  status->status = absl::OkStatus();
}

void TF_FunctionGetAttrValueProto(TF_Function* func, const char* attr_name,
                                  TF_Buffer* output_attr_value,
                                  TF_Status* status) {
  const auto& it = func->record->fdef().attr().find(attr_name);
  if (it == func->record->fdef().attr().end()) {
    status->status =
        InvalidArgument("Function '", func->record->fdef().signature().name(),
                        "' has no attr named '", attr_name, "'.");
    return;
  }
  status->status = MessageToBuffer(it->second, output_attr_value);
}

void TF_DeleteFunction(TF_Function* func) {
  if (func == nullptr) {
    return;
  }

  func->record->Unref();
  func->record = nullptr;
  delete func;
}
