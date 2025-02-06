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

#include "tensorflow/compiler/tf2xla/tf2xla.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/aot/aot_only_var_handle_op.h"
#include "tensorflow/compiler/tf2xla/graph_compiler_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_computation.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

// Converts the TensorFlow graph into an XLA computation, by executing the
// graph symbolically, with each op building up the XLA HLO.
absl::Status ConvertGraphToXla(std::unique_ptr<Graph> graph,
                               const tf2xla::Config& config,
                               xla::Client* client,
                               xla::XlaComputation* computation) {
  XlaOpRegistry::RegisterCompilationKernels();
  for (Node* node : graph->nodes()) {
    node->set_assigned_device_name(
        absl::StrCat("/device:", DEVICE_CPU_XLA_JIT));
  }
  std::vector<XlaCompiler::Argument> xla_args;
  TF_RETURN_IF_ERROR(CreateXlaArgs(*graph, &xla_args));

  PopulateXlaArgs(config, &xla_args);
  // Compile the graph into an XLA computation.
  XlaCompiler::Options compiler_options;
  compiler_options.client = client;
  compiler_options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
  compiler_options.flib_def = &graph->flib_def();
  compiler_options.graph_def_version = graph->versions().producer();
  compiler_options.allow_cpu_custom_calls = true;

  XlaCompiler compiler(compiler_options);

  XlaCompiler::CompilationResult result;

  XlaCompiler::CompileOptions options;
  options.alias_resource_update = true;
  TF_RETURN_IF_ERROR(compiler.CompileGraph(
      options, "tfcompile", std::move(graph), xla_args, &result));
  *computation = std::move(*result.computation);

  int num_const_results = 0;
  for (int i = 0, end = result.outputs.size(); i < end; ++i) {
    // Ending up with const results (i.e. output args) is an error, since it
    // means that one or more fetches that the user specified will be dropped
    // from the generated function.  It's most likely a configuration error,
    // since the user shouldn't be asking for output args that end up as consts.
    //
    // TODO(toddw): Provide a way for the user to access const output args,
    // e.g. perhaps hard-coded into the header, or somehow copied into the
    // output buffers.
    if (result.outputs[i].is_constant) {
      ++num_const_results;
      LOG(ERROR) << "ConstRetVal index:" << i
                 << " value:" << result.outputs[i].constant_value.DebugString();
    }
  }
  if (num_const_results > 0) {
    return errors::Unimplemented(
        "Conversion from TensorFlow graph to XLA resulted in ",
        num_const_results,
        " constant results.  The configuration of "
        "the output args (i.e. fetch ids) is probably wrong.");
  }
  {
    // Verify that the readonly bits on variables are set correctly by the user.
    std::vector<bool> updated_inputs(xla_args.size());
    for (const XlaCompiler::ResourceUpdate& update : result.resource_updates) {
      updated_inputs[update.input_index] = true;
    }
    int64_t input_index = xla_args.size() - config.variable_size();
    for (const tf2xla::Variable& variable : config.variable()) {
      if (variable.readonly() == updated_inputs[input_index]) {
        return errors::InvalidArgument(
            "Variable \"", variable.node_name(), "\" is marked as ",
            variable.readonly() ? "" : "not ", "readonly, but is ",
            updated_inputs[input_index] ? "" : "not ",
            "modified by the computation.");
      }
      ++input_index;
    }
  }
  return absl::OkStatus();
}

absl::Status ConvertVarHandlesToAotVarHandles(GraphDef* graph_def) {
  auto update_var_handle_op_node = [](NodeDef& node) -> absl::Status {
    if (node.op() == "VarHandleOp") {
      node.set_op(tfcompile::kXlaAotOnlyVarHandleOp);
      const auto& it = node.attr().find("allowed_devices");
      if (it != node.attr().end()) {
        if (!it->second.list().s().empty()) {
          return errors::InvalidArgument(
              "VarHandleOp with non-empty allowed devices is not supported.");
        }
        node.mutable_attr()->erase("allowed_devices");
      }
    }
    return absl::OkStatus();
  };
  for (auto& node : *graph_def->mutable_node()) {
    TF_RETURN_IF_ERROR(update_var_handle_op_node(node));
  }
  for (auto& fn : *graph_def->mutable_library()->mutable_function()) {
    for (auto& node : *fn.mutable_node_def()) {
      TF_RETURN_IF_ERROR(update_var_handle_op_node(node));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ConvertGraphDefToXla(GraphDef graph_def,
                                  const tf2xla::Config& config,
                                  xla::Client* client,
                                  xla::XlaComputation* computation) {
  std::unique_ptr<Graph> graph;
  TF_RETURN_IF_ERROR(ConvertVarHandlesToAotVarHandles(&graph_def));
  TF_RETURN_IF_ERROR(InitGraph(graph_def, config, &graph));
  TF_RETURN_IF_ERROR(
      ConvertGraphToXla(std::move(graph), config, client, computation));
  return absl::OkStatus();
}

}  // namespace tensorflow
