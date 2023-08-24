/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/optimized_function_graph_info.h"

#include <memory>
#include <utility>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

OptimizedFunctionGraph OptimizedFunctionGraphInfo::ToProto(
    const OptimizedFunctionGraphInfo& info) {
  OptimizedFunctionGraph proto;
  proto.set_name(info.name);
  GraphDef* function_graph_def = proto.mutable_function_graph();
  info.function_graph->ToGraphDef(function_graph_def);
  // Set lib_def into the function_graph.
  *function_graph_def->mutable_library() = info.lib_def.ToProto();
  *proto.mutable_ret_types() = {info.ret_types.begin(), info.ret_types.end()};
  proto.set_num_return_nodes(info.num_return_nodes);
  *proto.mutable_node_name_to_control_ret() = {
      info.node_name_to_control_ret.begin(),
      info.node_name_to_control_ret.end()};
  proto.set_optimization_time_usecs(info.optimization_duration_usecs);
  proto.set_source(info.optimization_source);
  return proto;
}

StatusOr<OptimizedFunctionGraphInfo> OptimizedFunctionGraphInfo::FromProto(
    const OptimizedFunctionGraph& proto) {
  // Reconstruct the lib_def.
  FunctionLibraryDefinition lib_def(OpRegistry::Global(),
                                    proto.function_graph().library());

  // Reconstruct the graph.
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.expect_device_spec = true;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(options, proto.function_graph(), graph.get()));

  // Clear both library and registry as the op lookup should be from lib_def.
  graph->mutable_flib_def()->set_default_registry(nullptr);
  graph->mutable_flib_def()->Clear();

  const int num_ret_types = proto.ret_types_size();
  DataTypeVector data_type_vector(num_ret_types);
  for (int i = 0; i < num_ret_types; ++i) {
    // Need to explicityly convert to the enum type.
    data_type_vector[i] = static_cast<DataType>(proto.ret_types().at(i));
  }
  return OptimizedFunctionGraphInfo(
      proto.name(), std::move(graph), std::move(lib_def),
      {proto.node_name_to_control_ret().begin(),
       proto.node_name_to_control_ret().end()},
      std::move(data_type_vector), proto.num_return_nodes(),
      proto.optimization_time_usecs(), proto.source());
}

}  // namespace tensorflow
