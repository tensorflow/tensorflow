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

#include "tensorflow/core/kernels/hexagon/graph_transfer_utils.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {

// function alias
constexpr auto AddOutputTensorShapeTypeByTensorShapeMap =
    &RemoteFusedGraphExecuteUtils::AddOutputTensorShapeTypeByTensorShapeMap;

/* static */ std::priority_queue<std::tuple<float, int, string>>
GraphTransferUtils::GetTopNFloatResults(const float* const data,
                                        const string* const labels,
                                        const int element_count) {
  CHECK(data != nullptr);
  CHECK(labels != nullptr);
  std::priority_queue<std::tuple<float, int, string>> queue;
  for (int i = 0; i < element_count; ++i) {
    queue.emplace(data[i], i, labels[i]);
  }
  return queue;
}

/* static */ void GraphTransferUtils::DumpTopNFloatResults(
    const float* const data, const string* const labels,
    const int element_count, const int top_n) {
  std::priority_queue<std::tuple<float, int, string>> queue =
      GetTopNFloatResults(data, labels, element_count);
  LOG(INFO) << "=== Dump ranking ===";
  for (int i = 0; i < top_n; ++i) {
    const std::tuple<float, int, string>& entry = queue.top();
    LOG(INFO) << i << ": " << std::get<1>(entry) << ", " << std::get<2>(entry)
              << ", " << std::get<0>(entry);
    queue.pop();
  }
}

/* static */ RemoteFusedGraphExecuteInfo
GraphTransferUtils::BuildRemoteFusedGraphExecuteInfo(
    const GraphDef& graph_def,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& outputs,
    const RemoteFusedGraphExecuteUtils::TensorShapeMap& tensor_shape_map) {
  RemoteFusedGraphExecuteInfo execute_info;
  execute_info.set_executor_name("build_hexagon_remote_fused_graph_executor");

  // copy graph
  *execute_info.mutable_remote_graph() = graph_def;

  for (const std::pair<string, Tensor>& input : inputs) {
    execute_info.add_graph_input_node_name(input.first);
    RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& tensor_shape_type =
        *execute_info.add_default_graph_input_tensor_shape();
    tensor_shape_type.set_dtype(input.second.dtype());
    TensorShapeProto& tensor_shape_proto = *tensor_shape_type.mutable_shape();
    for (const int64 dim : input.second.shape().dim_sizes()) {
      tensor_shape_proto.add_dim()->set_size(dim);
    }
  }

  for (const string& output_name : outputs) {
    const std::pair<DataType, TensorShape>* tensor_shape_type =
        RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                         output_name);
    CHECK_NOTNULL(tensor_shape_type);
    execute_info.add_graph_output_node_name(output_name);
    RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& tensor_shape_type_proto =
        *execute_info.add_default_graph_output_tensor_shape();
    tensor_shape_type_proto.set_dtype(tensor_shape_type->first);
    TensorShapeProto& tensor_shape_proto =
        *tensor_shape_type_proto.mutable_shape();
    for (const int64 dim : tensor_shape_type->second.dim_sizes()) {
      tensor_shape_proto.add_dim()->set_size(dim);
    }
  }

  return execute_info;
}

/* static */ GraphDef GraphTransferUtils::BuildFusedGraphDef(
    const IRemoteFusedGraphOpsDefinitions& ops_definitions,
    const string& remote_graph_execute_name,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& outputs, GraphDef* original_def) {
  RemoteFusedGraphExecuteUtils::TensorShapeMap tensor_shape_map;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
      *original_def, inputs, true /* initialize_by_zero */, &tensor_shape_map);
  for (NodeDef& node_def : *original_def->mutable_node()) {
    TF_CHECK_OK(
        AddOutputTensorShapeTypeByTensorShapeMap(tensor_shape_map, &node_def));
  }
  CHECK(status.ok());

  Scope root = Scope::NewRootScope();
  std::vector<Output> output_list;
  DataTypeVector input_types;
  for (const std::pair<string, Tensor>& input_node_info : inputs) {
    const Scope& scope = root.WithOpName(input_node_info.first);
    Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("Placeholder");
    auto builder = NodeBuilder(unique_name, "Placeholder")
                       .Attr("dtype", input_node_info.second.dtype())
                       .Attr("shape", input_node_info.second.shape());
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    TF_CHECK_OK(scope.status());
    output_list.emplace_back(Output(ret, 0));
    input_types.push_back(input_node_info.second.dtype());
  }

  const RemoteFusedGraphExecuteInfo execute_info =
      BuildRemoteFusedGraphExecuteInfo(*original_def, inputs, outputs,
                                       tensor_shape_map);

  DataTypeVector output_types;
  // Sanity-check to confirm all output data types are same.
  for (const string& output_node_name : outputs) {
    const std::pair<DataType, TensorShape>* tst =
        RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                         output_node_name);
    CHECK_NE(tst, nullptr);
    output_types.push_back(tst->first);
  }

  const Scope& scope = root.WithOpName(remote_graph_execute_name);
  CHECK(scope.ok());
  auto node_out_list = ops::AsNodeOutList(scope, InputList(output_list));
  Node* node;
  const auto unique_name = scope.GetUniqueNameForOp("RemoteFusedGraphExecute");

  auto builder = NodeBuilder(unique_name, "RemoteFusedGraphExecute")
                     .Input(node_out_list)
                     .Attr("Tinputs", input_types)
                     .Attr("Toutputs", output_types)
                     .Attr("serialized_remote_fused_graph_execute_info",
                           StringPiece(execute_info.SerializeAsString()));
  CHECK(scope.ok());
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &node));
  CHECK(scope.ok()) << scope.status();

  GraphDef fusedGraphDef;
  TF_CHECK_OK(root.ToGraphDef(&fusedGraphDef));
  return fusedGraphDef;
}

}  // namespace tensorflow
