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
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {

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
    const std::tuple<float, int, string> &entry = queue.top();
    LOG(INFO) << i << ": " << std::get<1>(entry) << ", " << std::get<2>(entry)
              << ", " << std::get<0>(entry);
    queue.pop();
  }
}

/* static */ RemoteFusedGraphExecuteInfo
GraphTransferUtils::BuildRemoteFusedGraphExecuteInfo(
    const GraphTransferInfo& graph_transfer_info) {
  RemoteFusedGraphExecuteInfo execute_info;
  execute_info.set_executor_name("build_hexagon_remote_fused_graph_executor");
  for (const GraphTransferInfo::GraphInputNodeInfo& input :
       graph_transfer_info.graph_input_node_info()) {
    execute_info.add_graph_input_node_name(input.name());
    RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& tensor_shape_type =
        *execute_info.add_default_graph_input_tensor_shape();
    tensor_shape_type.set_dtype(input.dtype());
    TensorShapeProto& tensor_shape_proto = *tensor_shape_type.mutable_shape();
    for (const int64 dim : input.shape()) {
      tensor_shape_proto.add_dim()->set_size(dim);
    }
  }

  for (const GraphTransferInfo::GraphOutputNodeInfo& output :
       graph_transfer_info.graph_output_node_info()) {
    execute_info.add_graph_output_node_name(output.name());
    RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& tensor_shape_type =
        *execute_info.add_default_graph_output_tensor_shape();
    tensor_shape_type.set_dtype(output.dtype());
    TensorShapeProto& tensor_shape_proto = *tensor_shape_type.mutable_shape();
    for (const int64 dim : output.shape()) {
      tensor_shape_proto.add_dim()->set_size(dim);
    }
  }

  execute_info.set_serialized_executor_parameters(
      graph_transfer_info.SerializeAsString());
  return execute_info;
}

/* static */ GraphDef GraphTransferUtils::BuildFusedGraphDef(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const string& remote_graph_execute_name,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& outputs, const GraphDef& def,
    GraphTransferer* const gt) {
  CHECK(gt != nullptr);
  RemoteFusedGraphExecuteUtils::TensorShapeMap tensor_shape_map;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
      def, inputs, true /* initialize_by_zero */, &tensor_shape_map);
  CHECK(status.ok());
  status = gt->LoadGraphFromProto(ops_definitions, def, inputs, outputs, false,
                                  tensor_shape_map);
  const DataType input_data_type =
      inputs.empty() ? DT_FLOAT : inputs.at(0).second.dtype();

  Scope root = Scope::NewRootScope();
  std::vector<Output> output_list;
  for (const std::pair<string, Tensor>& input_node_info : inputs) {
    const Scope& scope = root.WithOpName(input_node_info.first);
    Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("PlaceholderV2");
    const DataType dt = input_node_info.second.dtype();
    // DataType of input arguments should be same.
    CHECK_EQ(input_data_type, dt);
    auto builder = NodeBuilder(unique_name, "PlaceholderV2")
                       .Attr("dtype", input_node_info.second.dtype())
                       .Attr("shape", input_node_info.second.shape());
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    CHECK(scope.ok());
    output_list.emplace_back(Output(ret, 0));
  }

  const RemoteFusedGraphExecuteInfo execute_info =
      BuildRemoteFusedGraphExecuteInfo(gt->GetGraphTransferInfo());

  const std::pair<DataType, TensorShape>* tensor_shape_type =
      RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                       outputs.at(0));
  CHECK_NE(tensor_shape_type, nullptr);
  const DataType output_data_type = tensor_shape_type->first;
  // Sanity-check to confirm all output data types are same.
  for (const string& output_node_name : outputs) {
    const std::pair<DataType, TensorShape>* tst =
        RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                         output_node_name);
    CHECK_NE(tst, nullptr);
    const DataType dt = tensor_shape_type->first;
    CHECK_EQ(output_data_type, dt);
  }

  const Scope& scope = root.WithOpName(remote_graph_execute_name);
  CHECK(scope.ok());
  auto node_out_list = ops::AsNodeOutList(scope, InputList(output_list));
  Node* node;
  const auto unique_name = scope.GetUniqueNameForOp("RemoteFusedGraphExecute");
  auto builder = NodeBuilder(unique_name, "RemoteFusedGraphExecute")
                     .Input(node_out_list)
                     .Attr("M", static_cast<int64>(output_list.size()))
                     .Attr("N", static_cast<int64>(outputs.size()))
                     .Attr("T", input_data_type)
                     .Attr("U", output_data_type)
                     .Attr("serialized_graph_transfer_info",
                           StringPiece(execute_info.SerializeAsString()));
  CHECK(scope.ok());
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &node));
  CHECK(scope.ok());

  GraphDef fusedGraphDef;
  TF_CHECK_OK(root.ToGraphDef(&fusedGraphDef));
  return fusedGraphDef;
}

}  // namespace tensorflow
