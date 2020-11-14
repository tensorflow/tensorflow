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

#ifndef TENSORFLOW_CORE_KERNELS_REMOTE_FUSED_GRAPH_EXECUTE_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_REMOTE_FUSED_GRAPH_EXECUTE_UTILS_H_

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_ops_definitions.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

enum RemoteFusedGraphNodeType {
  UNUSED = 0,
  GRAPH_INPUT = 1,
  GRAPH_OUTPUT = 2,
  FUSED_NODE = 3,
  BORDER_INPUT = 4,
  BORDER_OUTPUT = 5,
};

class RemoteFusedGraphExecuteInfo;

// RemoteFusedGraphExecuteUtils provides APIs to register and get builder
// functions for IRemoteFusedGraphExecutor.
class RemoteFusedGraphExecuteUtils {
 public:
  // TODO(satok): Use "_output_data_types" to share a spec with other ops
  static constexpr const char* const ATTR_OUTPUT_DATA_TYPES =
      "_default_remote_graph_output_data_types";
  // TODO(satok): Use "_output_shapes" to share a spec with other ops
  static constexpr const char* const ATTR_OUTPUT_SHAPES =
      "_default_remote_output_shapes";
  static constexpr const char* const
      ATTR_SERIALIZED_REMOTE_FUSED_GRAPH_EXECUTE_INFO =
          "serialized_remote_fused_graph_execute_info";
  static constexpr const char* const ATTR_NODE_TYPE =
      "_remote_fused_graph_node_type";

  // Argument key strings to fuse a subgraph into RemoteFusedGraphExecuteOp.
  static constexpr const char* const
      TRANSFORM_ARG_REMOTE_FUSED_GRAPH_EXECUTOR_NAME =
          "remote_fused_graph_executor_name";
  static constexpr const char* const
      TRANSFORM_ARG_REMOTE_FUSED_GRAPH_NODE_NAME =
          "remote_fused_graph_node_name";
  static constexpr const char* const TRANSFORM_ARG_FUSED_NODES = "fused_nodes";
  static constexpr const char* const TRANSFORM_ARG_BORDER_INPUTS =
      "border_inputs";
  static constexpr const char* const TRANSFORM_ARG_BORDER_OUTPUTS =
      "border_outputs";
  static constexpr const char* const TRANSFORM_ARG_FUSED_OP_TYPES =
      "fused_op_types";
  static constexpr const char* const TRANSFORM_ARG_FUSE_BY_EXECUTOR =
      "fuse_by_executor";
  static constexpr const char* const TRANSFORM_ARG_INPUT_TYPES = "input_types";
  static constexpr const char* const TRANSFORM_ARG_INPUT_SHAPES =
      "input_shapes";

  using ExecutorBuildFunc = std::function<Status(
      std::unique_ptr<IRemoteFusedGraphExecutor>* executor)>;
  // Registrar class for IRemoteFusedGraphExecutor.
  class ExecutorBuildRegistrar {
   public:
    ExecutorBuildRegistrar(const string& name, ExecutorBuildFunc func);

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBuildRegistrar);
  };
  using ExecutorBuildRegistry = std::map<string, ExecutorBuildFunc>;

  using TensorShapeType = std::pair<DataType, TensorShape>;
  using TensorShapeMap = std::unordered_multimap<string,         // node name
                                                 std::pair<int,  // port
                                                           TensorShapeType>>;
  using ClusterInfo = std::tuple<std::unordered_set<string>,  // node names
                                 std::vector<string>,         // border inputs
                                 std::vector<string>>;        // border outputs

  // Return registered ExecutorBuildFunc for given name.
  static const ExecutorBuildFunc* GetExecutorBuildFunc(const string& name);

  // To determine shapes of output tensors of all nodes, dryrun the graph.
  // This function supplies memory allocation information when loading
  // the graph. This function is used to verify shape inference and actual
  // output shape.
  static Status DryRunInference(
      const GraphDef& graph_def,
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const std::vector<string>& output_node_names,
      const bool initialize_by_zero,
      std::vector<tensorflow::Tensor>* output_tensors);

  // Dry run inference to obtain shapes for all nodes.
  // CAVEAT: Do not add or modify output_tensors in output_tensor_info
  // otherwise, address map may be broken by re-allocation inside
  // std::vector.
  static Status DryRunInferenceForAllNode(
      const GraphDef& graph_def,
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const bool initialize_by_zero, TensorShapeMap* tensor_shape_map);

  static bool IsInputNode(
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const string& node_name);

  static void ConvertToTensorShapeMap(
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const std::vector<string>& output_node_names,
      const std::vector<tensorflow::Tensor>& output_tensors,
      TensorShapeMap* tensor_shape_map);

  static Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                    Tensor* tensor);

  static bool AddOutputTensorShapeType(const std::vector<DataType>& data_types,
                                       const std::vector<TensorShape>& shapes,
                                       NodeDef* node_def);

  static Status AddOutputTensorShapeTypeByTensorShapeMap(
      const TensorShapeMap& tensor_shape_map, NodeDef* node_def);

  static Status GetOutputTensorShapeType(AttrSlice attrs,
                                         std::vector<DataType>* data_types,
                                         std::vector<TensorShape>* shapes);

  static bool GetOutputTensorShapeType(const GraphDef& graph_def,
                                       const string& name_and_port,
                                       DataType* data_type, TensorShape* shape);

  static Status PropagateShapeInference(
      const GraphDef& graph_def,
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      Graph* graph, ShapeRefiner* shape_refiner);

  static Status BuildTensorShapeMapFromGraph(const Graph& graph,
                                             const ShapeRefiner& shape_refiner,
                                             TensorShapeMap* tensor_shape_map);

  static const TensorShapeType* GetTensorShapeType(
      const TensorShapeMap& tensor_shape_map, const string& node_name);

  static const TensorShapeType* GetTensorShapeType(
      const TensorShapeMap& tensor_shape_map, const string& node_name,
      const int port);

  static void BuildRemoteGraphInputsAndOutputsFromProto(
      const RemoteFusedGraphExecuteInfo& proto,
      std::vector<std::pair<string, Tensor>>* inputs,
      std::vector<string>* outputs);

  static Status BuildAndAddTensorShapes(
      const std::vector<std::pair<string, Tensor>>& input_tensors,
      const bool dry_run_inference, GraphDef* graph_def);

  // Build remote fused graph execute info.
  static Status BuildRemoteFusedGraphExecuteInfo(
      const string& executor_name, const GraphDef& subgraph_def,
      const std::vector<string>& inputs, const std::vector<string>& outputs,
      const bool require_shape_type, RemoteFusedGraphExecuteInfo* execute_info,
      DataTypeVector* input_types, DataTypeVector* output_types);

  // Build remote fused graph execute op node by fusing specified subgraph
  // as remote fused graph execute info.
  static Status BuildRemoteFusedGraphExecuteOpNode(
      const string& node_name, const string& executor_name,
      const GraphDef& subgraph_def, const std::vector<string>& inputs,
      const std::vector<string>& outputs, const bool require_shape_type,
      Graph* graph, Node** created_node);

  // Build Identity node to forward remote graph node output.
  static Status BuildIdentityOpNode(const string& node_name,
                                    const string& input_node_name,
                                    const int input_node_port,
                                    const DataType dt, Graph* graph,
                                    Node** created_node);

  // Create clusters of given nodes.
  static Status ClusterizeNodes(const std::unordered_set<string>& node_names,
                                const GraphDef& graph_def,
                                std::vector<ClusterInfo>* cluster_infos);

  // Build GraphDef of a given cluster.
  static Status BuildClusterSubgraphDef(const ClusterInfo& cluster,
                                        const GraphDef& graph_def,
                                        GraphDef* subgraph_def);

  // Build a cluster by given border.
  // CAVEAT: The border must be consistent for one cluster.
  static Status BuildClusterByBorder(const std::vector<string>& border_inputs,
                                     const std::vector<string>& border_outputs,
                                     const GraphDef& graph_def,
                                     ClusterInfo* cluster);

  // Fuse one cluster into a newly created RemoteFusedGraphExecuteOp node.
  // The subgraph is stored as a graph in RemoteFusedGraphExecuteInfo.
  // CAVEAT1: This transform strips unvisited nodes with given outputs.
  // CAVEAT2: If you want to use a graph output as a border output,
  // that graph output node is replaced by an identity node.  Therefore,
  // the number of output of the node must be 1.
  static Status FuseCluster(const GraphDef& input_graph_def,
                            const std::vector<string>& inputs,
                            const std::vector<string>& outputs,
                            const string& remote_fused_graph_node_name,
                            const ClusterInfo& cluster,
                            const string& remote_graph_executor_name,
                            const bool require_shape_type,
                            GraphDef* output_graph_def);

  // Fuse subgraph of specified nodes.
  static Status FuseRemoteGraphByNodeNames(
      const GraphDef& input_graph_def, const std::vector<string>& inputs,
      const std::vector<string>& outputs,
      const string& remote_fused_graph_node_name_prefix,
      const std::unordered_set<string>& subgraph_nodes,
      const string& remote_fused_graph_executor_name,
      const bool require_shape_type, GraphDef* output_graph_def);

  // Fuse subgraph of specified border.
  static Status FuseRemoteGraphByBorder(
      const GraphDef& input_graph_def, const std::vector<string>& inputs,
      const std::vector<string>& outputs,
      const string& remote_fused_graph_node_name,
      const std::vector<string>& border_inputs,
      const std::vector<string>& border_outputs,
      const string& remote_graph_executor_name, const bool require_shape_type,
      GraphDef* output_graph_def);

  // Fuse subgraph of specified op types.
  static Status FuseRemoteGraphByOpTypes(
      const GraphDef& input_graph_def, const std::vector<string>& inputs,
      const std::vector<string>& outputs,
      const string& remote_fused_graph_node_name_prefix,
      const std::unordered_set<string>& fused_op_types,
      const string& remote_fused_graph_executor_name,
      const bool require_shape_type, GraphDef* output_graph_def);

  // Place arguments to fuse remote graph.
  static Status PlaceRemoteGraphArguments(
      const std::vector<string>& inputs, const std::vector<string>& outputs,
      const std::unordered_set<string>& fused_node_names,
      const std::vector<string>& border_inputs,
      const std::vector<string>& border_outputs,
      const std::unordered_set<string>& fused_op_types,
      const string& remote_fused_graph_node_name,
      const string& remote_graph_executor_name, GraphDef* graph_def);

  // Fuse remote graph by placed arguments.
  static Status FuseRemoteGraphByPlacedArguments(
      const GraphDef& input_graph_def,
      const std::vector<std::pair<string, Tensor>>& input_tensors,
      GraphDef* output_graph_def);

  static Status FuseRemoteGraphByExecutor(const GraphDef& input_graph_def,
                                          const std::vector<string>& inputs,
                                          const std::vector<string>& outputs,
                                          const string& executor_name,
                                          GraphDef* output_graph_def);

  static bool IsFuseReady(
      const GraphDef& input_graph_def,
      const std::vector<std::pair<string, Tensor>>& input_tensors);

  // Copy a byte array to a tensor data.  Though tensor data must be
  // updated with typed information in general, we can't guarantee that
  // returned values from a remote processor has typed information because
  // a logic running in the remote processor possibly be in a separate binary
  // which may not link tensorflow libraries.  To deal with this situation,
  // remote fused graph needs to overwrite the tensor data by a byte array.
  static Status CopyByteArrayToTensor(const void* src_ptr, const int src_size,
                                      Tensor* tensor);

  static std::unordered_set<string> BuildNodeMapFromOpTypes(
      const GraphDef& graph_def, const std::unordered_set<string>& op_types);

  static std::unordered_set<string> BuildNodeMapFromOpsDefinitions(
      const GraphDef& graph_def,
      const IRemoteFusedGraphOpsDefinitions& ops_definitions);

 private:
  static void EmplaceTensorShapeType(const string& name, const Tensor& tensor,
                                     TensorShapeMap* tensor_shape_map);

  static Status ReplaceInputNodeByPlaceHolder(const string& input,
                                              const DataType type,
                                              const TensorShape& shape,
                                              GraphDef* graph_def);

  static ExecutorBuildRegistry* GetExecutorBuildRegistry();

  static string BuildNodeTypeAttr(const RemoteFusedGraphNodeType node_type,
                                  const int port, const int index,
                                  const string& executor_name,
                                  const string& node_name);

  static string BuildNodeTypeAttr(const RemoteFusedGraphNodeType node_type,
                                  const int port, const int index);

  static string BuildNodeTypeAttr(const RemoteFusedGraphNodeType node_type);

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteFusedGraphExecuteUtils);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REMOTE_FUSED_GRAPH_EXECUTE_UTILS_H_
