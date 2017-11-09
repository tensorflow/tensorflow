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

// bazel build tensorflow/contrib/hvx/hvx_ops_support_checker &&
// bazel-bin/tensorflow/contrib/hvx/hvx_ops_support_checker/hvx_ops_support_checker
// \
// --in_graph=graph_def.pb

#include <unordered_set>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/remote_fused_graph_execute_info.pb.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_ops_definitions.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/file_utils.h"

namespace tensorflow {

namespace {
static int ParseFlags(int argc, char* argv[], string* in_graph,
                      bool* dump_all_nodes, bool* dump_shape_and_type) {
  std::vector<Flag> flag_list = {
      Flag("in_graph", in_graph, "Input graph file name to check hvx support."),
      Flag("dump_all_nodes", dump_all_nodes, "Dump all nodes in the model."),
      Flag("dump_shape_and_type", dump_shape_and_type,
           "Dump shape and type of nodes"),
  };
  CHECK(Flags::Parse(&argc, argv, flag_list));
  // We need to call this to set up global state for TensorFlow.
  port::InitMain(argv[0], &argc, &argv);

  string usage = Flags::Usage(argv[0], flag_list);
  CHECK(!in_graph->empty()) << "in_graph graph can't be empty.\n" << usage;

  return 0;
}

static void SummarizeNode(const NodeDef& node_def,
                          const bool dump_shape_and_type) {
  LOG(INFO) << "Node(" << node_def.name() << ")";
  LOG(INFO) << "  op: " << node_def.op();
  for (const string& input : node_def.input()) {
    LOG(INFO) << " Input: " << input;
  }
  std::vector<DataType> data_types;
  std::vector<TensorShape> shapes;
  const Status status = RemoteFusedGraphExecuteUtils::GetOutputTensorShapeType(
      node_def, &data_types, &shapes);
  if (data_types.empty() || shapes.empty()) {
    return;
  }
  CHECK_EQ(data_types.size(), shapes.size());
  for (int i = 0; i < data_types.size(); ++i) {
    LOG(INFO) << " Output(" << i << "): " << DataType_Name(data_types.at(i))
              << ", " << shapes.at(i).DebugString();
  }
}

static void DumpRemoteFusedGraph(const NodeDef& node_def) {
  LOG(INFO) << "Remote fused graph found.";
  RemoteFusedGraphExecuteInfo info;
  string serialized_proto;
  GetNodeAttr(node_def,
              RemoteFusedGraphExecuteUtils::
                  ATTR_SERIALIZED_REMOTE_FUSED_GRAPH_EXECUTE_INFO,
              &serialized_proto)
      .IgnoreError();
  info.ParseFromString(serialized_proto);
  LOG(INFO) << "Node name: " << node_def.name();
  LOG(INFO) << "Executor name: " << info.executor_name();
  for (const string& input : info.graph_input_node_name()) {
    LOG(INFO) << "Input: " << input;
  }
  for (const RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& shape_type :
       info.default_graph_input_tensor_shape()) {
    LOG(INFO) << "Input shape type: " << shape_type.DebugString();
  }
  for (const string& output : info.graph_output_node_name()) {
    LOG(INFO) << "Output: " << output;
  }
  for (const RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& shape_type :
       info.default_graph_output_tensor_shape()) {
    LOG(INFO) << "Output shape type: " << shape_type.DebugString();
  }
  const int subgraph_node_size = info.remote_graph().node_size();
  LOG(INFO) << "Nodes in the graph: " << subgraph_node_size;
  for (int i = 0; i < subgraph_node_size; ++i) {
    LOG(INFO) << "node(" << i << "): " << info.remote_graph().node(i).name();
  }
}

static void CheckOpsSupport(const GraphDef& graph_def,
                            const bool dump_all_nodes,
                            const bool dump_shape_and_type) {
  const IRemoteFusedGraphOpsDefinitions& ops_definition =
      HexagonOpsDefinitions::getInstance();
  LOG(INFO) << "Checking " << graph_def.node_size() << " nodes";
  LOG(INFO) << "dump_all_nodes = " << dump_all_nodes
            << ", dump_shape_and_tpye = " << dump_shape_and_type;

  std::unordered_set<string> unsupported_ops;
  bool all_supported = true;
  bool contains_remote_graph = false;
  for (const NodeDef& node : graph_def.node()) {
    if (node.op() == "RemoteFusedGraphExecute") {
      contains_remote_graph = true;
      DumpRemoteFusedGraph(node);
      continue;
    }
    // TODO(satok): Set correct data type if it's given.
    const int op_id = ops_definition.GetOpIdFor(node.op(), {});
    if (op_id == IRemoteFusedGraphOpsDefinitions::INVALID_OP_ID) {
      all_supported = false;
      LOG(ERROR) << "OP type: " << node.op() << " is not supported on hvx. "
                 << "Name = " << node.name();
      unsupported_ops.emplace(node.op());
    }
  }

  LOG(INFO) << "\n";
  LOG(INFO) << "Unsupported ops:";
  int count = 0;
  for (const string& op_type : unsupported_ops) {
    LOG(INFO) << "(" << (++count) << ") " << op_type;
  }
  if (count == 0) {
    LOG(INFO) << "All ops supported!";
  } else {
    LOG(INFO) << count << " ops are not supported.";
  }

  if (contains_remote_graph || dump_all_nodes) {
    for (const NodeDef& node : graph_def.node()) {
      SummarizeNode(node, dump_shape_and_type);
    }
  }
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::string in_graph;
  bool dump_all_nodes;
  bool dump_shape_and_type;
  const int ret = tensorflow::ParseFlags(argc, argv, &in_graph, &dump_all_nodes,
                                         &dump_shape_and_type);
  if (ret != 0) {
    return ret;
  }

  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(tensorflow::graph_transforms::LoadTextOrBinaryGraphFile(
      in_graph, &graph_def));

  tensorflow::CheckOpsSupport(graph_def, dump_all_nodes, dump_shape_and_type);
  return 0;
}
