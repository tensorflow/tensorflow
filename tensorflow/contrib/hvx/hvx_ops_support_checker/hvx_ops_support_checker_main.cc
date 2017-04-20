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
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace {
static int ParseFlags(int argc, char* argv[], string* in_graph) {
  std::vector<Flag> flag_list = {
      Flag("in_graph", in_graph, "input graph file name"),
  };
  CHECK(Flags::Parse(&argc, argv, flag_list));
  string usage = Flags::Usage(argv[0], flag_list);
  CHECK(!in_graph->empty()) << "in_graph graph can't be empty.\n" << usage;

  return 0;
}

static void CheckOpsSupport(const GraphDef& graph_def) {
  const IGraphTransferOpsDefinitions& ops_definition =
      HexagonOpsDefinitions::getInstance();
  LOG(INFO) << "Checking " << graph_def.node_size() << " nodes";

  std::unordered_set<string> unsupported_ops;
  bool all_supported = true;
  for (const NodeDef& node : graph_def.node()) {
    const int op_id = ops_definition.GetOpIdFor(node.op());
    if (op_id == IGraphTransferOpsDefinitions::INVALID_OP_ID) {
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
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::string in_graph;
  const int ret = tensorflow::ParseFlags(argc, argv, &in_graph);
  if (ret != 0) {
    return ret;
  }

  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(tensorflow::graph_transforms::LoadTextOrBinaryGraphFile(
      in_graph, &graph_def));

  tensorflow::CheckOpsSupport(graph_def);
  return 0;
}
