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

#ifndef TENSORFLOW_PLATFORM_HEXAGON_GRAPH_TRANSFER_UTILS_H_
#define TENSORFLOW_PLATFORM_HEXAGON_GRAPH_TRANSFER_UTILS_H_

#include <queue>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/remote_fused_graph_execute_info.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class GraphTransferUtils {
 public:
  static std::priority_queue<std::tuple<float, int, string>>
  GetTopNFloatResults(const float* const data, const string* const labels,
                      const int element_count);

  static void DumpTopNFloatResults(const float* const data,
                                   const string* const labels,
                                   const int element_count, const int top_n);

  static RemoteFusedGraphExecuteInfo BuildRemoteFusedGraphExecuteInfo(
      const GraphTransferInfo& graph_transfer_info);

  static GraphDef BuildFusedGraphDef(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const string& remote_graph_execute_name,
      const std::vector<GraphTransferer::InputNodeInfo>& inputs,
      const std::vector<string>& outputs, const GraphDef& def,
      GraphTransferer* gt);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GraphTransferUtils);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_HEXAGON_GRAPH_TRANSFER_UTILS_H_
