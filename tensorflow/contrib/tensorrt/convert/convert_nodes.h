/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
#define TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

tensorflow::Status ConvertSubGraphToTensorRTNodeDef(
    const tensorflow::Graph& graph, const std::set<int>& subgraph_node_ids,
    const std::vector<std::pair<int, int>>&
        input_inds,  // {node_id, output_idx}
    const std::vector<std::pair<int, int>>&
        output_inds,  // {node_id, output_idx}
    size_t max_batch_size,
    size_t max_workspace_size,
    const tensorflow::grappler::GraphProperties& graph_prop,
    tensorflow::NodeDef* trt_node);

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
