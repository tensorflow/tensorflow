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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

const int FP32MODE = 0;
const int FP16MODE = 1;
const int INT8MODE = 2;

struct SubGraphParams {
  SubGraphParams(
      tensorflow::Graph& inp_graph,
      const std::set<int>& subgraph_node_id_numbers,
      const std::vector<std::pair<int, int>>& input_indices,
      const std::vector<std::pair<int, int>>& output_indices,
      size_t max_supported_batch_size, size_t max_consumed_workspace_size_bytes,
      const tensorflow::grappler::GraphProperties& current_graph_properties,
      std::unordered_map<string, std::pair<int, string>>* output_edges,
      tensorflow::NodeDef* constructed_trt_node,
      int engine_precision_mode = FP32MODE, const string& device_name = "",
      std::shared_ptr<nvinfer1::IGpuAllocator> allocator = nullptr,
      int cuda_gpu_id = 0)
      : graph(inp_graph),
        subgraph_node_ids(subgraph_node_id_numbers),
        input_inds(input_indices),
        output_inds(output_indices),
        max_batch_size(max_supported_batch_size),
        max_workspace_size_bytes(max_consumed_workspace_size_bytes),
        graph_properties(current_graph_properties),
        output_edge_map(output_edges),
        trt_node(constructed_trt_node),
        precision_mode(engine_precision_mode),
        device_name_(device_name),
        allocator_(allocator),
        cuda_gpu_id_(cuda_gpu_id) {}

  tensorflow::Graph& graph;
  const std::set<int>& subgraph_node_ids;
  const std::vector<std::pair<int, int>>& input_inds;   // {node_id, output_idx}
  const std::vector<std::pair<int, int>>& output_inds;  // {node_id, output_idx}
  size_t max_batch_size;
  size_t max_workspace_size_bytes;
  const tensorflow::grappler::GraphProperties& graph_properties;
  std::unordered_map<string, std::pair<int, string>>* output_edge_map;
  tensorflow::NodeDef* trt_node;
  const int precision_mode;
  const string device_name_;
  std::shared_ptr<nvinfer1::IGpuAllocator> allocator_;
  const int cuda_gpu_id_;
};

// TODO(sami): Replace references with const reference or pointers
tensorflow::Status ConvertSubGraphToTensorRTNodeDef(SubGraphParams& params);
tensorflow::Status InjectCalibrationNode(SubGraphParams& params);
tensorflow::Status ConvertCalibrationNodeToEngineNode(tensorflow::Graph& graph,
                                                      tensorflow::Node* c_node);
tensorflow::Status ConvertSegmentToGraphDef(
    tensorflow::tensorrt::convert::SubGraphParams& params,
    tensorflow::GraphDef* segment_def,
    std::unordered_map<string,string> input_placeholder_map);
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
