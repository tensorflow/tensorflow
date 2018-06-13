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
static const string kInputPHName = "InputPH_";
static const string kOutputPHName = "OutputPH_";
namespace convert {

const int FP32MODE = 0;
const int FP16MODE = 1;
const int INT8MODE = 2;
struct EngineConnections {
  EngineConnections(const string& outside, int out_id, int out_port,
                    const string& inside, int in_id, int in_port,
                    bool input_edge, int port)
      : outside_node_name(outside),
        outside_id(out_id),
        outside_port(out_port),
        inside_node_name(inside),
        inside_id(in_id),
        inside_port(in_port),
        is_input_edge(input_edge),
        port_number(port) {}
  const string outside_node_name;
  const int outside_id;
  const int outside_port;
  tensorflow::PartialTensorShape outside_shape;
  tensorflow::DataType connection_type;
  const string inside_node_name;
  const int inside_id;
  const int inside_port;
  tensorflow::PartialTensorShape inside_shape;
  bool is_input_edge;
  int port_number;
};

struct EngineInfo {
  EngineInfo()
      : engine_type(EngineType::TRTStatic),
        max_workspace_size_bytes(0),
        precision_mode(FP32MODE){};
  string engine_name;
  string device;
  tensorflow::GraphDef segment_graph_def;
  std::vector<EngineConnections> connections;  // order matters!
  enum class EngineType { TRTStatic = 0, TRTDynamic = 1 };
  EngineType engine_type;
  tensorflow::int64 max_workspace_size_bytes;
  int maximum_cached_engines;
  std::vector<int> cached_engine_batches;
  int precision_mode;
};
;

//  Constructs a graphdef from the segment in the given graph. Adds placeholder
//  nodes for input edges (InputPH_*) and identity nodes for output edges
//  (OutputPH_*).  This function needs to be called before TensorRT nodes
//  inserted in order to correctly get sizes from the original graph.
tensorflow::Status ConvertSegmentToGraphDef(
    const tensorflow::Graph* graph,
    const tensorflow::grappler::GraphProperties& graph_properties,
    const std::vector<int>& subgraph_node_ids,
    std::vector<EngineConnections>* connections,
    tensorflow::GraphDef* segment_def, string* common_scope);

// Converts given subgraph to a TRT engine.
tensorflow::Status ConvertSubgraphToEngine(
    const tensorflow::GraphDef& gdef, nvinfer1::IBuilder* builder,
    const std::vector<tensorflow::PartialTensorShape>& input_shapes,
    nvinfer1::ICudaEngine** engine, int precision_mode);
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_NODES_H_
