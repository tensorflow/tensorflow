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
#ifndef TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_GRAPH_H_
#define TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_GRAPH_H_

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

// This method converts an already generated calibration graph which was used in
// calibration runs to an inference graph
struct ConversionParams {
  ConversionParams()
      : input_graph_def(nullptr),
        max_batch_size(1),
        max_workspace_size_bytes(1 << 30),
        output_graph_def(nullptr),
        precision_mode(1),
        minimum_segment_size(3),
        graph_properties(nullptr),
        cluster(nullptr),
        is_dyn_op(false),
        fixed_input_size(true),
        max_cached_engines(1) {}
  const tensorflow::GraphDef* input_graph_def;
  const std::vector<string>* output_names;
  size_t max_batch_size;
  size_t max_workspace_size_bytes;
  tensorflow::GraphDef* output_graph_def;
  int precision_mode;
  int minimum_segment_size;
  const tensorflow::grappler::GraphProperties* graph_properties;
  const tensorflow::grappler::Cluster* cluster;
  bool is_dyn_op;
  bool fixed_input_size;
  int max_cached_engines;
  std::vector<int> cached_engine_batches;
};
tensorflow::Status ConvertCalibGraphToInferGraph(
    const tensorflow::GraphDef& graph_def, tensorflow::GraphDef* new_graph_def);

// max_batch_size: maximum batch size which can be used for inference for
//                 optimization targets inference run with max batch size.
// max_workspace_size_bytes: The upper bound of memory allowance for
//                 engine building.
tensorflow::Status ConvertGraphDefToTensorRT(
    const tensorflow::GraphDef& graph_def,
    const std::vector<string>& output_names, size_t max_batch_size,
    size_t max_workspace_size_bytes, tensorflow::GraphDef* new_graph_def,
    int precision_mode=1, int minimum_segment_size=3, bool is_dyn_op = false,
    int max_cached_engines = 1, std::vector<int> cached_engine_batches={});

// Method to call from optimization pass
tensorflow::Status ConvertAfterShapes(ConversionParams& params);
std::vector<int> GetLinkedTensorRTVersion();
std::vector<int> GetLoadedTensorRTVersion();
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_CONVERT_CONVERT_GRAPH_H_
