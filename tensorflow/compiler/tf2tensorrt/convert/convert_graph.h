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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_

#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

struct ConversionParams {
  const grappler::GrapplerItem* grappler_item = nullptr;
  const std::vector<string>* input_output_names = nullptr;
  string trt_logger_name;
  size_t max_batch_size = 1;
  size_t max_workspace_size_bytes = 1 << 30;
  GraphDef* output_graph_def = nullptr;
  TrtPrecisionMode precision_mode = TrtPrecisionMode::FP32;
  int minimum_segment_size = 3;
  const grappler::Cluster* cluster = nullptr;
  // Whether to create engine on conversion or execution time
  bool is_dyn_op = false;
  // maximum number of cached engines
  int max_cached_engines = 1;
  bool use_calibration = true;
  bool use_implicit_batch = true;
  ProfileStrategy profile_strategy = ProfileStrategy::kRange;
  bool allow_build_at_runtime = true;
  bool use_explicit_precision = false;
};

// Method to call from optimization pass
Status ConvertAfterShapes(const ConversionParams& params);

// Helper method for the conversion, expose for testing.
std::pair<int, Allocator*> GetDeviceAndAllocator(const ConversionParams& params,
                                                 const EngineInfo& engine);

// Helper method that registers `segment_graph` as a function to the function
// library in `graph`.
Status RegisterGraphToFunctionLibrary(const GraphDef& segment_graph_def,
                                      Graph* graph, const string& engine_name);

// Creates and serializes an ICudaEngine. Used only in is_dynamic_op=false,
// a.k.a. static engine mode.
Status CreateStaticEngine(const ConversionParams& params,
                          const EngineInfo& info, int max_batch_size,
                          const std::vector<PartialTensorShape>& input_shapes,
                          TrtShapeOptimizationProfile* profile,
                          string* segment_string);

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_
