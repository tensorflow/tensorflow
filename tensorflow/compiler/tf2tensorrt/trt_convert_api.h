/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_TRT_CONVERT_API_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_TRT_CONVERT_API_H_

#include <climits>
#include <string>
#include <vector>

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {

struct SavedModelBundle;

namespace tensorrt {

struct TfTrtConversionParams {
  // Corresponds 'workspaceSize' parameter of
  // nvinfer1::IBuilderConfig::setMaxWorkspaceSize.
#if IS_TRT_VERSION_GE(8, 4, 0, 0)
  // Must use `LLONG_MAX - 512` to avoid overflow during casting.
  size_t max_workspace_size_bytes = LLONG_MAX - 512;
#else
  size_t max_workspace_size_bytes = 1 << 30;  // 1,073,741,824
#endif

  // Minimum precision used by the TRT Engine.
  TrtPrecisionMode precision_mode = TrtPrecisionMode::FP32;

  // The minimum number of nodes required for a subgraph to be replaced by
  // TRTEngineOp. Note that many small TRT subgraphs could be detrimental for
  // performance, increasing the minimum segment size can help avoid the
  // problem.
  int minimum_segment_size = 3;

  // Max number of cached TRT engines for dynamic TRT ops (by default we have
  // dynamic TRT ops).
  int max_cached_engines = 1;

  // Note that calibration is currently not implemented with the C++ converter.
  // This argument is ignored if precision_mode is not INT8. If set to True, the
  // implementation will use the user provided inputs to generate calibration
  // data. If set to False, quantization nodes will be expected for every tensor
  // in the graph (excluding those which will be fused). If a range is missing,
  // an error will occur. Please note that accuracy may be negatively affected
  // if there is a mismatch between which tensors TRT quantizes and which
  // tensors were trained with fake quantization.
  bool use_calibration = true;

  // Whether to enable dynamic shape mode for the TRT engines. It is
  // recommended to use_dynamic_shape mode to handle dynamic input shape.
  // Enabling dynamic shape mode can also improve the conversion rate of graphs
  // with static input shape.
  bool use_dynamic_shape = true;

  // In dynamic shape mode we create an engine that can handle various input
  // shape ranges. We derive the shape optimization profiles for the TRT engines
  // in the graph based on user provided input data and profile_strategy.
  ProfileStrategy profile_strategy = ProfileStrategy::kRange;

  // Whether to allow bulding TRT engines at runtime. If no TensorRT engine can
  // be found in cache that can handle the given inputs during runtime, then a
  // new TensorRT engine is built at runtime if allow_build_at_runtime=True,
  // otherwise native TF is used. We recommend to set this value false and build
  // the engine in advance, to avoid runtime overhead.
  bool allow_build_at_runtime = true;

  // Record the TRT engine as an attribute of the TRTEngineOp. This is only
  // valid when max_cached_engines == 1. Note: the frozen graph together with
  // the serialized engines have to be below 2GiB (protobuf size limit). If
  // convert_to_static_engine = false, then the converted graph_def only
  // contains placeholder TRTEngineOp nodes.
  bool convert_to_static_engine = true;
};

/**
 * Converts the graph with TF-TRT.
 *
 * Performs TF-TRT conversion and returns the converted GraphDef. If inputs is
 * not empty and convert_to_static_engine is requested, we also build the
 * engines and convert the engines to static engines.
 *
 * Arguments:
 * - frozen_graph_def input graph, it is assumed to be frozen
 * - input_names names of the input tensors
 * - output_names names of the output tensors
 * - inputs tensors that we will use as input while building the TRT engines
 * - conv_params parameters for the TF-TRT conversion
 *
 * Returns the converted graph_def.
 */
StatusOr<GraphDef> ConvertAndBuild(
    const GraphDef& frozen_graph_def, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<std::vector<tensorflow::Tensor>>& inputs,
    const TfTrtConversionParams& conv_params);

StatusOr<GraphDef> ConvertAndBuild(
    SavedModelBundle* bundle,
    const std::string& signature_key = "serving_default",
    const std::vector<std::vector<tensorflow::Tensor>>& inputs = {},
    const TfTrtConversionParams& conversion_params = TfTrtConversionParams());

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_TRT_CONVERT_API_H_
