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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_TRT_CONVERT_API_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_TRT_CONVERT_API_H_

#include <climits>
#include <string>
#include <vector>

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace tensorrt {

#if IS_TRT_VERSION_GE(8, 4, 0, 0)
// Must use `LLONG_MAX - 512` to avoid overflow during casting.
#define DEFAULT_MAX_WORKSPACE_SIZE_BYTES LLONG_MAX - 512
#else
#define DEFAULT_MAX_WORKSPACE_SIZE_BYTES 1 << 30  // 1,073,741,824
#endif

struct TrtConversionParams {
  // Corresponds 'workspaceSize' parameter of
  // nvinfer1::IBuilderConfig::setMaxWorkspaceSize.
  size_t max_workspace_size_bytes = DEFAULT_MAX_WORKSPACE_SIZE_BYTES;

  // Minimum precision used by the TRT Engine.
  TrtPrecisionMode precision_mode = TrtPrecisionMode::FP32;

  // The minimum number of nodes required for a subgraph to be replaced by
  // TRTEngineOp. Note that many small TRT subgraphs could be detrimental for
  // performance, increasing the minimum segment size can help avoid the
  // problem.
  int minimum_segment_size = 3;

  // Max number of cached TRT engines for dynamic TRT ops (by default we have
  // dynamic TRT ops).
  int maximum_cached_engines = 1;

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
  bool use_dynamic_shape = false;

  // In dynamic shape mode we create an engine that can handle various input
  // shape ranges. We derive the shape optimization profiles for the TRT engines
  // in the graph based on user provided input data and profile_strategy.
  ProfileStrategy dynamic_shape_profile_strategy = ProfileStrategy::kRange;

  // Whether to allow bulding TRT engines at runtime. If no TensorRT engine can
  // be found in cache that can handle the given inputs during runtime, then a
  // new TensorRT engine is built at runtime if allow_build_at_runtime=True,
  // otherwise native TF is used. We recommend to set this value false and build
  // the engine in advance, to avoid runtime overhead.
  bool allow_build_at_runtime = true;
};

class TrtGraphConverter {
 public:
  static StatusOr<std::unique_ptr<TrtGraphConverter>> Create(
      const GraphDef& frozen_graph_def,
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      const TrtConversionParams& conversion_params = TrtConversionParams());

  // Experimental feature that does not freeze the graph.
  static StatusOr<std::unique_ptr<TrtGraphConverter>> Create(
      const std::string& saved_model_dir,
      const std::string& signature_key = "serving_default",
      const std::unordered_set<std::string>& tags = {"serve"},
      const TrtConversionParams& conversion_params = TrtConversionParams());

  // Converts the graph with TF-TRT.
  //
  // Arguments:
  // - inputs: the calibration input data for INT8 precision. If
  //   `use_dynamic_shape` is false, we additionally run
  //   calibration and build the calibrated engine.
  // - disable_non_trt_optimizers: if true, disable all other optimizations
  //   and only run the TF-TRT segmentation.
  // - device_requested: if set, moves the graph to the requested GPU.
  //
  // Returns: the converted GraphDef.
  StatusOr<GraphDef> Convert(
      const std::vector<std::vector<tensorflow::Tensor>>& inputs = {},
      bool disable_non_trt_optimizers = false,
      const std::string& device_requested = "");

  // Run inference with the converted graph in order to build TensorRT engines.
  //
  // Arguments:
  // - inputs: the input data that will be used to generate the TRT engines.
  //
  // Returns: the converted graph with static engines.
  StatusOr<GraphDef> Build(
      const std::vector<std::vector<tensorflow::Tensor>>& inputs);

  // Serialize the built TensorRT engines to disk.
  //
  // Arguments:
  // - out_dir: the output directory to write to.
  //
  // Returns: a map of engine names and their corresponding filepaths.
  StatusOr<std::map<std::string, std::string>> SerializeEngines(
      const std::string& out_dir, bool save_gpu_specific_engines);

  // The GraphDef to convert.
  GraphDef input_graph_def;

  // Session that is used for building and calibration. This session
  // can be used for inference on the built model.
  std::unique_ptr<tensorflow::Session> session;

 private:
  TrtGraphConverter(const GraphDef& input_graph_def,
                    const std::vector<std::string>& input_names,
                    const std::vector<std::string>& output_names,
                    const TrtConversionParams& conversion_params);

  Status Validate(bool expect_frozen = true);

  Status ExecuteCalibration();

  // The resulting segmented GraphDef from calling Convert.
  GraphDef segmented_graph_def_;

  // Names of input tensors for the graph.
  const std::vector<std::string> input_names_;

  // Names of output tensors for the graph.
  const std::vector<std::string> output_names_;

  // A TrtConversionParams instance.
  const TrtConversionParams conversion_params_;

  // Prefix of the segmented graph in `session` after conversion.
  std::string segmented_prefix_;

  // Inputs to use for calibration.
  std::vector<std::vector<tensorflow::Tensor>> calibration_inputs_;

  enum class ConversionStatus { kUnconverted, kConverted, kBuilt };
  ConversionStatus conversion_status_;

  enum class CalibrationStatus {
    kShouldNotCalibrate,
    kNeedsCalibration,
    kCalibrated
  };
  CalibrationStatus calibration_status_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_TRT_CONVERT_API_H_
