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

#include "tensorflow/compiler/tf2tensorrt/experimental/trt_convert_api.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/tf2tensorrt/experimental/utils/model_optim.h"
#include "tensorflow/compiler/tf2tensorrt/experimental/utils/session_utils.h"
#include "tensorflow/compiler/tf2tensorrt/experimental/utils/trt_op_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_experimental_features.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/platform/errors.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

namespace {

// Sets the _profile_generation mode attribute of all TRTEngineOp nodes in the
// graph to mode.
Status SetProfileGenerationMode(GraphDef* graph_def, bool mode) {
  VLOG(3) << "Setting _profile_generation_mode=" << mode;
  std::string op{"TRTEngineOp"};
  for (auto& node : *(graph_def->mutable_node())) {
    if (!op.compare(node.op())) {
      auto* attr = node.mutable_attr();
      AttrValue profile_generation_mode;
      profile_generation_mode.set_b(mode);
      (*attr)["_profile_generation_mode"] = profile_generation_mode;
    }
  }
  return OkStatus();
}

Status SaveCalibrationTable(Session* session, NodeDef& node,
                            const std::string& prefix) {
  std::string data;
  auto status = GetCalibrationData(session, node, prefix, &data);
  if (errors::IsUnknown(status) || errors::IsNotFound(status)) {
    LOG(WARNING) << "Warning calibration error for " << node.name();
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(status);
  auto* attr = node.mutable_attr();
  AttrValue calib_data;
  calib_data.set_s(data);
  (*attr)["calibration_data"] = calib_data;
  return OkStatus();
}

// Simple sanity check that the graph is frozen.
// Returns ok if no variable nodes are found.
Status IsFrozenGraph(const GraphDef& graph) {
  std::set<std::string> variable_ops = {"VariableV2", "VarHandleOp",
                                        "ReadVariableOp", "ResourceGather",
                                        "ResourceGatherNd"};
  for (const auto& node : graph.node()) {
    if (variable_ops.find(node.op()) != variable_ops.end()) {
      return errors::InvalidArgument("Input graph must be frozen.");
    }
  }
  return OkStatus();
}

Status ValidateConversionParams(const TrtConversionParams& p) {
  if (p.minimum_segment_size < -1) {
    return errors::InvalidArgument(
        "Minimum segment size should be positive or -1 (to disable main graph "
        "conversion).");
  }
  return OkStatus();
}

}  // namespace

// static
StatusOr<std::unique_ptr<TrtGraphConverter>> TrtGraphConverter::Create(
    const GraphDef& frozen_graph_def,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const TrtConversionParams& conversion_params) {
  std::unique_ptr<TrtGraphConverter> converter =
      absl::WrapUnique(new TrtGraphConverter(frozen_graph_def, input_names,
                                             output_names, conversion_params));
  TF_RETURN_IF_ERROR(converter->Validate());
  return converter;
}

// static
StatusOr<std::unique_ptr<TrtGraphConverter>> TrtGraphConverter::Create(
    const std::string& saved_model_dir, const std::string& signature_key,
    const std::unordered_set<std::string>& tags,
    const TrtConversionParams& conversion_params) {
  if (!isExperimentalFeatureActivated("disable_graph_freezing")) {
    return errors::InvalidArgument(
        "The `disable_graph_freezing` experimental feature must be enabled.");
  }
  if (!conversion_params.use_dynamic_shape) {
    return errors::InvalidArgument(
        "Disabling graph freezing is only possible in dynamic shape mode.");
  }

  SavedModelBundle bundle;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  TF_RETURN_IF_ERROR(LoadSavedModel(saved_model_dir, signature_key, tags,
                                    &bundle, &input_names, &output_names));

  // Inline
  TF_RETURN_IF_ERROR(ApplyInlining(bundle.meta_graph_def, saved_model_dir,
                                   input_names, output_names, &bundle.session));
  TF_RETURN_IF_ERROR(
      AnnotateVariableOps(bundle.meta_graph_def.mutable_graph_def()));

  // Create converter
  std::unique_ptr<TrtGraphConverter> converter = absl::WrapUnique(
      new TrtGraphConverter(bundle.meta_graph_def.graph_def(), input_names,
                            output_names, conversion_params));
  TF_RETURN_IF_ERROR(converter->Validate(false));
  converter->session.release();
  converter->session = std::move(bundle.session);
  converter->segmented_prefix_ = "TrtSegmentedGraph";
  return converter;
}

TrtGraphConverter::TrtGraphConverter(
    const GraphDef& input_graph_def,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const TrtConversionParams& conversion_params)
    : input_graph_def(input_graph_def),
      session(NewSession(GetSessionConfig())),
      input_names_(input_names),
      output_names_(output_names),
      conversion_params_(conversion_params),
      conversion_status_(ConversionStatus::kUnconverted),
      calibration_status_(
          (conversion_params.use_calibration &&
           conversion_params.precision_mode == TrtPrecisionMode::INT8)
              ? CalibrationStatus::kNeedsCalibration
              : CalibrationStatus::kShouldNotCalibrate) {}

Status TrtGraphConverter::Validate(bool expect_frozen) {
  if (!session.get()) {
    return errors::Internal("Failed to create build session");
  }
  if (expect_frozen) {
    TF_RETURN_IF_ERROR(IsFrozenGraph(input_graph_def));
  }
  TF_RETURN_IF_ERROR(ValidateConversionParams(conversion_params_));
  return OkStatus();
}

Status TrtGraphConverter::ExecuteCalibration() {
  for (auto const& input : calibration_inputs_) {
    TF_RETURN_IF_ERROR(RunSession(session.get(), input_names_, output_names_,
                                  input, segmented_prefix_));
  }
  std::string op{"TRTEngineOp"};
  std::string calib_prefix{"TRTGetCalibrationData"};
  TF_RETURN_IF_ERROR(SetupGetCalibrationDataOp(session.get(), calib_prefix));
  for (auto& node : *(segmented_graph_def_.mutable_node())) {
    if (!op.compare(node.op())) {
      TF_RETURN_IF_ERROR(
          SaveCalibrationTable(session.get(), node, calib_prefix));
    }
  }
  calibration_status_ = CalibrationStatus::kCalibrated;
  return OkStatus();
}

StatusOr<GraphDef> TrtGraphConverter::Convert(
    const std::vector<std::vector<tensorflow::Tensor>>& inputs,
    bool disable_non_trt_optimizers, const std::string& device_requested) {
  if (conversion_status_ != ConversionStatus::kUnconverted) {
    return errors::Unimplemented("The graph has already been converted.");
  }
  if (calibration_status_ == CalibrationStatus::kNeedsCalibration &&
      inputs.empty()) {
    return errors::InvalidArgument(
        "Should specify inputs because INT8 calibration is needed");
  }
  if (calibration_status_ != CalibrationStatus::kNeedsCalibration &&
      !inputs.empty()) {
    return errors::InvalidArgument(
        "Should not specify inputs because INT8 calibration is not needed");
  }

  std::string device_requested_lower = device_requested;
  std::transform(device_requested_lower.cbegin(), device_requested_lower.cend(),
                 device_requested_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (!device_requested.empty() &&
      device_requested_lower.find("gpu") == std::string::npos) {
    return errors::InvalidArgument("Specified device is not a GPU: ",
                                   device_requested);
  }
  if (!device_requested.empty() &&
      device_requested_lower.find("gpu:0") == std::string::npos) {
    LOG(INFO) << "Placing imported graph on device: " << device_requested;
    // Set nodes with unset devices to `device_requested`
    graph::SetDefaultDevice(device_requested, &input_graph_def);
  }

  MetaGraphDef meta_graph;
  meta_graph.mutable_graph_def()->CopyFrom(input_graph_def);

  ConfigProto config_proto;
  TF_RETURN_IF_ERROR(GetTrtRewriterConfig(
      conversion_params_,
      config_proto.mutable_graph_options()->mutable_rewrite_options(),
      disable_non_trt_optimizers));
  // Run TRTOptimizer Grappler pass
  TF_RETURN_IF_ERROR(RunGrappler(meta_graph, input_names_, output_names_,
                                 config_proto, &segmented_graph_def_));

  VLOG(1) << "TF-TRT conversion finished";

  TF_RETURN_IF_ERROR(SetProfileGenerationMode(&segmented_graph_def_, false));

  if (!segmented_prefix_.empty()) {
    // TODO: Remove intial model from session to reduce unnecessary memory usage
    TF_RETURN_IF_ERROR(ImportGraphDefToSession(
        session.get(), segmented_graph_def_, segmented_prefix_));
  } else {
    TF_RETURN_IF_ERROR(session->Create(segmented_graph_def_));
  }

  if (calibration_status_ == CalibrationStatus::kNeedsCalibration) {
    calibration_inputs_ = inputs;
    // Execute calibration here only if not in dynamic shape mode
    if (!conversion_params_.use_dynamic_shape) {
      TF_RETURN_IF_ERROR(ExecuteCalibration());
    }
  }

  conversion_status_ = ConversionStatus::kConverted;
  return segmented_graph_def_;
}

StatusOr<GraphDef> TrtGraphConverter::Build(
    const std::vector<std::vector<tensorflow::Tensor>>& inputs) {
  if (conversion_status_ != ConversionStatus::kConverted) {
    return errors::Unimplemented(
        "Either convert() has not been called before build() "
        "or build() is already called. It is not supported to call "
        "build() more than once.");
  }
  if (inputs.empty()) {
    return errors::InvalidArgument(
        "Method build() needs inputs to be specified in order "
        "to build TensorRT engines.");
  }

  // The TRTOptimization pass has inserted placeholder TRTEngineOps. Here we
  // trigger conversion by inferring the graph.
  VLOG(2) << "Building the model";
  bool need_collect_profiles =
      conversion_params_.use_dynamic_shape && inputs.size() > 1;
  if (need_collect_profiles) {
    std::string prefix = "TrtBuildStep";
    GraphDef profile_graph_def;
    profile_graph_def.CopyFrom(segmented_graph_def_);
    TF_RETURN_IF_ERROR(SetProfileGenerationMode(&profile_graph_def, true));
    VLOG(3) << "Importing graph with _profile_generation_mode enabled";
    TF_RETURN_IF_ERROR(
        ImportGraphDefToSession(session.get(), profile_graph_def, prefix));
    for (auto const& input : inputs) {
      TF_RETURN_IF_ERROR(RunSession(session.get(), input_names_, output_names_,
                                    input, prefix));
    }
  }

  // Run calibration if required, this would have been skipped in the convert
  // step.
  if (calibration_status_ == CalibrationStatus::kNeedsCalibration) {
    TF_RETURN_IF_ERROR(ExecuteCalibration());
  } else {
    TF_RETURN_IF_ERROR(RunSession(session.get(), input_names_, output_names_,
                                  *inputs.begin(), segmented_prefix_));
  }

  conversion_status_ = ConversionStatus::kBuilt;
  return segmented_graph_def_;
}

StatusOr<std::map<std::string, std::string>>
TrtGraphConverter::SerializeEngines(const std::string& out_dir,
                                    bool save_gpu_specific_engines) {
  if (conversion_status_ == ConversionStatus::kUnconverted) {
    return errors::Unimplemented(
        "The graph has not yet been converted with convert().");
  }
  if (calibration_status_ == CalibrationStatus::kNeedsCalibration) {
    return errors::Unimplemented(
        "A model that requires INT8 calibration has to be "
        "built before saving it. Call build() to build and "
        "calibrate the TensorRT engines.");
  }

  std::string prefix = "TRTSerialize";
  prefix += save_gpu_specific_engines ? "_t" : "_f";
  TF_RETURN_IF_ERROR(SetupSerializeTRTResourceOp(
      session.get(), save_gpu_specific_engines, prefix));

  std::map<std::string, std::string> m;
  std::string op{"TRTEngineOp"};
  for (auto& node : *(segmented_graph_def_.mutable_node())) {
    if (!op.compare(node.op())) {
      // Don't dump the same cache twice
      std::string engine_name = GetCanonicalEngineName(node.name());
      if (m.find(engine_name) != m.end()) {
        continue;
      }

      std::string filename;
      auto status =
          SerializeTRTResource(session.get(), node, out_dir, prefix, &filename);
      if (status.ok()) {
        m[engine_name] = filename;
      } else if (errors::IsNotFound(status)) {
        LOG(INFO) << "Could not find " << engine_name << " in TF-TRT cache. "
                  << "This can happen if build() is not called, "
                  << "which means TensorRT engines will be build "
                  << "and cached at runtime.";
      } else {
        return status;
      }
    }
  }
  return m;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
