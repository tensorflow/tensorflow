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

#include "tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.h"

#include <memory>

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
namespace convert {
using absl::AsciiStrToUpper;
using absl::StrAppend;
using absl::StrCat;

namespace {

bool ShouldUseExplicitPrecision(const GraphDef& gdef) {
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {
    return false;
  }
  return absl::c_any_of(gdef.node(), [](const auto& node) {
    return (absl::c_find(kExplicitQuantizationOpNames, node.op()) !=
            kExplicitQuantizationOpNames.end());
  });
}

StatusOr<bool> ShouldConvertFunction(const grappler::GrapplerItem& item) {
  if (item.id == "tf_graph") {
    return false;
  }
  const auto& func_item =
      tensorflow::down_cast<const grappler::GrapplerFunctionItem&>(item);
  const AttrSlice& attr = func_item.func_attr();
  const AttrValue* attr_value = attr.FindByString("_tftrt_convert_function");
  if (attr_value != nullptr) {
    bool result = false;
    TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_convert_function", &result));
    return result;
  }
  VLOG(1) << "Attribute _tftrt_convert_function was not found.";
  return false;
}

// Converts function conversion attributes to conversion parameters.
Status UpdateFunctionSpecificConversionParams(
    TRTOptimizationPass::ConversionParams& cp,
    const tensorflow::AttrSlice& attr) {
  auto get_size_attr = [](const AttrSlice& attr, absl::string_view name,
                          size_t* dst) -> Status {
    int tmp = 0;
    TF_RETURN_IF_ERROR(GetNodeAttr(attr, name, &tmp));
    *dst = static_cast<size_t>(tmp);
    return OkStatus();
  };

  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_trt_logger_name", &cp.trt_logger_name));
  TF_RETURN_IF_ERROR(
      get_size_attr(attr, "_tftrt_max_batch_size", &cp.max_batch_size));
  TF_RETURN_IF_ERROR(get_size_attr(attr, "_tftrt_max_workspace_size_bytes",
                                   &cp.max_workspace_size_bytes));
  std::string precision_mode;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_precision_mode", &precision_mode));
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeFromName(precision_mode, &cp.precision_mode));
  TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_minimum_segment_size",
                                 &cp.minimum_segment_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_is_dyn_op", &cp.is_dynamic_op));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_max_cached_engines", &cp.max_cached_engines));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_use_calibration", &cp.use_calibration));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_use_implicit_batch", &cp.use_implicit_batch));
  std::string profile_strategy;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(attr, "_tftrt_profile_strategy", &profile_strategy));
  TF_RETURN_IF_ERROR(
      ProfileStrategyFromName(profile_strategy, &cp.profile_strategy));
  TF_RETURN_IF_ERROR(GetNodeAttr(attr, "_tftrt_allow_build_at_runtime",
                                 &cp.allow_build_at_runtime));
  return OkStatus();
}
}  // namespace

Status TRTOptimizationPass::Init(
    const RewriterConfig_CustomGraphOptimizer* config) {
  if (config == nullptr) {
    return OkStatus();
  }
  const auto params = config->parameter_map();
  if (params.count("minimum_segment_size")) {
    params_.minimum_segment_size = params.at("minimum_segment_size").i();
  }
  if (params.count("max_batch_size")) {
    params_.max_batch_size = params.at("max_batch_size").i();
  }
  if (params.count("is_dynamic_op")) {
    params_.is_dynamic_op = params.at("is_dynamic_op").b();
  }
  if (params.count("maximum_cached_engines")) {
    params_.max_cached_engines = params.at("maximum_cached_engines").i();
  }
  if (params.count("max_workspace_size_bytes")) {
    params_.max_workspace_size_bytes =
        params.at("max_workspace_size_bytes").i();
  }
  if (params.count("precision_mode")) {
    TF_RETURN_IF_ERROR(TrtPrecisionModeFromName(
        AsciiStrToUpper(params.at("precision_mode").s()),
        &params_.precision_mode));
  }
  if (params.count("use_calibration")) {
    params_.use_calibration = params.at("use_calibration").b();
  }
  if (params.count("trt_logger")) {
    params_.trt_logger_name = params.at("trt_logger").s();
  }
  if (params.count("allow_build_at_runtime")) {
    params_.allow_build_at_runtime = params.at("allow_build_at_runtime").b();
  }
  if (params.count("use_implicit_batch")) {
    params_.use_implicit_batch = params.at("use_implicit_batch").b();
  }
  if (params.count("profile_strategy")) {
    TF_RETURN_IF_ERROR(ProfileStrategyFromName(
        params.at("profile_strategy").s(), &params_.profile_strategy));
  }
  return OkStatus();
}

static bool ExplicitPrecisionModePolicy() {
  return IS_TRT_VERSION_GE(8, 0, 0, 0);
}

Status TRTOptimizationPass::Optimize(grappler::Cluster* cluster,
                                     const grappler::GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  VLOG(1) << "Called TRTOptimization Pass " << name_
          << " on a grappler item with id=" << item.id;
  TF_ASSIGN_OR_RETURN(bool do_function_conversion, ShouldConvertFunction(item));
  // Optimizing the main graph(identified with `item.id == "tf_graph"`) with
  // `minimim_segment_size == -1` indicates skipping main graph conversion.
  if ((params_.minimum_segment_size == -1 && item.id == "tf_graph") ||
      (item.id != "tf_graph" && !do_function_conversion)) {
    VLOG(1) << "Not optimizing this grappler item: " << item.id;
    *optimized_graph = item.graph;
    return OkStatus();
  }

  if (params_.use_calibration &&
      params_.precision_mode != TrtPrecisionMode::INT8) {
    LOG(WARNING) << "Calibration with FP32 or FP16 is not implemented. "
                 << "Falling back to use_calibration = False."
                 << "Note that the default value of use_calibration is True.";
    params_.use_calibration = false;
  }

  params_.use_explicit_precision = ShouldUseExplicitPrecision(item.graph);
  if (params_.use_explicit_precision) {
    LOG(INFO) << "[TF-TRT] Using explicit QDQ mode";
    if (params_.precision_mode != TrtPrecisionMode::INT8 ||
        params_.use_calibration) {
      LOG(WARNING)
          << "Explicit precision mode with calibration or FP32/FP16 mode is "
             "not supported."
          << " Setting precision mode to INT8 and calibration to false.";
      params_.precision_mode = TrtPrecisionMode::INT8;
      params_.use_calibration = false;
    }
  }

  // Create a copy of the graph to optimize.
  grappler::GrapplerItem optimized_item(item);

  std::vector<string> nodes_to_preserve;
  const auto& old_nodes_to_preserve = item.NodesToPreserve();
  nodes_to_preserve.reserve(old_nodes_to_preserve.size());
  for (const auto& n : old_nodes_to_preserve) {
    auto tokens = str_util::Split(n, ":");
    string s = tokens.at(0);
    for (int i = 1; i < tokens.size() - 1; ++i) {
      StrAppend(&s, ":", tokens.at(i));
    }
    int dumm_port = -1;
    // If the last token is not an integer, it must be part of the name.
    // Otherwise it is port number.
    if (tokens.size() > 1 &&
        !strings::safe_strto32(tokens.back(), &dumm_port)) {  // non-absl ok
      StrAppend(&s, ":", tokens.back());
    }
    nodes_to_preserve.push_back(s);
  }

  if (item.id != "tf_graph" && do_function_conversion) {
    const grappler::GrapplerFunctionItem& func_item =
        tensorflow::down_cast<const grappler::GrapplerFunctionItem&>(item);
    TF_RETURN_IF_ERROR(
        UpdateFunctionSpecificConversionParams(params_, func_item.func_attr()));
  }

  return ConvertGraph(params_, optimized_item, nodes_to_preserve, cluster,
                      optimized_graph);
}

static grappler::CustomGraphOptimizerRegistrar TRTOptimizationPass_Registrar(
    []() {
      VLOG(1)
          << "Instantiating CustomOptimizationPass object TensorRTOptimizer";
      return new TRTOptimizationPass("TensorRTOptimizer");
    },
    ("TensorRTOptimizer"));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
