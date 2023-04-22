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

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stacktrace.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
namespace convert {
// TODO(sami): Remove VLOG messages once the code matures
using absl::AsciiStrToUpper;
using absl::StrAppend;
using absl::StrCat;

namespace {

Status ValidateValueCase(const AttrValue* attr_value,
                         const AttrValue::ValueCase& value_case) {
  if (attr_value->value_case() != value_case) {
    return errors::InvalidArgument("AttrValue had value with type '",
                                   attr_value->value_case(), "' when '",
                                   value_case, "' was expected.");
  }
  return Status::OK();
}

template <typename T>
Status GetAttrBoolValue(T* value, const AttrValue* attr_value) {
  *value = static_cast<T>(attr_value->b());
  return Status::OK();
}

template <typename T>
Status GetAttrIntValue(T* value, const AttrValue* attr_value) {
  *value = static_cast<T>(attr_value->i());
  return Status::OK();
}

template <typename T>
Status GetAttrStringValue(T* value, const AttrValue* attr_value) {
  *value = attr_value->s();
  return Status::OK();
}

template <typename T>
Status GetAttrTrtPrecisionModeValue(T* value, const AttrValue* attr_value) {
  return TrtPrecisionModeFromName(attr_value->s(), value);
}

template <typename T>
Status GetAttrProfileStrategyValue(T* value, const AttrValue* attr_value) {
  return ProfileStrategyFromName(attr_value->s(), value);
}

template <AttrValue::ValueCase value_case, typename T, typename F>
Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, T* value, F value_getter) {
  const AttrValue* attr_value = attr_slice.FindByString(attr_name);
  if (attr_value != nullptr) {
    TF_RETURN_IF_ERROR(ValidateValueCase(attr_value, value_case));
    TF_RETURN_IF_ERROR(value_getter(value, attr_value));
    VLOG(1) << "Updated cp." << attr_name.substr(7) << ".";
  }
  return Status::OK();
}

Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, std::string* value) {
  return GetAttrValue<AttrValue::ValueCase::kS, std::string,
                      Status(std::string*, const AttrValue*)>(
      attr_slice, attr_name, value, GetAttrStringValue);
}

Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, size_t* value) {
  return GetAttrValue<AttrValue::ValueCase::kI, size_t,
                      Status(size_t*, const AttrValue*)>(
      attr_slice, attr_name, value, GetAttrIntValue);
}

Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, int* value) {
  return GetAttrValue<AttrValue::ValueCase::kI, int,
                      Status(int*, const AttrValue*)>(attr_slice, attr_name,
                                                      value, GetAttrIntValue);
}

Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, TrtPrecisionMode* value) {
  return GetAttrValue<AttrValue::ValueCase::kS, TrtPrecisionMode,
                      Status(TrtPrecisionMode*, const AttrValue*)>(
      attr_slice, attr_name, value, GetAttrTrtPrecisionModeValue);
}

Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, int64_t* value) {
  return GetAttrValue<AttrValue::ValueCase::kI, int64_t,
                      Status(int64_t*, const AttrValue*)>(
      attr_slice, attr_name, value, GetAttrIntValue);
}

Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, bool* value) {
  return GetAttrValue<AttrValue::ValueCase::kB, bool,
                      Status(bool*, const AttrValue*)>(attr_slice, attr_name,
                                                       value, GetAttrBoolValue);
}

Status GetAttrValue(const tensorflow::AttrSlice& attr_slice,
                    const std::string& attr_name, ProfileStrategy* value) {
  return GetAttrValue<AttrValue::ValueCase::kS, ProfileStrategy,
                      Status(ProfileStrategy*, const AttrValue*)>(
      attr_slice, attr_name, value, GetAttrProfileStrategyValue);
}

StatusOr<bool> ShouldConvertFunction(const grappler::GrapplerItem& item) {
  if (item.id == "tf_graph") {
    return false;
  }
  const grappler::GrapplerFunctionItem& func_item =
      tensorflow::down_cast<const grappler::GrapplerFunctionItem&>(item);
  const tensorflow::AttrSlice& attr = func_item.func_attr();
  const AttrValue* attr_value = attr.FindByString("_tftrt_convert_function");
  if (attr_value != nullptr) {
    TF_RETURN_IF_ERROR(ValidateValueCase(attr_value, AttrValue::ValueCase::kB));
    return attr_value->b();
  }
  VLOG(1) << "Attribute _tftrt_convert_function was not found.";
  return false;
}

StatusOr<bool> CheckForFunctionConversionAttribute(
    const tensorflow::AttrSlice& attr) {
  const AttrValue* attr_value = attr.FindByString("_tftrt_convert_function");
  if (attr_value != nullptr) {
    TF_RETURN_IF_ERROR(ValidateValueCase(attr_value, AttrValue::ValueCase::kB));
    return attr_value->b();
  } else {
    VLOG(1) << "Attribute _tftrt_convert_function was not found.";
  }
  return false;
}

// Converts function conversion attributes to conversion parameters.
Status UpdateFunctionSpecificConversionParams(
    ConversionParams& cp, const tensorflow::AttrSlice& attr) {
  TF_RETURN_IF_ERROR(
      GetAttrValue(attr, "_tftrt_trt_logger_name", &cp.trt_logger_name));
  TF_RETURN_IF_ERROR(
      GetAttrValue(attr, "_tftrt_max_batch_size", &cp.max_batch_size));
  TF_RETURN_IF_ERROR(GetAttrValue(attr, "_tftrt_max_workspace_size_bytes",
                                  &cp.max_workspace_size_bytes));
  TF_RETURN_IF_ERROR(
      GetAttrValue(attr, "_tftrt_precision_mode", &cp.precision_mode));
  TF_RETURN_IF_ERROR(GetAttrValue(attr, "_tftrt_minimum_segment_size",
                                  &cp.minimum_segment_size));
  TF_RETURN_IF_ERROR(GetAttrValue(attr, "_tftrt_is_dyn_op", &cp.is_dyn_op));
  TF_RETURN_IF_ERROR(
      GetAttrValue(attr, "_tftrt_max_cached_engines", &cp.max_cached_engines));
  TF_RETURN_IF_ERROR(
      GetAttrValue(attr, "_tftrt_use_calibration", &cp.use_calibration));
  TF_RETURN_IF_ERROR(
      GetAttrValue(attr, "_tftrt_use_implicit_batch", &cp.use_implicit_batch));
  TF_RETURN_IF_ERROR(
      GetAttrValue(attr, "_tftrt_profile_strategy", &cp.profile_strategy));
  TF_RETURN_IF_ERROR(GetAttrValue(attr, "_tftrt_allow_build_at_runtime",
                                  &cp.allow_build_at_runtime));
  return Status::OK();
}

}  // namespace

Status TRTOptimizationPass::Init(
    const RewriterConfig_CustomGraphOptimizer* config) {
  VLOG(1) << "Called INIT for " << name_ << " with config = " << config;
  if (config == nullptr) {
    return Status::OK();
  }
  VLOG(1) << "config = " << config->DebugString();
  const auto params = config->parameter_map();
  if (params.count("minimum_segment_size")) {
    minimum_segment_size_ = params.at("minimum_segment_size").i();
  }
  if (params.count("max_batch_size")) {
    maximum_batch_size_ = params.at("max_batch_size").i();
  }
  if (params.count("is_dynamic_op")) {
    is_dynamic_op_ = params.at("is_dynamic_op").b();
  }
  if (params.count("maximum_cached_engines")) {
    max_cached_batches_ = params.at("maximum_cached_engines").i();
  }
  if (params.count("max_workspace_size_bytes")) {
    max_workspace_size_bytes_ = params.at("max_workspace_size_bytes").i();
  }
  if (params.count("precision_mode")) {
    TF_RETURN_IF_ERROR(TrtPrecisionModeFromName(
        AsciiStrToUpper(params.at("precision_mode").s()), &precision_mode_));
  }
  if (params.count("use_calibration")) {
    use_calibration_ = params.at("use_calibration").b();
  }
  if (params.count("trt_logger")) {
    trt_logger_name_ = params.at("trt_logger").s();
  }
  if (params.count("allow_build_at_runtime")) {
    allow_build_at_runtime_ = params.at("allow_build_at_runtime").b();
  }
  if (params.count("use_implicit_batch")) {
    use_implicit_batch_ = params.at("use_implicit_batch").b();
  }
  if (params.count("profile_strategy")) {
    TF_RETURN_IF_ERROR(ProfileStrategyFromName(
        params.at("profile_strategy").s(), &profile_strategy_));
  }
  return Status::OK();
}

void TRTOptimizationPass::PrintDebugInfo(grappler::Cluster* cluster,
                                         const grappler::GrapplerItem& item) {
  LOG(INFO) << "Cluster = " << cluster;
  string offset("  ");
  string offset2 = StrCat(offset, offset);
  string offset3 = StrCat(offset2, offset);
  string offset4 = StrCat(offset2, offset2);

  if (cluster) {
    LOG(INFO) << offset << "type             = " << cluster->type();
    LOG(INFO) << offset << "num warmup steps = " << cluster->NumWarmupSteps();
    const auto dev_names = cluster->GetDeviceNames();
    if (!dev_names.empty()) {
      LOG(INFO) << offset << " Device names:";
      for (const auto& s : dev_names) {
        LOG(INFO) << offset2 << s;
      }
    }
    std::unordered_map<string, uint64> peak_mem;
    auto status = cluster->GetPeakMemoryUsage(&peak_mem);
    if (status == Status::OK()) {
      LOG(INFO) << offset << "Peak Memory Usage :";
      for (const auto& s : peak_mem) {
        LOG(INFO) << offset2 << s.first << " = " << s.second;
      }
    }

    const auto dev_props = cluster->GetDevices();
    if (!dev_props.empty()) {
      LOG(INFO) << offset << "Device properties:";
      for (const auto& k : dev_props) {
        LOG(INFO) << offset2 << k.first;
        const auto& dt = k.second;
        LOG(INFO) << offset3 << "type          = " << dt.type();
        LOG(INFO) << offset3 << "vendor        = " << dt.vendor();
        LOG(INFO) << offset3 << "model         = " << dt.model();
        LOG(INFO) << offset3 << "frequency     = " << dt.frequency();
        LOG(INFO) << offset3 << "num cores     = " << dt.num_cores();
        LOG(INFO) << offset3 << "num registers = " << dt.num_registers();
        LOG(INFO) << offset3 << "L1 cache size = " << dt.l1_cache_size();
        LOG(INFO) << offset3 << "L2 cache size = " << dt.l2_cache_size();
        LOG(INFO) << offset3 << "L3 cache size = " << dt.l3_cache_size();
        LOG(INFO) << offset3 << "SHMem per SMP = "
                  << dt.shared_memory_size_per_multiprocessor();
        LOG(INFO) << offset3 << "memory size   = " << dt.memory_size();
        LOG(INFO) << offset3 << "bandwidth     = " << dt.bandwidth();
        if (dt.environment_size()) {
          LOG(INFO) << offset3 << "environment   :";
          for (const auto& e : dt.environment()) {
            LOG(INFO) << offset4 << e.first << " = " << e.second;
          }
        }
      }
    }

    if (cluster->GetDeviceSet()) {
      for (const auto dev : cluster->GetDeviceSet()->devices()) {
        LOG(INFO) << "Device name= " << dev->name() << "Pased name= "
                  << DeviceNameUtils::ParsedNameToString(dev->parsed_name());
      }
    }
  }

  LOG(INFO) << "item: " << item.id;
  if (!item.feed.empty()) {
    LOG(INFO) << offset << "Feeds  :";
    for (const auto& f : item.feed) {
      const auto& shape = f.second.shape();
      LOG(INFO) << offset2 << f.first << " = shaped " << shape.DebugString();
    }
  } else {
    LOG(INFO) << offset << "No Feeds";
  }
  if (!item.fetch.empty()) {
    LOG(INFO) << offset << "Fetches  :";
    for (const auto& f : item.fetch) {
      LOG(INFO) << offset2 << f;
    }
  } else {
    LOG(INFO) << offset << "No Fetches";
  }

  if (!item.init_ops.empty()) {
    LOG(INFO) << offset << "init ops  :";
    for (const auto& f : item.init_ops) {
      LOG(INFO) << offset2 << f;
    }
  } else {
    LOG(INFO) << offset << "No init ops";
  }
  LOG(INFO) << "Save Op = " << item.save_op;
  LOG(INFO) << "Restore Op = " << item.restore_op;
  LOG(INFO) << "save_restore_loc_tensor = " << item.save_restore_loc_tensor;
  if (!item.keep_ops.empty()) {
    LOG(INFO) << offset << "keep ops  :";
    for (const auto& f : item.keep_ops) {
      LOG(INFO) << offset2 << f;
    }
  } else {
    LOG(INFO) << offset << "No keep ops";
  }
}

Status TRTOptimizationPass::Optimize(grappler::Cluster* cluster,
                                     const grappler::GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  VLOG(1) << "Called TRTOptimization Pass " << name_
          << " on a grappler item with id=" << item.id;
  TF_ASSIGN_OR_RETURN(bool do_function_conversion, ShouldConvertFunction(item));
  if (minimum_segment_size_ == -1 ||
      (item.id != "tf_graph" && !do_function_conversion)) {
    VLOG(1) << "Not optimizing this grappler item: " << item.id;
    *optimized_graph = item.graph;
    return Status::OK();
  }
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << CurrentStackTrace();
    PrintDebugInfo(cluster, item);
  }

  if (use_calibration_ && precision_mode_ != TrtPrecisionMode::INT8) {
    VLOG(1) << "Calibration with FP32 or FP16 is not implemented. "
            << "Falling back to use_calibration = False."
            << "Note that the default value of use_calibration is True.";
    use_calibration_ = false;
  }

  std::vector<string> nodes_to_preserve;
  for (const auto& n : item.NodesToPreserve()) {
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

  ConversionParams cp;
  cp.grappler_item = &item;
  cp.output_names = &nodes_to_preserve;
  cp.trt_logger_name = trt_logger_name_;
  cp.max_batch_size = maximum_batch_size_;
  cp.max_workspace_size_bytes = max_workspace_size_bytes_;
  cp.output_graph_def = optimized_graph;
  cp.precision_mode = precision_mode_;
  cp.minimum_segment_size = minimum_segment_size_;
  cp.cluster = cluster;
  cp.is_dyn_op = is_dynamic_op_;
  cp.max_cached_engines = max_cached_batches_;
  cp.use_calibration = use_calibration_;
  cp.use_implicit_batch = use_implicit_batch_;
  cp.profile_strategy = profile_strategy_;
  cp.allow_build_at_runtime = allow_build_at_runtime_;

  if (item.id != "tf_graph" && do_function_conversion) {
    const grappler::GrapplerFunctionItem& func_item =
        tensorflow::down_cast<const grappler::GrapplerFunctionItem&>(item);
    TF_RETURN_IF_ERROR(
        UpdateFunctionSpecificConversionParams(cp, func_item.func_attr()));
  }

  auto status = ConvertAfterShapes(cp);
  VLOG(1) << "Returning from " << name_;
  return status;
}

class VerboseCustomGraphOptimizerRegistrar
    : public grappler::CustomGraphOptimizerRegistrar {
 public:
  VerboseCustomGraphOptimizerRegistrar(
      const grappler::CustomGraphOptimizerRegistry::Creator& cr,
      const string& name)
      : grappler::CustomGraphOptimizerRegistrar(cr, name) {
    VLOG(1) << "Constructing a CustomOptimizationPass registration object for "
            << name;
  }
};

static VerboseCustomGraphOptimizerRegistrar TRTOptimizationPass_Registrar(
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
