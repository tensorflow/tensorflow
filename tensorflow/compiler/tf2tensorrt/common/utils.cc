/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/errors.h"
#include "third_party/tensorrt/NvInferPlugin.h"
#endif

namespace tensorflow {
namespace tensorrt {

std::tuple<int, int, int> GetLinkedTensorRTVersion() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  return std::tuple<int, int, int>{NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                                   NV_TENSORRT_PATCH};
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

std::tuple<int, int, int> GetLoadedTensorRTVersion() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  int ver = getInferLibVersion();
  int major = ver / 1000;
  ver = ver - major * 1000;
  int minor = ver / 100;
  int patch = ver - minor * 100;
  return std::tuple<int, int, int>{major, minor, patch};
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

}  // namespace tensorrt
}  // namespace tensorflow

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {

Status GetTrtBindingIndex(const char* tensor_name, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index) {
  // If the engine has been built for K profiles, the first getNbBindings() / K
  // bindings are used by profile number 0, the following getNbBindings() / K
  // bindings are used by profile number 1 etc.
  //
  // GetBindingIndex(tensor_name) returns the binding index for the progile 0.
  // We can also consider it as a "binding_index_within_profile".
  *binding_index = cuda_engine->getBindingIndex(tensor_name);
  if (*binding_index == -1) {
    const string msg = absl::StrCat("Input node ", tensor_name, " not found");
    return errors::NotFound(msg);
  }
  int n_profiles = cuda_engine->getNbOptimizationProfiles();
  // If we have more then one optimization profile, then we need to shift the
  // binding index according to the following formula:
  // binding_index_within_engine = binding_index_within_profile +
  //                               profile_index * bindings_per_profile
  const int bindings_per_profile = cuda_engine->getNbBindings() / n_profiles;
  *binding_index = *binding_index + profile_index * bindings_per_profile;
  return Status::OK();
}

Status GetTrtBindingIndex(int network_input_index, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index) {
  const string input_name =
      absl::StrCat(IONamePrefixes::kInputPHName, network_input_index);
  return GetTrtBindingIndex(input_name.c_str(), profile_index, cuda_engine,
                            binding_index);
}

namespace {

void InitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
#if defined(PLATFORM_WINDOWS)
  LOG_WARNING_WITH_PREFIX
      << "Windows support is provided experimentally. No guarantee is made "
         "regarding functionality or engineering support. Use at your own "
         "risk.";
#endif
  LOG(INFO) << "Linked TensorRT version: "
            << absl::StrJoin(GetLinkedTensorRTVersion(), ".");
  LOG(INFO) << "Loaded TensorRT version: "
            << absl::StrJoin(GetLoadedTensorRTVersion(), ".");

  bool plugin_initialized = initLibNvInferPlugins(trt_logger, "");
  if (!plugin_initialized) {
    LOG(ERROR) << "Failed to initialize TensorRT plugins, and conversion may "
                  "fail later.";
  }

  int num_trt_plugins = 0;
  nvinfer1::IPluginCreator* const* trt_plugin_creator_list =
      getPluginRegistry()->getPluginCreatorList(&num_trt_plugins);
  if (!trt_plugin_creator_list) {
    LOG_WARNING_WITH_PREFIX << "Can not find any TensorRT plugins in registry.";
  } else {
    VLOG(1) << "Found the following " << num_trt_plugins
            << " TensorRT plugins in registry:";
    for (int i = 0; i < num_trt_plugins; ++i) {
      if (!trt_plugin_creator_list[i]) {
        LOG_WARNING_WITH_PREFIX
            << "TensorRT plugin at index " << i
            << " is not accessible (null pointer returned by "
               "getPluginCreatorList for this plugin)";
      } else {
        VLOG(1) << "  " << trt_plugin_creator_list[i]->getPluginName();
      }
    }
  }
}

}  // namespace

void MaybeInitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
  static absl::once_flag once;
  absl::call_once(once, InitializeTrtPlugins, trt_logger);
}

}  // namespace tensorrt
}  // namespace tensorflow

namespace nvinfer1 {
std::ostream& operator<<(std::ostream& os,
                         const nvinfer1::TensorFormat& format) {
  os << "nvinfer1::TensorFormat::";
  switch (format) {
    case nvinfer1::TensorFormat::kLINEAR:
      os << "kLINEAR";
      break;

    case nvinfer1::TensorFormat::kCHW2:
      os << "kCHW2";
      break;

    case nvinfer1::TensorFormat::kHWC8:
      os << "kHWC8";
      break;

    case nvinfer1::TensorFormat::kCHW4:
      os << "kCHW4";
      break;

    case nvinfer1::TensorFormat::kCHW16:
      os << "kCHW16";
      break;

    case nvinfer1::TensorFormat::kCHW32:
      os << "kCHW32";
      break;

#if IS_TRT_VERSION_GE(8, 0, 0, 0)
    case nvinfer1::TensorFormat::kDHWC8:
      os << "kDHWC8";
      break;

    case nvinfer1::TensorFormat::kCDHW32:
      os << "kCDHW32";
      break;

    case nvinfer1::TensorFormat::kHWC:
      os << "kHWC";
      break;

    case nvinfer1::TensorFormat::kDLA_LINEAR:
      os << "kDLA_LINEAR";
      break;

    case nvinfer1::TensorFormat::kDLA_HWC4:
      os << "kDLA_HWC4";
      break;

    case nvinfer1::TensorFormat::kHWC16:
      os << "kHWC16";
      break;
#endif

    default:
      os << "unknown format";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const nvinfer1::DataType& v) {
  os << "nvinfer1::DataType::";
  switch (v) {
    case nvinfer1::DataType::kFLOAT:
      os << "kFLOAT";
      break;
    case nvinfer1::DataType::kHALF:
      os << "kHalf";
      break;
    case nvinfer1::DataType::kINT8:
      os << "kINT8";
      break;
    case nvinfer1::DataType::kINT32:
      os << "kINT32";
      break;
    case nvinfer1::DataType::kBOOL:
      os << "kBOOL";
      break;
  }
  return os;
}
}  // namespace nvinfer1

#endif
