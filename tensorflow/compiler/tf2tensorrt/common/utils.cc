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
#include "absl/strings/str_join.h"
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
namespace {

void InitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
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
#endif
