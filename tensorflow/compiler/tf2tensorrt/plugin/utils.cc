/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <mutex>
#include <stack>
#include <unordered_set>
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/plugin/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "absl/base/call_once.h"
#include "absl/strings/str_join.h"
#include "third_party/tensorrt/NvInferPlugin.h"
#include "third_party/tensorrt/plugin/efficientNMSPlugin/tftrt/efficientNMSExplicitTFTRTPlugin.h"
#include "third_party/tensorrt/plugin/efficientNMSPlugin/tftrt/efficientNMSImplicitTFTRTPlugin.h"
#endif

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {

namespace {

class PluginCreatorRegistry {
 public:
  static PluginCreatorRegistry& GetInstance() {
    static PluginCreatorRegistry instance;
    return instance;
  }

  template <typename CreatorType>
  static void RegisterPlugin() {
    PluginCreatorRegistry::GetInstance().AddPluginCreator<CreatorType>();
  }

  template <typename CreatorType>
  void AddPluginCreator() {
    std::lock_guard<std::mutex> lock(registry_lock_);
    std::unique_ptr<CreatorType> plugin_creator{new CreatorType{}};
    plugin_creator->setPluginNamespace("");
    std::string plugin_type =
        std::string{plugin_creator->getPluginName()} + " version " +
        std::string{plugin_creator->getPluginVersion()};
    if (registry_list_.find(plugin_type) == registry_list_.end()) {
      bool status = getPluginRegistry()->registerCreator(*plugin_creator, "");
      if (status) {
        registry_.push(std::move(plugin_creator));
        registry_list_.insert(plugin_type);
        VLOG(1) << "Registered plugin creator - " + plugin_type;
      } else {
        LOG(ERROR) << "Could not register plugin creator - " + plugin_type;
      }
    } else {
      VLOG(1) << "Plugin creator already registered - " + plugin_type;
    }
  }

  ~PluginCreatorRegistry() {
    std::lock_guard<std::mutex> lock(registry_lock_);
    while (!registry_.empty()) {
      registry_.pop();
    }
    registry_list_.clear();
  }

 private:
  PluginCreatorRegistry() {}

  std::mutex registry_lock_;
  std::stack<std::unique_ptr<nvinfer1::IPluginCreator>> registry_;
  std::unordered_set<std::string> registry_list_;

 public:
  PluginCreatorRegistry(PluginCreatorRegistry const&) = delete;
  void operator=(PluginCreatorRegistry const&) = delete;
};


void InitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
  LOG(INFO) << "Linked TensorRT version: "
            << absl::StrJoin(GetLinkedTensorRTVersion(), ".");
  LOG(INFO) << "Loaded TensorRT version: "
            << absl::StrJoin(GetLoadedTensorRTVersion(), ".");

  PluginCreatorRegistry::RegisterPlugin<EfficientNMSExplicitTFTRTPluginCreator>();
  PluginCreatorRegistry::RegisterPlugin<EfficientNMSImplicitTFTRTPluginCreator>();

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

nvinfer1::IPluginCreator* GetPluginCreator(const char* name,
                                           const char* version) {
  nvinfer1::IPluginCreator* creator =
      getPluginRegistry()->getPluginCreator(name, version, "");
  if (creator) {
    return creator;
  }
  return nullptr;
}

void MaybeInitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
  static absl::once_flag once;
  absl::call_once(once, InitializeTrtPlugins, trt_logger);
}

}  // namespace tensorrt
}  // namespace tensorflow
#endif
