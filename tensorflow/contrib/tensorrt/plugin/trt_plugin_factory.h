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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN_FACTORY
#define TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN_FACTORY

#include <memory>
#include <mutex>
#include <unordered_map>

#include "tensorflow/contrib/tensorrt/plugin/trt_plugin.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

class PluginFactoryTensorRT : public nvinfer1::IPluginFactory {
 public:
  static PluginFactoryTensorRT* GetInstance();

  // Deserialization method
  PluginTensorRT* createPlugin(const char* layer_name, const void* serial_data,
                               size_t serial_length) override;

  // Plugin construction, PluginFactoryTensorRT owns the plugin.
  PluginTensorRT* CreatePlugin(const string& op_name);

  bool RegisterPlugin(const string& op_name,
                      PluginDeserializeFunc deserialize_func,
                      PluginConstructFunc construct_func);

  bool IsPlugin(const string& op_name) {
    return plugin_registry_.find(op_name) != plugin_registry_.end();
  }

  size_t CountOwnedPlugins() { return owned_plugins_.size(); }

  void DestroyPlugins();

 protected:
  std::unordered_map<string,
                     std::pair<PluginDeserializeFunc, PluginConstructFunc>>
      plugin_registry_;

  // TODO(jie): Owned plugin should be associated with different sessions;
  //            should really hand ownership of plugins to resource management;
  std::vector<std::unique_ptr<PluginTensorRT>> owned_plugins_;
  std::mutex instance_m_;
};

class TrtPluginRegistrar {
 public:
  TrtPluginRegistrar(const string& name,
                     PluginDeserializeFunc deserialize_func,
                     PluginConstructFunc construct_func) {
    auto factory = PluginFactoryTensorRT::GetInstance();
    QCHECK(factory->RegisterPlugin(name, deserialize_func, construct_func))
        << "Failed to register plugin: " << name;
  }
};

#define REGISTER_TRT_PLUGIN(name, deserialize_func, construct_func) \
    REGISTER_TRT_PLUGIN_UNIQ_HELPER(                                \
        __COUNTER__, name, deserialize_func, construct_func)
#define REGISTER_TRT_PLUGIN_UNIQ_HELPER(         \
    ctr, name, deserialize_func, construct_func) \
    REGISTER_TRT_PLUGIN_UNIQ(ctr, name, deserialize_func, construct_func)
#define REGISTER_TRT_PLUGIN_UNIQ(ctr, name, deserialize_func, construct_func) \
    static ::tensorflow::tensorrt::TrtPluginRegistrar                         \
        trt_plugin_registrar##ctr TF_ATTRIBUTE_UNUSED =                       \
            ::tensorflow::tensorrt::TrtPluginRegistrar(                       \
                name, deserialize_func, construct_func)

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN_FACTORY
