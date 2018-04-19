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

#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

PluginTensorRT* PluginFactoryTensorRT::createPlugin(const char* layer_name,
                                                    const void* serial_data,
                                                    size_t serial_length) {
  size_t parsed_byte = 0;
  // extract op_name from serial_data
  string encoded_op_name =
      ExtractOpName(serial_data, serial_length, &parsed_byte);

  if (!IsPlugin(encoded_op_name)) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(instance_m_);
  auto plugin_ptr =
      plugin_registry_[encoded_op_name].first(serial_data, serial_length);
  owned_plugins_.emplace_back(plugin_ptr);

  return plugin_ptr;
}

PluginTensorRT* PluginFactoryTensorRT::CreatePlugin(const string& op_name) {
  if (!IsPlugin(op_name)) return nullptr;

  std::lock_guard<std::mutex> lock(instance_m_);
  auto plugin_ptr = plugin_registry_[op_name].second();
  owned_plugins_.emplace_back(plugin_ptr);

  return plugin_ptr;
}

bool PluginFactoryTensorRT::RegisterPlugin(
    const string& op_name, PluginDeserializeFunc deserialize_func,
    PluginConstructFunc construct_func) {
  if (IsPlugin(op_name)) return false;

  std::lock_guard<std::mutex> lock(instance_m_);
  auto ret = plugin_registry_.emplace(
      op_name, std::make_pair(deserialize_func, construct_func));

  return ret.second;
}

void PluginFactoryTensorRT::DestroyPlugins() {
  std::lock_guard<std::mutex> lock(instance_m_);
  for (auto& owned_plugin_ptr : owned_plugins_) {
    owned_plugin_ptr.release();
  }
  owned_plugins_.clear();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // GOOGLE_TENSORRT
