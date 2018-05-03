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

#include "tensorflow/contrib/tensorrt/custom_plugin_examples/inc_op_kernel.h"
#include "tensorflow/contrib/tensorrt/custom_plugin_examples/inc_op_plugin.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

const char* kPluginName = "IncPluginTRT";

IncOpPlugin* CreateIncPlugin() { return new IncOpPlugin(); }

IncOpPlugin* CreateIncPluginDeserialize(const void* buffer, size_t length) {
  return new IncOpPlugin(buffer, length);
}

REGISTER_TRT_PLUGIN(kPluginName, CreateIncPluginDeserialize, CreateIncPlugin);

IncOpPlugin::IncOpPlugin() : plugin_name_(kPluginName) {}

IncOpPlugin::IncOpPlugin(const void* serialized_data, size_t length)
    : PluginTensorRT(serialized_data, length), plugin_name_(kPluginName) {
  // account for the consumed pointer.
  size_t consumed_data = PluginTensorRT::getSerializationSize();
  assert(length - consumed_data >= sizeof(float));
  const char* buffer = reinterpret_cast<const char*>(serialized_data);
  SetAttribute("inc", buffer + consumed_data, sizeof(float));
}

bool IncOpPlugin::SetAttribute(const string& key, const void* ptr,
                               const size_t size) {
  if (strcmp(key.c_str(), "inc") == 0 && size == sizeof(float)) {
    StoreAttribute(key, ptr, size);  // save the attribute to own the data;
    inc_ = *static_cast<const float*>(ptr);
    return true;
  }
  return false;
}

bool IncOpPlugin::GetAttribute(const string& key, const void** ptr,
                               size_t* size) const {
  const auto& iter = attr_map_.find(key);
  if (iter != attr_map_.end()) {
    *ptr = iter->second.data();
    *size = iter->second.size();
    return true;
  }
  return false;
}

int IncOpPlugin::enqueue(int batch_size, const void* const* inputs,
                         void** outputs, void*, cudaStream_t stream) {
  int count = 1;
  for (int i = 0; i < input_dim_list_[0].nbDims; i++) {
    count *= input_dim_list_[0].d[i];
  }
  count *= batch_size;
  const float* input = reinterpret_cast<const float*>(inputs[0]);
  float* output = reinterpret_cast<float*>(outputs[0]);
  IncrementKernel(input, inc_, output, count, stream);
  return 0;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // GOOGLE_TENSORRT
