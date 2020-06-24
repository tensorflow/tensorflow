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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_TRT_PLUGIN_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_TRT_PLUGIN_H_

#include <vector>

#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

extern const char* kTfTrtPluginVersion;
extern const char* kTfTrtPluginNamespace;

#if NV_TENSORRT_MAJOR > 5 || (NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR >= 1)
// A wrapper class for TensorRT plugin. User application should inherit from
// this class to write custom kernels.
class TrtPlugin : public nvinfer1::IPluginV2Ext {
 public:
  TrtPlugin() { setPluginNamespace(kTfTrtPluginNamespace); }

  TrtPlugin(const void* serialized_data, size_t length) {}

  TrtPlugin(const TrtPlugin& rhs) : namespace_(rhs.namespace_) {}

  int initialize() override { return 0; }

  void terminate() override {}

  void destroy() override { delete this; }

  void setPluginNamespace(const char* plugin_namespace) override {
    namespace_ = plugin_namespace;
  }

  const char* getPluginNamespace() const override { return namespace_.c_str(); }

 protected:
  template <typename T>
  void WriteToBuffer(const T& val, char** buffer) const {
    *reinterpret_cast<T*>(*buffer) = val;
    *buffer += sizeof(T);
  }

  template <typename T>
  T ReadFromBuffer(const char** buffer) {
    T val = *reinterpret_cast<const T*>(*buffer);
    *buffer += sizeof(T);
    return val;
  }

 private:
  std::string namespace_;
};
#endif

template <typename T>
class TrtPluginRegistrar {
 public:
  TrtPluginRegistrar() {
    getPluginRegistry()->registerCreator(creator, kTfTrtPluginNamespace);
  }

 private:
  T creator;
};

#define REGISTER_TFTRT_PLUGIN(name)                       \
  static ::tensorflow::tensorrt::TrtPluginRegistrar<name> \
      plugin_registrar_##name {}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_TRT_PLUGIN_H_
