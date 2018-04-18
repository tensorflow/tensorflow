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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN
#define TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

// A wrapper class for TensorRT plugin
// User application should inherit from this class to write custom kernels.
// Allows user to insert custom op in TensorRT engine
// To register plugin in converter, user should also register custom
// PluginDeserializeFunc & PluginConstructFunc through PluginFactoryTensorRT
class PluginTensorRT : public nvinfer1::IPlugin {
 public:
  PluginTensorRT(){};
  PluginTensorRT(const void* serialized_data, size_t length);
  virtual const std::string& GetPluginName() const = 0;
  virtual bool Finalize() = 0;

  virtual bool SetAttribute(const std::string& key, const void* ptr,
                            const size_t size) = 0;
  virtual bool GetAttribute(const std::string& key, const void** ptr,
                            size_t* size) const = 0;

  void configure(const nvinfer1::Dims* inputs, int num_inputs,
                 const nvinfer1::Dims* outputs, int num_outputs,
                 int max_batch_size) override;

  virtual bool StoreAttribute(const std::string& key, const void* ptr,
                              const size_t size);

  virtual size_t getSerializationSize() override;
  virtual void serialize(void* buffer) override;

 protected:
  std::unordered_map<std::string, std::vector<char> > attr_map_;

  std::vector<nvinfer1::Dims> input_dim_list_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN
