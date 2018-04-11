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

using std::string;
using std::unordered_map;

class PluginTensorRT : public nvinfer1::IPlugin {
 public:
  PluginTensorRT(){};
  PluginTensorRT(const void* serialized_data, size_t length);
  // PluginTensorRT(const void* serialized_data, size_t length, size_t
  // &incremental);
  virtual string GetPluginName() = 0;
  virtual bool Finalize() = 0;

  virtual bool SetAttribute(const string& key, const void* ptr,
                            const size_t size) = 0;
  virtual bool GetAttribute(const string& key, const void* ptr,
                            size_t& size) = 0;

  void configure(const nvinfer1::Dims* inputs, int nbInputs,
                 const nvinfer1::Dims* outputs, int nbOutputs,
                 int maxBatchSize) override {
    for (int index = 0; index < nbInputs; index++) {
      nvinfer1::Dims dim;
      dim.nbDims = inputs[index].nbDims;
      for (int i = 0; i < dim.nbDims; i++) {
        dim.d[i] = inputs[index].d[i];
        dim.type[i] = inputs[index].type[i];
      }
      input_dim_list_.emplace_back(dim);
    }
    return;
  }

  virtual bool StoreAttribute(const string& key, const void* ptr,
                              const size_t size);

  virtual size_t getSerializationSize() override;
  virtual void serialize(void* buffer) override;

 protected:
  std::unordered_map<string, std::vector<char> > attr_map_;

  std::vector<nvinfer1::Dims> input_dim_list_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN
