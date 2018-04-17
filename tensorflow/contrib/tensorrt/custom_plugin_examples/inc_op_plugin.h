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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_INC_OP_PLUGIN
#define TENSORFLOW_CONTRIB_TENSORRT_INC_OP_PLUGIN

#include "tensorflow/contrib/tensorrt/plugin/trt_plugin.h"
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <iostream>

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

using std::string;
using std::unordered_map;

class IncOpPlugin : public PluginTensorRT
{
public:
  static const string plugin_name_;
  IncOpPlugin() {};
  IncOpPlugin(const void* serialized_data, size_t length);
  const string GetPluginName() override {return plugin_name_;};
  bool Finalize() override {return true;};
  bool SetAttribute(const string &key, const void *ptr, const size_t size) override;
  bool GetAttribute(const string &key, const void *ptr, size_t &size) override;

  // TRT IPlugin methods
  int getNbOutputs() const override {return 1;}

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override {
    assert(index==0);
    assert(nbInputDims==1);
    return inputs[0];
  }

  // no configure needed
  // use configure to setup input dimensions
  void configure(const nvinfer1::Dims *inputs, int nbInputs, const nvinfer1::Dims *outputs, int nbOutputs, int maxBatchSize) override {
    assert(nbInputs==1);
    PluginTensorRT::configure(inputs, nbInputs, outputs, nbOutputs, maxBatchSize);
    return;
  }

  int initialize() override {
    return 0;
  }

  void terminate() override {
    return;
  }

  size_t getWorkspaceSize(int maxBatchSize) const override {
    return 0;
  }

  int enqueue(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream) override; 

  size_t getSerializationSize() override {
    return PluginTensorRT::getSerializationSize() + sizeof(float);
  }

  void serialize(void* buffer) override {
    // serializa parent stuff
    //   OpName
    PluginTensorRT::serialize(buffer);

    // incremented buffer after parent serialization;
    buffer = static_cast<char*>(buffer) + PluginTensorRT::getSerializationSize();

    std::memcpy(buffer, &inc_, sizeof(float));
    buffer = static_cast<char*>(buffer) + sizeof(float);
    return;
  }

protected:
  float inc_;
  nvinfer1::Dims dim_;
  // std::unordered_map<string, std::vector<char> > attr_map_;
};

IncOpPlugin* CreateIncPlugin(); 
IncOpPlugin* CreateIncPluginDeserialize(const void*, size_t);
bool RegisterIncOpPlugin();
void IncrementKernel(const float* d_input, float inc, float* d_output, int count, cudaStream_t stream);


}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_INC_OP_PLUGIN
