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

#include <cassert>
#include <cstring>

#include "tensorflow/contrib/tensorrt/plugin/trt_plugin.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

class IncOpPlugin : public PluginTensorRT {
 public:
  IncOpPlugin();

  IncOpPlugin(const void* serialized_data, size_t length);

  const string& GetPluginName() const override { return plugin_name_; };

  bool Finalize() override { return true; };

  bool SetAttribute(const string& key, const void* ptr,
                    const size_t size) override;

  bool GetAttribute(const string& key, const void** ptr,
                    size_t* size) const override;

  int getNbOutputs() const override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int num_input_dims) override {
    assert(index == 0);
    assert(num_input_dims == 1);
    return inputs[0];
  }

  // use configure to setup input dimensions
  void configure(const nvinfer1::Dims* inputs, int num_inputs,
                 const nvinfer1::Dims* outputs, int num_outputs,
                 int max_batch_size) override {
    assert(num_inputs == 1);
    PluginTensorRT::configure(inputs, num_inputs, outputs, num_outputs,
                              max_batch_size);
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int max_batch_size) const override { return 0; }

  int enqueue(int batch_size, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override;

  size_t getSerializationSize() override {
    return PluginTensorRT::getSerializationSize() + sizeof(float);
  }

  void serialize(void* buffer) override {
    // Serialize parent data.
    PluginTensorRT::serialize(buffer);
    // Incremented buffer after parent serialization.
    buffer =
        static_cast<char*>(buffer) + PluginTensorRT::getSerializationSize();
    std::memcpy(buffer, &inc_, sizeof(float));
    buffer = static_cast<char*>(buffer) + sizeof(float);
  }

 protected:
  float inc_;
  nvinfer1::Dims dim_;

 private:
  const string plugin_name_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_INC_OP_PLUGIN
