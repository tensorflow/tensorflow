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

#include <string>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace test {

class StubPlugin : public PluginTensorRT {
 public:
  static const std::string plugin_name_;
  StubPlugin(){};
  StubPlugin(const void* serialized_data, size_t length)
      : PluginTensorRT(serialized_data, length){};
  const std::string& GetPluginName() override { return plugin_name_; };
  virtual bool Finalize() { return true; };
  virtual bool SetAttribute(const std::string& key, const void* ptr,
                            const size_t size) {
    return true;
  };
  virtual bool GetAttribute(const std::string& key, const void* ptr,
                            size_t& size) {
    return true;
  };
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override {
    return inputs[0];
  }
  int initialize() override { return 0; }
  void terminate() override {}
  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }
  int enqueue(int batch_size, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override {
    return 0;
  }
};

const std::string StubPlugin::plugin_name_ = "StubPlugin";

StubPlugin* CreateStubPlugin() { return new StubPlugin(); }

StubPlugin* CreateStubPluginDeserialize(const void* serialized_data,
                                        size_t length) {
  return new StubPlugin(serialized_data, length);
}

class PluginTest : public ::testing::Test {
 public:
  bool RegisterStubPlugin() {
    if (PluginFactoryTensorRT::GetInstance()->IsPlugin(
            StubPlugin::plugin_name_))
      return true;
    return PluginFactoryTensorRT::GetInstance()->RegisterPlugin(
        StubPlugin::plugin_name_, CreateStubPluginDeserialize,
        CreateStubPlugin);
  }
};

TEST_F(PluginTest, Registration) {
  EXPECT_FALSE(
      PluginFactoryTensorRT::GetInstance()->IsPlugin(StubPlugin::plugin_name_));
  EXPECT_TRUE(RegisterStubPlugin());

  ASSERT_TRUE(
      PluginFactoryTensorRT::GetInstance()->IsPlugin(StubPlugin::plugin_name_));
}

TEST_F(PluginTest, CreationDeletion) {
  EXPECT_TRUE(RegisterStubPlugin());
  ASSERT_TRUE(
      PluginFactoryTensorRT::GetInstance()->IsPlugin(StubPlugin::plugin_name_));

  PluginFactoryTensorRT::GetInstance()->DestroyPlugins();
  ASSERT_TRUE(PluginFactoryTensorRT::GetInstance()->CreatePlugin(
      StubPlugin::plugin_name_));
  ASSERT_EQ(1, PluginFactoryTensorRT::GetInstance()->CountOwnedPlugins());
  PluginFactoryTensorRT::GetInstance()->DestroyPlugins();
  ASSERT_EQ(0, PluginFactoryTensorRT::GetInstance()->CountOwnedPlugins());
}

}  // namespace test
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
