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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include <functional>
#include <numeric>
#include <stack>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferPlugin.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"
#include "third_party/tensorrt/plugin/efficientNMSPlugin/efficientNMSPlugin.h"

namespace tensorflow {
namespace tensorrt {

class ScopedWeights {
 public:
  ScopedWeights(float value) : value_(value) {
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = &value_;
    w.count = 1;
  }
  const nvinfer1::Weights& get() { return w; }

 private:
  float value_;
  nvinfer1::Weights w;
};

class ScopedShapedWeights {
 public:
  ScopedShapedWeights(nvinfer1::Dims dims, float value)
      : dims_(dims),
        value_(std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                               std::multiplies<>()),
               value) {
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = value_.data();
    w.count = value_.size();
  }

  nvinfer1::Dims dims_;
  std::vector<float> value_;
  nvinfer1::Weights w;
};

const char* kInputTensor1 = "input1";
const char* kInputTensor2 = "input2";
const char* kOutputTensor1 = "output";
const char* kOutputTensor2 = "output-nms";

std::unique_ptr<nvinfer1::plugin::EfficientNMSPluginCreator>
MakeNMSPluginCreator(const std::string& plugin_namespace = "tftrt") {
  auto pluginCreator =
      std::make_unique<nvinfer1::plugin::EfficientNMSPluginCreator>();
  pluginCreator->setPluginNamespace(plugin_namespace.c_str());
  std::string pluginType = std::string{pluginCreator->getPluginNamespace()} +
                           "::" + std::string{pluginCreator->getPluginName()} +
                           " version " +
                           std::string{pluginCreator->getPluginVersion()};
  VLOG(0) << "Created plugin type " << pluginType;
  return pluginCreator;
}

struct PluginDeleter {
  void operator()(nvinfer1::IPluginV2* t);
};

void PluginDeleter::operator()(nvinfer1::IPluginV2* t) { t->destroy(); }

std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter> createPlugin(
    const std::string& name, nvinfer1::IPluginCreator* pluginCreator,
    const std::vector<nvinfer1::PluginField>& pluginFields) {
  if (!pluginCreator) {
    return nullptr;
  }
  nvinfer1::PluginFieldCollection fc;
  fc.nbFields = pluginFields.size();
  fc.fields = pluginFields.data();
  return std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter>{
      pluginCreator->createPlugin(name.c_str(), &fc)};
}

// Creates a network to compute x+y.
TrtUniquePtrType<nvinfer1::IHostMemory> CreateSerializedEngine() {
  Logger& logger = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(
          1U << static_cast<uint32_t>(
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  // Add the input.
  auto input1 = network->addInput(kInputTensor1, nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{1, 1, 1, 1});
  auto input2 = network->addInput(kInputTensor2, nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{1, 1, 1, 1});
  EXPECT_NE(input1, nullptr);
  EXPECT_NE(input2, nullptr);
  // Add an ILayer layer.
  auto layer = network->addElementWise(*input1, *input2,
                                       nvinfer1::ElementWiseOperation::kSUM);
  EXPECT_NE(layer, nullptr);

  // Add an efficient nms plugin.
  ScopedShapedWeights boxes_weights(nvinfer1::Dims3(1, 10, 4), 0.0f);
  ScopedShapedWeights scores_weights(nvinfer1::Dims3(1, 10, 10), 0.0f);
  nvinfer1::IConstantLayer* boxes =
      network->addConstant(boxes_weights.dims_, boxes_weights.w);
  nvinfer1::IConstantLayer* scores =
      network->addConstant(scores_weights.dims_, scores_weights.w);

  std::array<nvinfer1::ITensor*, 2> nms_inputs = {boxes->getOutput(0),
                                                  scores->getOutput(0)};

  auto plugin_creator = MakeNMSPluginCreator("tftrt");
  auto plugin = createPlugin("nms_plugin_instance", plugin_creator.get(), {});
  auto nms = network->addPluginV2(nms_inputs.data(), 2, *plugin);

  // Mark the output.
  auto output = layer->getOutput(0);
  output->setName(kOutputTensor1);
  network->markOutput(*output);
  nms->getOutput(0)->setName(kOutputTensor2);
  network->markOutput(*nms->getOutput(0));

  // Build the engine.
  builder->setMaxBatchSize(1);
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builderConfig(
      builder->createBuilderConfig());
  builderConfig->setMaxWorkspaceSize(1 << 20);
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *builderConfig));
  EXPECT_NE(engine, nullptr);
  // Serialize the engine to create a model, then close everything.
  TrtUniquePtrType<nvinfer1::IHostMemory> model(engine->serialize());
  return model;
}

template <typename T>
unsigned GetBindingSizeBytes(const nvinfer1::ICudaEngine& engine, int index,
                             unsigned batch_size) {
  unsigned vol = batch_size;
  auto dims = engine.getBindingDimensions(index);
  int vecDim = engine.getBindingVectorizedDim(index);
  if (-1 != vecDim)  // i.e., 0 != lgScalarsPerVector
  {
    int scalarsPerVec = engine.getBindingComponentsPerElement(index);
    // Divide round up.
    dims.d[vecDim] = (dims.d[vecDim] + scalarsPerVec - 1 / scalarsPerVec);
    vol *= scalarsPerVec;
  }
  vol *= std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<>());
  return vol * sizeof(T);
}

// Executes the network.
void Execute(nvinfer1::IExecutionContext* context, const float* input1,
             const float* input2, float* output1, float* output2) {
  const nvinfer1::ICudaEngine& engine = context->getEngine();

  // We have two bindings: input and output.
  ASSERT_EQ(engine.getNbBindings(), 4);
  const int input_index1 = engine.getBindingIndex(kInputTensor1);
  const int input_index2 = engine.getBindingIndex(kInputTensor2);
  const int output_index1 = engine.getBindingIndex(kOutputTensor1);
  const int output_index2 = engine.getBindingIndex(kOutputTensor2);

  // Create GPU buffers and a stream
  std::vector<void*> buffers(engine.getNbBindings());
  for (int i = 0; i < buffers.size(); i++) {
    ASSERT_EQ(
        0, cudaMalloc(&buffers[i], GetBindingSizeBytes<float>(engine, i, 1)));
  }

  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));

  // Copy the input to the GPU, execute the network, and copy the output back.
  //
  // Note that since the host buffer was not created as pinned memory, these
  // async copies are turned into sync copies. So the following synchronization
  // could be removed.
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input_index1], input1, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input_index2], input2, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  context->enqueueV2(buffers.data(), stream, nullptr);
  ASSERT_EQ(0, cudaMemcpyAsync(output1, buffers[output_index1], sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  ASSERT_EQ(
      0, cudaMemcpyAsync(output2, buffers[output_index2],
                         GetBindingSizeBytes<int32>(engine, output_index2, 1),
                         cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release the stream and the buffers
  for (int i = 0; i < buffers.size(); i++) {
    ASSERT_EQ(0, cudaFree(buffers[i]));
  }
  cudaStreamDestroy(stream);
}

TEST(TensorrtTest, BasicFunctions) {
  // We must register the plugin creator in order to deserialize the plugin.
  auto plugin_creator = MakeNMSPluginCreator("tftrt");
  getPluginRegistry()->registerCreator(*plugin_creator, "tftrt");

  // Handle the case where the test is run on machine with no gpu available.
  if (CHECK_NOTNULL(GPUMachineManager())->VisibleDeviceCount() <= 0) {
    LOG(WARNING) << "No gpu device available, probably not being run on a gpu "
                    "machine. Skipping...";
    return;
  }

  // Create a serialized engine
  TrtUniquePtrType<nvinfer1::IHostMemory> model = CreateSerializedEngine();
  // Use the model to create an engine and then an execution context.
  Logger& logger = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(model->data(), model->size(), nullptr));
  TrtUniquePtrType<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());

  // Execute the network.
  float input1 = 1234;
  float input2 = 567;
  std::vector<float> output1(
      GetBindingSizeBytes<float>(*engine, 2, 1) / sizeof(float), 0.0f);
  std::vector<float> output2(
      GetBindingSizeBytes<int32>(*engine, 3, 1) / sizeof(int32), 0.0f);
  ASSERT_EQ(output1.size(), 1);
  ASSERT_EQ(output2.size(), 1);
  Execute(context.get(), &input1, &input2, output1.data(), output2.data());
  EXPECT_EQ(output1[0], input1 + input2);
  EXPECT_EQ(output2[0], 0);
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
