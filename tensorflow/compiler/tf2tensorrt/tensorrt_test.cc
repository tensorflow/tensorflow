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

#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "third_party/tensorrt/NvInfer.h"

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

const char* kInputTensor = "input";
const char* kOutputTensor = "output";

// Creates a network to compute y=2x+3.
TrtUniquePtrType<nvinfer1::IHostMemory> CreateSerializedEngine() {
  Logger& logger = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  ScopedWeights weights(2.0);
  ScopedWeights bias(3.0);
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(0L));
#else
  nvinfer1::INetworkDefinition* network = builder->createNetwork();
#endif
  // Add the input.
  auto input = network->addInput(kInputTensor, nvinfer1::DataType::kFLOAT,
                                 nvinfer1::Dims3{1, 1, 1});
  EXPECT_NE(input, nullptr);
  // Add the hidden layer.
  auto layer = network->addFullyConnected(*input, 1, weights.get(), bias.get());
  EXPECT_NE(layer, nullptr);
  // Mark the output.
  auto output = layer->getOutput(0);
  output->setName(kOutputTensor);
  network->markOutput(*output);
  // Build the engine
  builder->setMaxBatchSize(1);
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builderConfig(
      builder->createBuilderConfig());
  builderConfig->setMaxWorkspaceSize(1 << 10);
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *builderConfig));
#else
  builder->setMaxWorkspaceSize(1 << 10);
  auto engine = builder->buildCudaEngine(*network);
#endif
  EXPECT_NE(engine, nullptr);
  // Serialize the engine to create a model, then close everything.
  TrtUniquePtrType<nvinfer1::IHostMemory> model(engine->serialize());
  return model;
}

// Executes the network.
void Execute(nvinfer1::IExecutionContext* context, const float* input,
             float* output) {
  const nvinfer1::ICudaEngine& engine = context->getEngine();

  // We have two bindings: input and output.
  ASSERT_EQ(engine.getNbBindings(), 2);
  const int input_index = engine.getBindingIndex(kInputTensor);
  const int output_index = engine.getBindingIndex(kOutputTensor);

  // Create GPU buffers and a stream
  void* buffers[2];
  ASSERT_EQ(0, cudaMalloc(&buffers[input_index], sizeof(float)));
  ASSERT_EQ(0, cudaMalloc(&buffers[output_index], sizeof(float)));
  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));

  // Copy the input to the GPU, execute the network, and copy the output back.
  //
  // Note that since the host buffer was not created as pinned memory, these
  // async copies are turned into sync copies. So the following synchronization
  // could be removed.
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input_index], input, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  context->enqueue(1, buffers, stream, nullptr);
  ASSERT_EQ(0, cudaMemcpyAsync(output, buffers[output_index], sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release the stream and the buffers
  ASSERT_EQ(0, cudaFree(buffers[input_index]));
  ASSERT_EQ(0, cudaFree(buffers[output_index]));
  cudaStreamDestroy(stream);
}

TEST(TensorrtTest, BasicFunctions) {
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
  float input = 1234;
  float output;
  Execute(context.get(), &input, &output);
  EXPECT_EQ(output, input * 2 + 3);
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
