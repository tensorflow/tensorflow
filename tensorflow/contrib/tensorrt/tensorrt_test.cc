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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    switch (severity) {
      case Severity::kINFO:
        LOG(INFO) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      default:
        break;
    }
  }
};

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
nvinfer1::IHostMemory* CreateNetwork() {
  Logger logger;
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
  ScopedWeights weights(2.0);
  ScopedWeights bias(3.0);

  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  // Add the input.
  auto input = network->addInput(kInputTensor, nvinfer1::DataType::kFLOAT,
                                 nvinfer1::DimsCHW{1, 1, 1});
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
  builder->setMaxWorkspaceSize(1 << 10);
  auto engine = builder->buildCudaEngine(*network);
  EXPECT_NE(engine, nullptr);
  // Serialize the engine to create a model, then close everything.
  nvinfer1::IHostMemory* model = engine->serialize();
  network->destroy();
  engine->destroy();
  builder->destroy();
  return model;
}

// Executes the network.
void Execute(nvinfer1::IExecutionContext& context, const float* input,
             float* output) {
  const nvinfer1::ICudaEngine& engine = context.getEngine();

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
  context.enqueue(1, buffers, stream, nullptr);
  ASSERT_EQ(0, cudaMemcpyAsync(output, buffers[output_index], sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release the stream and the buffers
  cudaStreamDestroy(stream);
  ASSERT_EQ(0, cudaFree(buffers[input_index]));
  ASSERT_EQ(0, cudaFree(buffers[output_index]));
}

TEST(TensorrtTest, BasicFunctions) {
  // Create the network model.
  nvinfer1::IHostMemory* model = CreateNetwork();
  // Use the model to create an engine and then an execution context.
  Logger logger;
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
  nvinfer1::ICudaEngine* engine =
      runtime->deserializeCudaEngine(model->data(), model->size(), nullptr);
  model->destroy();
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  // Execute the network.
  float input = 1234;
  float output;
  Execute(*context, &input, &output);
  EXPECT_EQ(output, input * 2 + 3);

  // Destroy the engine.
  context->destroy();
  engine->destroy();
  runtime->destroy();
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
