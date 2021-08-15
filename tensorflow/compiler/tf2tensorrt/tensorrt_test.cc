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

const char* kInputTensor1 = "input1";
const char* kInputTensor2 = "input2";
const char* kOutputTensor = "output";

// Creates a network to compute x+y.
TrtUniquePtrType<nvinfer1::IHostMemory> CreateSerializedEngine() {
  Logger& logger = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(0L));
  // Add the input.
  auto input1 = network->addInput(kInputTensor1, nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 1, 1});
  auto input2 = network->addInput(kInputTensor2, nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 1, 1});
  EXPECT_NE(input1, nullptr);
  EXPECT_NE(input2, nullptr);
  // Add an ILayer layer.
  auto layer = network->addElementWise(*input1, *input2,
                                       nvinfer1::ElementWiseOperation::kSUM);
  EXPECT_NE(layer, nullptr);
  // Mark the output.
  auto output = layer->getOutput(0);
  output->setName(kOutputTensor);
  network->markOutput(*output);
  // Build the engine.
  builder->setMaxBatchSize(1);
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builderConfig(
      builder->createBuilderConfig());
  builderConfig->setMaxWorkspaceSize(1 << 10);
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *builderConfig));
  EXPECT_NE(engine, nullptr);
  // Serialize the engine to create a model, then close everything.
  TrtUniquePtrType<nvinfer1::IHostMemory> model(engine->serialize());
  return model;
}

// Executes the network.
void Execute(nvinfer1::IExecutionContext* context, const float* input1,
             const float* input2, float* output) {
  const nvinfer1::ICudaEngine& engine = context->getEngine();

  // We have two bindings: input and output.
  ASSERT_EQ(engine.getNbBindings(), 3);
  const int input_index1 = engine.getBindingIndex(kInputTensor1);
  const int input_index2 = engine.getBindingIndex(kInputTensor2);
  const int output_index = engine.getBindingIndex(kOutputTensor);

  // Create GPU buffers and a stream
  void* buffers[3];
  ASSERT_EQ(0, cudaMalloc(&buffers[input_index1], sizeof(float)));
  ASSERT_EQ(0, cudaMalloc(&buffers[input_index2], sizeof(float)));
  ASSERT_EQ(0, cudaMalloc(&buffers[output_index], sizeof(float)));
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
  context->enqueue(1, buffers, stream, nullptr);
  ASSERT_EQ(0, cudaMemcpyAsync(output, buffers[output_index], sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release the stream and the buffers
  ASSERT_EQ(0, cudaFree(buffers[input_index1]));
  ASSERT_EQ(0, cudaFree(buffers[input_index2]));
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
  float input1 = 1234;
  float input2 = 567;
  float output;
  Execute(context.get(), &input1, &input2, &output);
  EXPECT_EQ(output, input1 + input2);
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
