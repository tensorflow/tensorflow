/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#import <Metal/Metal.h>

#include <iostream>
#include <string>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/model_transformations.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

void FillInputTensor(tflite::Interpreter* interpreter) {
  for (int k = 0; k < interpreter->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr = interpreter->tensor(interpreter->inputs()[k]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);
    if (tensor_ptr->type == kTfLiteFloat32) {
      float* p = interpreter->typed_input_tensor<float>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = std::sin(i);
      }
    }
    if (tensor_ptr->type == kTfLiteInt32) {
      int* p = interpreter->typed_input_tensor<int>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 32;
      }
    }
    if (tensor_ptr->type == kTfLiteInt8) {
      int8_t* p = interpreter->typed_input_tensor<int8_t>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 256 - 128;
      }
    }
    if (tensor_ptr->type == kTfLiteUInt8) {
      uint8_t* p = interpreter->typed_input_tensor<uint8_t>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 256;
      }
    }
  }
}

absl::Status CompareCPUGPUResults(tflite::Interpreter* cpu, const std::vector<Value*>& outputs,
                                  InferenceContext* gpu_context, float per_element_eps) {
  for (int i = 0; i < outputs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(outputs[i]->tensor.ref);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);

    std::cout << "Output " << tensor_ptr->name << ":" << std::endl;

    tflite::gpu::TensorFloat32 gpu_tensor;
    RETURN_IF_ERROR(gpu_context->GetOutputTensor(outputs[i]->id, &gpu_tensor));

    const int kMaxPrint = 10;
    int printed = 0;
    int total_different = 0;
    for (int k = 0; k < tensor_elements_count; ++k) {
      float cpu_val = 0.0f;
      float gpu_val = 0.0f;
      if (tensor_ptr->type == kTfLiteFloat32) {
        const float* cpu_out = tensor_ptr->data.f;
        const float* gpu_out = gpu_tensor.data.data();
        cpu_val = cpu_out[k];
        gpu_val = gpu_out[k];
      }
      const float abs_diff = fabs(cpu_val - gpu_val);
      if (abs_diff > per_element_eps) {
        total_different++;
        if (printed < kMaxPrint) {
          std::cout << "Element #" << k << ": CPU value - " << cpu_val << ", GPU value - "
                    << gpu_val << ", abs diff - " << abs_diff << std::endl;
          printed++;
        }
        if (printed == kMaxPrint) {
          std::cout << "Printed " << kMaxPrint << " different elements, threshold - "
                    << per_element_eps << ", next different elements skipped" << std::endl;
          printed++;
        }
      }
    }
    std::cout << "Total " << total_different << " different elements, for output #" << i
              << ", threshold - " << per_element_eps << std::endl;
  }
  return absl::OkStatus();
}

absl::Status TestCorrectnessVsTfliteCPU(const std::unique_ptr<FlatBufferModel>& flatbuffer,
                                        GraphFloat32* graph, bool use_fp16 = true,
                                        float per_element_eps = 1e-4f) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  std::string device_name = std::string([[device name] UTF8String]);
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);
  CalculationsPrecision precision;
  if (use_fp16) {
    if (gpu_info.IsRoundToNearestSupported()) {
      precision = CalculationsPrecision::F16;
    } else {
      precision = CalculationsPrecision::F32_F16;
    }
  } else {
    precision = CalculationsPrecision::F32;
  }

  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = GetFastestStorageType(gpu_info);
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  InferenceContext inference_context;
  RETURN_IF_ERROR(inference_context.InitFromGraphWithTransforms(create_info, graph, device));

  ops::builtin::BuiltinOpResolver op_resolver;
  tflite::InterpreterBuilder builder(*flatbuffer, op_resolver);

  // CPU.
  std::unique_ptr<tflite::Interpreter> cpu_inference;
  builder(&cpu_inference);
  if (!cpu_inference) {
    return absl::InternalError("Failed to build CPU inference.");
  }
  auto status = cpu_inference->AllocateTensors();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to AllocateTensors for CPU inference.");
  }
  FillInputTensor(cpu_inference.get());
  status = cpu_inference->Invoke();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to Invoke CPU inference.");
  }

  for (auto& input : graph->inputs()) {
    TensorFloat32 src_tensor;
    src_tensor.id = input->id;
    src_tensor.shape = input->tensor.shape;
    src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
    for (int j = 0; j < src_tensor.data.size(); ++j) {
      src_tensor.data[j] = std::sin(j);
    }
    RETURN_IF_ERROR(inference_context.SetInputTensor(input->id, src_tensor));
  }

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  inference_context.EncodeWithEncoder(encoder);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  RETURN_IF_ERROR(CompareCPUGPUResults(cpu_inference.get(), graph->outputs(), &inference_context,
                                       per_element_eps));
  return absl::OkStatus();
}

absl::Status GPUBenchmark(GraphFloat32* graph, int num_tests, int iterations,
                          bool use_fp16 = true) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  std::string device_name = std::string([[device name] UTF8String]);
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);
  CalculationsPrecision precision;
  if (use_fp16) {
    if (gpu_info.IsRoundToNearestSupported()) {
      precision = CalculationsPrecision::F16;
    } else {
      precision = CalculationsPrecision::F32_F16;
    }
  } else {
    precision = CalculationsPrecision::F32;
  }

  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = GetFastestStorageType(gpu_info);
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  InferenceContext inference_context;
  RETURN_IF_ERROR(inference_context.InitFromGraphWithTransforms(create_info, graph, device));

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  bool kPerOpProfiling = false;
  if (kPerOpProfiling) {
    ProfilingInfo profiling_info;
    inference_context.Profile(device, &profiling_info);
    std::cout << profiling_info.GetDetailedReport() << std::endl;
  }
  uint64_t runtime_mem_bytes = inference_context.GetIntermediateTensorsSize();
  std::cout << "Memory for intermediate tensors - " << runtime_mem_bytes / 1024.0 / 1024.0 << " MB"
            << std::endl;
  const uint64_t const_mem_bytes = inference_context.GetConstantTensorsSize();
  std::cout << "Memory for constant tensors - " << const_mem_bytes / 1024.0 / 1024.0 << " MB"
            << std::endl;
  std::cout << "Total tensors memory(const + intermediate) - "
            << (const_mem_bytes + runtime_mem_bytes) / 1024.0 / 1024.0 << " MB" << std::endl;
  const std::string precision_str = use_fp16 ? "FP16" : "FP32";
  std::cout << "Measuring started: (" << num_tests << " tests, " << iterations
            << " iterations every test, " << precision_str << " precision)" << std::endl;
  for (int j = 0; j < num_tests; ++j) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        inference_context.EncodeWithEncoder(encoder);
        [encoder endEncoding];
        [command_buffer commit];
        if (i == iterations - 1) {
          [command_buffer waitUntilCompleted];
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double t0 = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) /
                iterations;
    std::cout << "  Test: #" << j << " - " << t0 << "ms" << std::endl;
  }
  return absl::OkStatus();
}

absl::Status GPUBenchmarkSerialized(GraphFloat32* graph, bool use_fp16 = true) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  std::string device_name = std::string([[device name] UTF8String]);
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);
  CalculationsPrecision precision;
  if (use_fp16) {
    if (gpu_info.IsRoundToNearestSupported()) {
      precision = CalculationsPrecision::F16;
    } else {
      precision = CalculationsPrecision::F32_F16;
    }
  } else {
    precision = CalculationsPrecision::F32;
  }

  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = GetFastestStorageType(gpu_info);
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  InferenceContext inference_context;
  std::vector<uint8_t> serialized_model;
  RETURN_IF_ERROR(
      inference_context.InitFromGraphWithTransforms(create_info, graph, device, &serialized_model));

  {  // calculating time without building serialized model
    InferenceContext test_context;
    const auto start = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(test_context.InitFromGraphWithTransforms(create_info, graph, device));
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    std::cout << "Inference context initialization total time - " << total_time_ms << "ms"
              << std::endl;
  }

  std::vector<TensorFloat32> src_tensors(graph->inputs().size());
  for (int i = 0; i < graph->inputs().size(); ++i) {
    src_tensors[i].id = graph->inputs()[i]->id;
    src_tensors[i].shape = graph->inputs()[i]->tensor.shape;
    src_tensors[i].data.resize(src_tensors[i].shape.DimensionsProduct());
    for (int j = 0; j < src_tensors[i].data.size(); ++j) {
      src_tensors[i].data[j] = std::sin(j);
    }
  }
  for (int i = 0; i < graph->inputs().size(); ++i) {
    RETURN_IF_ERROR(inference_context.SetInputTensor(graph->inputs()[i]->id, src_tensors[i]));
  }
  @autoreleasepool {
    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    inference_context.EncodeWithEncoder(encoder);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  std::vector<TensorFloat32> dst_tensors(graph->outputs().size());
  for (int i = 0; i < graph->outputs().size(); ++i) {
    RETURN_IF_ERROR(inference_context.GetOutputTensor(graph->outputs()[i]->id, &dst_tensors[i]));
  }

  InferenceContext serialized_context;
  {
    const auto start = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(serialized_context.RestoreDeserialized(serialized_model, device));
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    std::cout << "Serialized inference context initialization total time - " << total_time_ms
              << "ms" << std::endl;
  }
  for (int i = 0; i < graph->inputs().size(); ++i) {
    RETURN_IF_ERROR(serialized_context.SetInputTensor(graph->inputs()[i]->id, src_tensors[i]));
  }

  @autoreleasepool {
    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    serialized_context.EncodeWithEncoder(encoder);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  std::vector<TensorFloat32> dst_tensors_v2(graph->outputs().size());
  for (int i = 0; i < graph->outputs().size(); ++i) {
    RETURN_IF_ERROR(
        serialized_context.GetOutputTensor(graph->outputs()[i]->id, &dst_tensors_v2[i]));
  }

  for (int i = 0; i < graph->outputs().size(); ++i) {
    if (dst_tensors[i].data.size() != dst_tensors_v2[i].data.size()) {
      std::cout << "Different sizes for " << i << " output tensor" << std::endl;
      break;
    }
    for (int j = 0; j < dst_tensors[i].data.size(); ++j) {
      if (dst_tensors[i].data[j] != dst_tensors_v2[i].data[j]) {
        std::cout << "Different elements for " << j << " element in " << i
                  << " tensor: " << dst_tensors[i].data[j] << " - " << dst_tensors_v2[i].data[j]
                  << std::endl;
        break;
      }
    }
  }

  return absl::OkStatus();
}

class DelegateContext {
 public:
  bool Init(TfLiteContext* context, const TfLiteDelegateParams* delegate_params) {
    auto denormalized_graph = reinterpret_cast<GraphFloat32*>(delegate_params->delegate->data_);
    absl::Status status = BuildModel(context, delegate_params, denormalized_graph);
    if (!status.ok()) {
      TF_LITE_KERNEL_LOG(context, std::string(status.message()).c_str());
    }
    return status.ok();
  }
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  const TfLiteRegistration kRegistration = {
      .init = [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        auto* delegate_context = new DelegateContext();
        if (!delegate_context->Init(context,
                                    reinterpret_cast<const TfLiteDelegateParams*>(buffer))) {
          delete delegate_context;
          return nullptr;
        }
        return delegate_context;
      },
      .free = [](TfLiteContext* context, void* buffer) -> void {
        delete reinterpret_cast<DelegateContext*>(buffer);
      },
      .prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return node->user_data ? kTfLiteOk : kTfLiteError;
      },
      .invoke = nullptr,
  };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(context, kRegistration,
                                                                     ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

absl::Status FlatBufferToGPUGraph(const std::unique_ptr<tflite::FlatBufferModel>& flatbuffer,
                                  GraphFloat32* graph) {
  ops::builtin::BuiltinOpResolver op_resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(*flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    return absl::InternalError("Unable to prepare TfLite interpreter.");
  }
  TfLiteDelegate delegate;
  delegate.data_ = graph;
  delegate.flags = kTfLiteDelegateFlagsNone;
  delegate.Prepare = DelegatePrepare;
  delegate.CopyFromBufferHandle = nullptr;
  delegate.CopyToBufferHandle = nullptr;
  delegate.FreeBufferHandle = nullptr;

  if (interpreter->ModifyGraphWithDelegate(&delegate) != kTfLiteOk) {
    return absl::InternalError("Conversion from TfLite model failed.");
  }

  ModelTransformer transformer(graph);
  if (!ApplyModelTransformations(&transformer)) {
    return absl::InternalError("Graph transformations failed");
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace tflite

int main(int argc, char** argv) {
  @autoreleasepool {
    NSBundle* main = [NSBundle mainBundle];
    NSArray<NSString*>* model_paths = [main pathsForResourcesOfType:@"tflite" inDirectory:nil];
    for (id model_path in model_paths) {
      NSString* model_name = [[model_path lastPathComponent] stringByDeletingPathExtension];
      std::string m_name = std::string([model_name UTF8String]);
      std::string path = std::string([model_path UTF8String]);
      std::cout << m_name << std::endl;
      auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(path.c_str());
      if (!flatbuffer) {
        std::cout << "Failed flatbuffer reading." << std::endl;
      }

      tflite::gpu::GraphFloat32 graph;
      auto s = tflite::gpu::metal::FlatBufferToGPUGraph(flatbuffer, &graph);
      if (!s.ok()) {
        std::cout << "Failed flatbuffer to graph conversion. " << s.message() << std::endl;
      }

      s = tflite::gpu::metal::GPUBenchmark(&graph, 5, 200, /*use_fp16*/ true);
      if (!s.ok()) {
        std::cout << "Error in GPUBenchmark. " << s.message() << std::endl;
      }

      s = tflite::gpu::metal::TestCorrectnessVsTfliteCPU(flatbuffer, &graph, /*use_fp16*/ true);
      if (!s.ok()) {
        std::cout << "Error in GPUBenchmark. " << s.message() << std::endl;
      }

      s = tflite::gpu::metal::GPUBenchmarkSerialized(&graph, true);
      if (!s.ok()) {
        std::cout << "Error in GPUBenchmark. " << s.message() << std::endl;
      }
    }
  }

  return 0;
}
