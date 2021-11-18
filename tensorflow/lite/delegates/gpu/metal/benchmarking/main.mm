/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the Licensgoe is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#import <Metal/Metal.h>

#include <iostream>
#include <string>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
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
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

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

  InferenceContext::CreateInferenceInfo create_info;
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
  uint64_t mem_bytes = inference_context.GetIntermediateTensorsSize();
  std::cout << "Memory for intermediate tensors - " << mem_bytes / 1024.0 / 1024.0 << " MB"
            << std::endl;
  const std::string precision_str = use_fp16 ? "FP16" : "FP32";
  std::cout << "Measuring started: (" << num_tests << " tests, " << iterations
      << " iterations every test, " << precision_str << " precision)" << std::endl;
  for (int j = 0; j < num_tests; ++j) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        inference_context.EncodeWithEncoder(encoder);
        [encoder endEncoding];
        [command_buffer commit];
        if (i == iterations - 1) {
          [command_buffer waitUntilCompleted];
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double t0 = double(std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                           .count()) /
                iterations;
    std::cout << "  Test: #" << j << " - " << t0 << "ms" << std::endl;
  }
  return absl::OkStatus();
}

class DelegateContext {
 public:
  bool Init(TfLiteContext* context,
            const TfLiteDelegateParams* delegate_params) {
    auto denormalized_graph =
        reinterpret_cast<GraphFloat32*>(delegate_params->delegate->data_);
    absl::Status status =
        BuildModel(context, delegate_params, denormalized_graph);
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
        if (!delegate_context->Init(
                context,
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
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

absl::Status FlatBufferToGPUGraph(
    const std::unique_ptr<tflite::FlatBufferModel>& flatbuffer,
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
    NSBundle *main = [NSBundle mainBundle];
    NSArray<NSString*>* model_paths = [main pathsForResourcesOfType:@"tflite" inDirectory:nil];
    for (id model_path in model_paths) {
      NSString *model_name = [[model_path lastPathComponent] stringByDeletingPathExtension];
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

      s = tflite::gpu::metal::GPUBenchmark(&graph, 5, 200, true);
      if (!s.ok()) {
        std::cout << "Error in GPUBenchmark. " << s.message() << std::endl;
      }
    }
  }

  return 0;
}
