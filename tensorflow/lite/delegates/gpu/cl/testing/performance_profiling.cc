/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>
#include <string>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/cl/model_hints.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/general_transformations.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

class DelegateContext {
 public:
  bool Init(TfLiteContext* context,
            const TfLiteDelegateParams* delegate_params) {
    auto denormalized_graph =
        reinterpret_cast<GraphFloat32*>(delegate_params->delegate->data_);
    Status status = BuildModel(context, delegate_params, denormalized_graph);
    if (!status.ok()) {
      context->ReportError(context, "Failed to convert a model: %s",
                           status.error_message().c_str());
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

Status FlatBufferToGPUGraph(
    const std::unique_ptr<tflite::FlatBufferModel>& flatbuffer,
    GraphFloat32* graph) {
  tflite::ops::builtin::BuiltinOpResolver op_resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(*flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    return InternalError("Unable to prepare TfLite interpreter.");
  }
  interpreter->UseNNAPI(false);
  TfLiteDelegate delegate;
  delegate.data_ = graph;
  delegate.flags = kTfLiteDelegateFlagsNone;
  delegate.Prepare = DelegatePrepare;
  delegate.CopyFromBufferHandle = nullptr;
  delegate.CopyToBufferHandle = nullptr;
  delegate.FreeBufferHandle = nullptr;

  if (interpreter->ModifyGraphWithDelegate(&delegate) != kTfLiteOk) {
    return InternalError("Conversion from TfLite model failed.");
  }

  NullTransformationReporter reporter;
  ModelTransformer transformer(graph, &reporter);
  if (!ApplyGeneralTransformations(&transformer)) {
    return InternalError("Graph general transformations failed");
  }

  return OkStatus();
}
}  // namespace

Status RunModelSample(const std::string& model_name) {
  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  RETURN_IF_ERROR(FlatBufferToGPUGraph(flatbuffer, &graph_cl));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  InferenceContext::CreateInferenceInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device());
  std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
  std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
  InferenceContext context;
  RETURN_IF_ERROR(
      context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

  auto* queue = env.profiling_queue();
  ProfilingInfo profiling_info;
  RETURN_IF_ERROR(context.Profile(queue, &profiling_info));
  std::cout << profiling_info.GetDetailedReport() << std::endl;
  uint64_t mem_bytes = context.GetSizeOfMemoryAllocatedForIntermediateTensors();
  std::cout << "Memory for intermediate tensors - "
            << mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;

  const int num_runs_per_sec = std::max(
      1, static_cast<int>(1000.0f / absl::ToDoubleMilliseconds(
                                        profiling_info.GetTotalTime())));

  const int kNumRuns = 10;
  for (int i = 0; i < kNumRuns; ++i) {
    const auto start = absl::Now();
    for (int k = 0; k < num_runs_per_sec; ++k) {
      RETURN_IF_ERROR(context.AddToQueue(env.queue()));
    }
    RETURN_IF_ERROR(env.queue()->WaitForCompletion());
    const auto end = absl::Now();
    const double total_time_ms =
        static_cast<double>((end - start) / absl::Nanoseconds(1)) * 1e-6;
    const double average_inference_time = total_time_ms / num_runs_per_sec;
    std::cout << "Total time - " << average_inference_time << "ms" << std::endl;
  }

  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << "Expected model path as second argument.";
    return -1;
  }

  auto load_status = tflite::gpu::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cerr << load_status.message();
    return -1;
  }

  auto run_status = tflite::gpu::cl::RunModelSample(argv[1]);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }

  return EXIT_SUCCESS;
}
