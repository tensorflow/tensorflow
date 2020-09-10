/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/api.h"
#ifndef CL_DELEGATE_NO_GL
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#endif
#include "tensorflow/lite/delegates/gpu/gpu_backend.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace gpu {

absl::Status InitializeOpenClApi(
    GraphFloat32* graph, const TfLiteGpuDelegateOptionsV2& delegate_options,
    std::unique_ptr<InferenceBuilder>* builder,
    std::unique_ptr<cl::InferenceEnvironment>* inference_environment,
    bool* graph_is_destroyed) {
  if (graph_is_destroyed) {
    *graph_is_destroyed = false;
  }
  cl::InferenceEnvironmentOptions env_options;
  cl::InferenceEnvironmentProperties properties;
  RETURN_IF_ERROR(cl::NewInferenceEnvironment(
      env_options, inference_environment, &properties));
  cl::InferenceOptions options;
  // If is_precision_loss_allowed == -1, then just use priorities instead
  // of paying attention to is_precision_loss_allowed value.
  if (delegate_options.is_precision_loss_allowed == -1) {
    options.priority1 =
        GpuBackend::ToPriority(delegate_options.inference_priority1);
    options.priority2 =
        GpuBackend::ToPriority(delegate_options.inference_priority2);
    options.priority3 =
        GpuBackend::ToPriority(delegate_options.inference_priority3);
  } else {
    // Users set is_precision_loss_allowed explicitly, thus use it explicitly.
    if (delegate_options.is_precision_loss_allowed == 0) {
      options.priority1 = InferencePriority::MAX_PRECISION;
    } else {
      options.priority1 = InferencePriority::MIN_LATENCY;
    }
  }
  options.usage = GpuBackend::ToUsage(delegate_options.inference_preference);
  if (graph_is_destroyed) {
    *graph_is_destroyed = true;
  }
  RETURN_IF_ERROR(
      (*inference_environment)
          ->NewInferenceBuilder(options, std::move(*graph), builder));
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Initialized OpenCL-based API.");
  return absl::OkStatus();
}

#ifndef CL_DELEGATE_NO_GL
absl::Status InitializeOpenGlApi(
    GraphFloat32* graph, const TfLiteGpuDelegateOptionsV2& delegate_options,
    std::unique_ptr<InferenceBuilder>* builder,
    std::unique_ptr<gl::InferenceEnvironment>* inference_environment) {
  gl::InferenceEnvironmentOptions env_options;
  gl::InferenceEnvironmentProperties properties;
  RETURN_IF_ERROR(
      NewInferenceEnvironment(env_options, inference_environment, &properties));
  gl::InferenceOptions options;
  options.usage = GpuBackend::ToUsage(delegate_options.inference_preference);
  options.priority1 =
      GpuBackend::ToPriority(delegate_options.inference_priority1);
  options.priority2 =
      GpuBackend::ToPriority(delegate_options.inference_priority2);
  options.priority3 =
      GpuBackend::ToPriority(delegate_options.inference_priority3);
  RETURN_IF_ERROR(
      (*inference_environment)
          ->NewInferenceBuilder(std::move(*graph), options, builder));
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Initialized OpenGL-based API.");
  return absl::OkStatus();
}
#endif

class GpuBackendDefault : public GpuBackend {
 public:
  GpuBackendDefault() {}

  absl::Status Prepare(
      const TfLiteGpuDelegateOptionsV2& delegate_options, GraphFloat32* graph,
      std::function<absl::Status(GraphFloat32* graph)> initialize_graph,
      std::unique_ptr<InferenceBuilder>* builder) override {
#ifdef CL_DELEGATE_NO_GL
    return InitializeOpenClApi(graph, delegate_options, builder,
                               &cl_inference_environment_, nullptr);
#else
    bool graph_is_destroyed;
    const int experimental_flags = delegate_options.experimental_flags;
    if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY) {
      RETURN_IF_ERROR(InitializeOpenClApi(graph, delegate_options, builder,
                                          &cl_inference_environment_,
                                          &graph_is_destroyed));
    } else if (experimental_flags & TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY) {
      RETURN_IF_ERROR(InitializeOpenGlApi(graph, delegate_options, builder,
                                          &gl_inference_environment_));
    } else {
      // By default, we try CL first & fall back to GL if that fails.
      absl::Status status =
          InitializeOpenClApi(graph, delegate_options, builder,
                              &cl_inference_environment_, &graph_is_destroyed);
      if (!status.ok()) {
        TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                             std::string(status.message()).c_str());
        TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO, "Falling back to OpenGL");

        // Graph needs to be re-created because it is moved above.
        GraphFloat32 graph2;
        if (graph_is_destroyed) {
          RETURN_IF_ERROR(initialize_graph(&graph2));
        }
        RETURN_IF_ERROR(InitializeOpenGlApi(
            graph_is_destroyed ? &graph2 : graph, delegate_options, builder,
            &gl_inference_environment_));
      }
    }
    return absl::OkStatus();
#endif
  }

 private:
  std::unique_ptr<cl::InferenceEnvironment> cl_inference_environment_;
#ifndef CL_DELEGATE_NO_GL
  std::unique_ptr<gl::InferenceEnvironment> gl_inference_environment_;
#endif
};

}  // namespace gpu
}  // namespace tflite

extern "C" TfLiteDelegate* TfLiteGpuDelegateV2Create(
    const TfLiteGpuDelegateOptionsV2* options) {
  return tflite::gpu::TfLiteGpuDelegateCreateInternal(
      new tflite::gpu::GpuBackendDefault(), options);
}

extern "C" void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate) {
  return tflite::gpu::TfLiteGpuDelegateDeleteInternal(delegate);
}
