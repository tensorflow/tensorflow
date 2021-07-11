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
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"

namespace tflite {
namespace delegates {
namespace {

TfLiteGpuInferencePriority ConvertInferencePriority(
    GPUInferencePriority priority) {
  switch (priority) {
    case GPUInferencePriority_GPU_PRIORITY_AUTO:
      return TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    case GPUInferencePriority_GPU_PRIORITY_MAX_PRECISION:
      return TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    case GPUInferencePriority_GPU_PRIORITY_MIN_LATENCY:
      return TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    case GPUInferencePriority_GPU_PRIORITY_MIN_MEMORY_USAGE:
      return TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
  }
}

}  // namespace

class GpuPlugin : public DelegatePluginInterface {
 public:
  TfLiteDelegatePtr Create() override {
    return TfLiteDelegatePtr(TfLiteGpuDelegateV2Create(&options_),
                             TfLiteGpuDelegateV2Delete);
  }
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override { return 0; }
  static std::unique_ptr<DelegatePluginInterface> New(
      const TFLiteSettings& acceleration) {
    return absl::make_unique<GpuPlugin>(acceleration);
  }
  explicit GpuPlugin(const TFLiteSettings& tflite_settings)
      : options_(TfLiteGpuDelegateOptionsV2Default()) {
    if (tflite_settings.max_delegated_partitions() >= 0) {
      options_.max_delegated_partitions =
          tflite_settings.max_delegated_partitions();
    }

    const auto* gpu_settings = tflite_settings.gpu_settings();
    if (!gpu_settings) return;

    options_.inference_preference = gpu_settings->inference_preference();

    if (gpu_settings->inference_priority1() > 0) {
      // User has specified their own inference priorities, so just copy over.
      options_.inference_priority1 =
          ConvertInferencePriority(gpu_settings->inference_priority1());
      options_.inference_priority2 =
          ConvertInferencePriority(gpu_settings->inference_priority2());
      options_.inference_priority3 =
          ConvertInferencePriority(gpu_settings->inference_priority3());
    } else {
      options_.inference_priority1 =
          gpu_settings->is_precision_loss_allowed()
              ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
              : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    }

    if (gpu_settings->enable_quantized_inference()) {
      options_.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
    }
    if (gpu_settings->force_backend() == GPUBackend_OPENCL) {
      options_.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
    } else if (gpu_settings->force_backend() == GPUBackend_OPENGL) {
      options_.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
    }
    if (gpu_settings->cache_directory() &&
        gpu_settings->cache_directory()->size() > 0 &&
        gpu_settings->model_token() && gpu_settings->model_token()->size()) {
      cache_dir_ = gpu_settings->cache_directory()->string_view();
      model_token_ = gpu_settings->model_token()->string_view();
      options_.serialization_dir = cache_dir_.c_str();
      options_.model_token = model_token_.c_str();
      options_.experimental_flags |=
          TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    }
  }

 private:
  TfLiteGpuDelegateOptionsV2 options_;
  std::string cache_dir_;
  std::string model_token_;
};

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(GpuPlugin, GpuPlugin::New);

}  // namespace delegates
}  // namespace tflite
