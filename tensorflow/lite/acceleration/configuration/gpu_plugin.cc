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
#include "tensorflow/lite/acceleration/configuration/gpu_plugin.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"

namespace tflite {
namespace delegates {

int GpuPlugin::GetDelegateErrno(TfLiteDelegate* from_delegate) { return 0; }

std::unique_ptr<DelegatePluginInterface> GpuPlugin::New(
    const TFLiteSettings& acceleration) {
  return std::make_unique<GpuPlugin>(acceleration);
}

#if TFLITE_SUPPORTS_GPU_DELEGATE

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

std::string GetCacheDirImpl(const TFLiteSettings& tflite_settings) {
  if (tflite_settings.compilation_caching_settings() &&
      tflite_settings.compilation_caching_settings()->cache_dir() &&
      tflite_settings.compilation_caching_settings()->cache_dir()->size() > 0) {
    return tflite_settings.compilation_caching_settings()->cache_dir()->str();
  } else if (tflite_settings.gpu_settings() &&
             tflite_settings.gpu_settings()->cache_directory() &&
             tflite_settings.gpu_settings()->cache_directory()->size() > 0) {
    return tflite_settings.gpu_settings()->cache_directory()->str();
  } else {
    return "";
  }
}

std::string GetModelTokenImpl(const TFLiteSettings& tflite_settings) {
  if (tflite_settings.compilation_caching_settings() &&
      tflite_settings.compilation_caching_settings()->model_token() &&
      tflite_settings.compilation_caching_settings()->model_token()->size() >
          0) {
    return tflite_settings.compilation_caching_settings()->model_token()->str();
  } else if (tflite_settings.gpu_settings() &&
             tflite_settings.gpu_settings()->model_token() &&
             tflite_settings.gpu_settings()->model_token()->size()) {
    return tflite_settings.gpu_settings()->model_token()->str();
  } else {
    return "";
  }
}

}  // namespace

TfLiteDelegatePtr GpuPlugin::Create() {
  return TfLiteDelegatePtr(TfLiteGpuDelegateV2Create(&options_),
                           TfLiteGpuDelegateV2Delete);
}

GpuPlugin::GpuPlugin(const TFLiteSettings& tflite_settings)
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

  const std::string cache_dir = GetCacheDirImpl(tflite_settings);
  const std::string model_token = GetModelTokenImpl(tflite_settings);
  if (!cache_dir.empty() && !model_token.empty()) {
    cache_dir_ = cache_dir;
    model_token_ = model_token;
    options_.serialization_dir = cache_dir_.c_str();
    options_.model_token = model_token_.c_str();
    options_.experimental_flags |=
        TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
  }
}

#elif defined(REAL_IPHONE_DEVICE)

TfLiteDelegatePtr GpuPlugin::Create() {
  return TfLiteDelegatePtr(TFLGpuDelegateCreate(&options_),
                           &TFLGpuDelegateDelete);
}

GpuPlugin::GpuPlugin(const TFLiteSettings& tflite_settings) {
  options_ = {0};
  const auto* gpu_settings = tflite_settings.gpu_settings();
  if (!gpu_settings) return;

  options_.allow_precision_loss = gpu_settings->is_precision_loss_allowed();
  options_.enable_quantization = gpu_settings->enable_quantized_inference();
}

#else

TfLiteDelegatePtr GpuPlugin::Create() {
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

// In case GPU acceleration is not supported for this platform, we still need to
// construct an empty object so that Create() can later be called on it.
GpuPlugin::GpuPlugin(const TFLiteSettings& tflite_settings) {}

#endif

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(GpuPlugin, GpuPlugin::New);

}  // namespace delegates
}  // namespace tflite
