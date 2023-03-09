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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_GPU_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_GPU_PLUGIN_H_

// This file provides the GpuPlugin class, which implements the
// TFLite Delegate Plugin for the GPU Delegate.

#if defined(__ANDROID__) || defined(CL_DELEGATE_NO_GL)
#define TFLITE_SUPPORTS_GPU_DELEGATE 1
#endif

#include <string>

#if TFLITE_SUPPORTS_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if (TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR) || \
    (TARGET_OS_OSX && TARGET_CPU_ARM64)
// Only enable metal delegate when using a real iPhone device or Apple Silicon.
#define REAL_IPHONE_DEVICE
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif
#endif

#include "tensorflow/lite/core/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace delegates {

// Note that if running on GPU is not supported for some reason (e.g., desktop
// machine with no OpenGL/CL), this library will still compile but calling
// Create() will return a nullptr.
class GpuPlugin : public DelegatePluginInterface {
 public:
  explicit GpuPlugin(const TFLiteSettings& tflite_settings);
  static std::unique_ptr<DelegatePluginInterface> New(
      const TFLiteSettings& acceleration);

  TfLiteDelegatePtr Create() override;
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override;

#if TFLITE_SUPPORTS_GPU_DELEGATE
  const TfLiteGpuDelegateOptionsV2& Options() { return options_; }
#elif defined(REAL_IPHONE_DEVICE)
  const TFLGpuDelegateOptions& Options() { return options_; }
#endif

  std::string GetCacheDir() const { return cache_dir_; }
  std::string GetModelToken() const { return model_token_; }

 private:
#if TFLITE_SUPPORTS_GPU_DELEGATE
  TfLiteGpuDelegateOptionsV2 options_;
#elif defined(REAL_IPHONE_DEVICE)
  TFLGpuDelegateOptions options_;
#endif
  std::string cache_dir_;
  std::string model_token_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_GPU_PLUGIN_H_
