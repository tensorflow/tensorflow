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

#include <string>

#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"

namespace tflite {
namespace delegates {

class GpuPlugin : public DelegatePluginInterface {
 public:
  explicit GpuPlugin(const TFLiteSettings& tflite_settings);
  static std::unique_ptr<DelegatePluginInterface> New(
      const TFLiteSettings& acceleration);

  TfLiteDelegatePtr Create() override;
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override;
  const TfLiteGpuDelegateOptionsV2& Options();

 private:
  TfLiteGpuDelegateOptionsV2 options_;
  std::string cache_dir_;
  std::string model_token_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_GPU_PLUGIN_H_
