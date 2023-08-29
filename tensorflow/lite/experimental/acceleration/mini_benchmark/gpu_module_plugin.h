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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_GPU_MODULE_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_GPU_MODULE_PLUGIN_H_

// This file provides the GpuPlugin class, which implements the
// TFLite Delegate Plugin for the GPU Delegate.

#include <memory>
#include <string>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

namespace tflite {
namespace acceleration {

// A DelegatePlugin that uses external library to create GPU Plugin.
class GpuModulePlugin : public delegates::DelegatePluginInterface {
 public:
  static std::unique_ptr<DelegatePluginInterface> New(
      const TFLiteSettings& acceleration);

  // Move only.
  GpuModulePlugin(GpuModulePlugin&& other) = default;
  GpuModulePlugin& operator=(GpuModulePlugin&& other) = default;
  ~GpuModulePlugin() override;

  delegates::TfLiteDelegatePtr Create() override;
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override;

 private:
  explicit GpuModulePlugin(const TFLiteSettings& tflite_settings);

  // The handle to the loaded external library.
  void* module_ = nullptr;
  const TfLiteDelegatePlugin* plugin_handle_ = nullptr;
  // A copy of the input tflite_settings.
  flatbuffers::FlatBufferBuilder fbb_;
  // A pointer to the data in fbb_.
  const TFLiteSettings* tflite_settings_;
  MinibenchmarkStatus error_code_ = kMinibenchmarkSuccess;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_GPU_MODULE_PLUGIN_H_
