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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/gpu_module_plugin.h"

#include <dlfcn.h>

#include <memory>
#include <string>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace acceleration {

using ::tflite::delegates::TfLiteDelegatePtr;
using SymbolFunc = const TfLiteDelegatePlugin*();

// Function name used to get a pointer to GpuDelegatePlugin.
constexpr char kPluginGetterSymbolName[] = "TfLiteGpuDelegatePluginCApi";

std::unique_ptr<delegates::DelegatePluginInterface> GpuModulePlugin::New(
    const TFLiteSettings& acceleration) {
  return std::unique_ptr<GpuModulePlugin>(new GpuModulePlugin(acceleration));
}

int GpuModulePlugin::GetDelegateErrno(TfLiteDelegate* from_delegate) {
  if (!plugin_handle_) {
    return error_code_;
  }
  return plugin_handle_->get_delegate_errno(from_delegate);
}

TfLiteDelegatePtr GpuModulePlugin::Create() {
  if (!plugin_handle_) {
    return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  }
  return TfLiteDelegatePtr(plugin_handle_->create(tflite_settings_),
                           (plugin_handle_->destroy));
}

// In case GPU acceleration is not supported for this platform, we still need to
// construct an empty object so that Create() can later be called on it.
GpuModulePlugin::GpuModulePlugin(const TFLiteSettings& tflite_settings) {
  TFLiteSettingsT settings_obj;
  tflite_settings.UnPackTo(&settings_obj);
  fbb_.Finish(CreateTFLiteSettings(fbb_, &settings_obj));
  tflite_settings_ =
      flatbuffers::GetRoot<TFLiteSettings>(fbb_.GetBufferPointer());

  module_ = dlopen(tflite_settings_->stable_delegate_loader_settings()
                       ->delegate_path()
                       ->c_str(),
                   RTLD_NOW | RTLD_LOCAL);
  if (!module_) {
    TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Failed to load Gpu Module from %s",
                    tflite_settings_->stable_delegate_loader_settings()
                        ->delegate_path()
                        ->c_str());
    error_code_ = kMinibenchmarkCannotLoadGpuModule;
    return;
  }
  void* sym = dlsym(module_, kPluginGetterSymbolName);
  if (!sym) {
    TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Failed to create symbol '%s'",
                    kPluginGetterSymbolName);
    error_code_ = kMinibenchmarkCannotLoadGpuModule;
    return;
  }

  plugin_handle_ = reinterpret_cast<SymbolFunc*>(sym)();
  if (!plugin_handle_) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_WARNING,
        "GPU Module loaded successfully from %s, but plugin handle is null.",
        tflite_settings_->stable_delegate_loader_settings()
            ->delegate_path()
            ->c_str());
    error_code_ = kMinibenchmarkDelegatePluginNotFound;
  }
}

GpuModulePlugin::~GpuModulePlugin() {
  if (module_) {
    dlclose(module_);
  }
}

static auto* g_delegate_plugin_GpuModulePlugin =
    new tflite::delegates::DelegatePluginRegistry::Register(
        "GpuModulePlugin", GpuModulePlugin ::New);
}  // namespace acceleration
}  // namespace tflite
