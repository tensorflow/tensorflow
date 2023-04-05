/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_PLUGIN_H_

// This file provides the StableDelegatePlugin class, which implements the
// TFLite Delegate Plugin Interface for the stable delegates.

#include <memory>
#include <string>

#include "tensorflow/lite/core/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/delegate_plugin.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace delegates {

class StableDelegatePlugin : public DelegatePluginInterface {
 public:
  static std::unique_ptr<StableDelegatePlugin> New(
      const TFLiteSettings& tflite_settings) {
    return std::make_unique<StableDelegatePlugin>(tflite_settings);
  }

  explicit StableDelegatePlugin(const TFLiteSettings& tflite_settings) {
    // Creates a copy of TFLiteSettings within the stable delegate plugin.
    TFLiteSettingsT tflite_settings_t;
    tflite_settings.UnPackTo(&tflite_settings_t);
    tflite_settings_builder_.Finish(
        CreateTFLiteSettings(tflite_settings_builder_, &tflite_settings_t));
    const StableDelegateLoaderSettings* stable_delegate_loader_settings =
        GetTFLiteSettings()->stable_delegate_loader_settings();
    if (!stable_delegate_loader_settings ||
        !stable_delegate_loader_settings->delegate_path() ||
        stable_delegate_loader_settings->delegate_path()->Length() == 0) {
      TFLITE_LOG(ERROR) << "The delegate path field is not available from the "
                           "provided stable delegate loader settings.";
      return;
    }
    const auto* stable_delegate_ = utils::LoadDelegateFromSharedLibrary(
        stable_delegate_loader_settings->delegate_path()->str());
    if (!stable_delegate_) {
      TFLITE_LOG(ERROR) << "Failed to load stable delegate plugin symbol from "
                        << stable_delegate_loader_settings->delegate_path();
      return;
    }
    stable_delegate_plugin_ = stable_delegate_->delegate_plugin;
    TFLITE_LOG(INFO)
        << "The stable delegate plugin has loaded delegate plugin for "
        << stable_delegate_->delegate_name;
  }

  TfLiteDelegatePtr Create() override {
    return TfLiteDelegatePtr(
        stable_delegate_plugin_->create(GetTFLiteSettings()),
        stable_delegate_plugin_->destroy);
  }

  int GetDelegateErrno(TfLiteOpaqueDelegate* from_delegate) override {
    return stable_delegate_plugin_->get_delegate_errno(from_delegate);
  }

 private:
  const TFLiteSettings* GetTFLiteSettings() {
    return flatbuffers::GetRoot<TFLiteSettings>(
        tflite_settings_builder_.GetBufferPointer());
  }

  const TfLiteOpaqueDelegatePlugin* stable_delegate_plugin_;
  flatbuffers::FlatBufferBuilder tflite_settings_builder_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_PLUGIN_H_
