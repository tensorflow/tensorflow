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
#ifndef TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_PLUGIN_H_
#define TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_PLUGIN_H_

// This file provides the StableDelegatePlugin class, which implements the
// TFLite Delegate Plugin Interface for the stable delegates.

#include <memory>
#include <string>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
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
    const StableDelegateLoaderSettings* stable_delegate_loader_settings =
        tflite_settings.stable_delegate_loader_settings();
    if (!stable_delegate_loader_settings ||
        !stable_delegate_loader_settings->delegate_path() ||
        stable_delegate_loader_settings->delegate_path()->size() == 0) {
      TFLITE_LOG(ERROR) << "The delegate path field is not available from the "
                           "provided stable delegate loader settings.";
      return;
    }
    // Creates a copy of TFLiteSettings within the stable delegate plugin.
    TFLiteSettingsT tflite_settings_t;
    tflite_settings.UnPackTo(&tflite_settings_t);
    tflite_settings_builder_.Finish(
        CreateTFLiteSettings(tflite_settings_builder_, &tflite_settings_t));

    const TfLiteStableDelegate* stable_delegate =
        utils::LoadDelegateFromSharedLibrary(
            stable_delegate_loader_settings->delegate_path()->str());
    if (!stable_delegate) {
      TFLITE_LOG(ERROR)
          << "Failed to load stable delegate plugin symbol from "
          << stable_delegate_loader_settings->delegate_path()->str();
      return;
    }
    if (!stable_delegate->delegate_plugin || !stable_delegate->delegate_name) {
      TFLITE_LOG(ERROR)
          << "Invalid stable delegate struct loaded from "
          << stable_delegate_loader_settings->delegate_path()->str();
      return;
    }
    stable_delegate_plugin_ = stable_delegate->delegate_plugin;
    TFLITE_LOG(INFO)
        << "The stable delegate plugin has loaded delegate plugin for "
        << stable_delegate->delegate_name;
  }

  TfLiteDelegatePtr Create() override {
    if (!stable_delegate_plugin_ || !stable_delegate_plugin_->create ||
        !stable_delegate_plugin_->destroy) {
      return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
    }
    return TfLiteDelegatePtr(
        reinterpret_cast<TfLiteDelegate*>(
            stable_delegate_plugin_->create(GetTFLiteSettings())),
        reinterpret_cast<void (*)(TfLiteDelegate*)>(
            stable_delegate_plugin_->destroy));
  }

  int GetDelegateErrno(TfLiteDelegate* from_delegate) override {
    if (!stable_delegate_plugin_ ||
        !stable_delegate_plugin_->get_delegate_errno) {
      return 0;
    }
    return stable_delegate_plugin_->get_delegate_errno(
        reinterpret_cast<TfLiteOpaqueDelegate*>(from_delegate));
  }

 private:
  const TFLiteSettings* GetTFLiteSettings() const {
    return flatbuffers::GetRoot<TFLiteSettings>(
        tflite_settings_builder_.GetBufferPointer());
  }

  const TfLiteOpaqueDelegatePlugin* stable_delegate_plugin_ = nullptr;
  flatbuffers::FlatBufferBuilder tflite_settings_builder_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_PLUGIN_H_
