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

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/minimal_logging.h"

// Guarding anyway although this file not expected to be compiled for non-Apple.
#if defined(__APPLE__)
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"

namespace tflite {
namespace delegates {
class CoreMLPlugin : public DelegatePluginInterface {
 public:
  TfLiteDelegatePtr Create() override {
    TfLiteDelegate* delegate_ptr = TfLiteCoreMlDelegateCreate(&options_);
    TfLiteDelegatePtr delegate(delegate_ptr, [](TfLiteDelegate* delegate) {
      TfLiteCoreMlDelegateDelete(delegate);
    });
    return delegate;
  }
  int GetDelegateErrno(TfLiteDelegate* /* from_delegate */) override {
    return 0;
  }
  static std::unique_ptr<CoreMLPlugin> New(
      const TFLiteSettings& tflite_settings) {
    return absl::make_unique<CoreMLPlugin>(tflite_settings);
  }
  explicit CoreMLPlugin(const TFLiteSettings& tflite_settings) {
    const CoreMLSettings* settings = tflite_settings.coreml_settings();
    options_ = TfLiteCoreMlDelegateOptions({});
    // Using the proto defaults if the settings were not set.
    switch (settings->enabled_devices()) {
      case tflite::CoreMLSettings_::EnabledDevices_DEVICES_ALL:
        options_.enabled_devices = TfLiteCoreMlDelegateAllDevices;
        break;
      case tflite::CoreMLSettings_::EnabledDevices_DEVICES_WITH_NEURAL_ENGINE:
        options_.enabled_devices = TfLiteCoreMlDelegateDevicesWithNeuralEngine;
        break;
      default:
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Invalid devices enum: %d",
                        settings->enabled_devices());
    }
    options_.coreml_version = settings->coreml_version();
    options_.max_delegated_partitions = settings->max_delegated_partitions();
    options_.min_nodes_per_partition = settings->min_nodes_per_partition();
  }

 private:
  TfLiteCoreMlDelegateOptions options_;
};

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(CoreMLPlugin, CoreMLPlugin::New);

}  // namespace delegates
}  // namespace tflite

#endif  // __APPLE__
