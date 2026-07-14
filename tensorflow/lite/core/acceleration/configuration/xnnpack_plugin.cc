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

#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace delegates {
class XNNPackPlugin : public DelegatePluginInterface {
 public:
  TfLiteDelegatePtr Create() override {
    return TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&options_),
                             TfLiteXNNPackDelegateDelete);
  }
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override { return 0; }
  static std::unique_ptr<DelegatePluginInterface> New(
      const TFLiteSettings& acceleration) {
    return std::make_unique<XNNPackPlugin>(acceleration);
  }
  explicit XNNPackPlugin(const TFLiteSettings& tflite_settings)
      : options_(TfLiteXNNPackDelegateOptionsDefault()) {
    // LINT.IfChange(tflite_settings_to_xnnpack_delegate_options)
    const auto* xnnpack_settings = tflite_settings.xnnpack_settings();
    if (xnnpack_settings) {
      options_.num_threads = xnnpack_settings->num_threads();
      // If xnnpack_settings->flags is zero, then leave options.flags
      // unmodified, i.e. use the default flags (not zero).
      // If xnnpack_settings->flags is nonzero, then use exactly
      // those flags (i.e. discard the default flags).
      if (xnnpack_settings->flags()) {
        options_.flags = xnnpack_settings->flags();
      }
      if (xnnpack_settings->weight_cache_file_path()) {
        options_.weight_cache_file_path =
            xnnpack_settings->weight_cache_file_path()->c_str();
      }
    }
    // LINT.ThenChange(c/xnnpack_plugin.cc:tflite_settings_to_xnnpack_delegate_options)
  }

 private:
  TfLiteXNNPackDelegateOptions options_;
};

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(XNNPackPlugin, XNNPackPlugin::New);

}  // namespace delegates
}  // namespace tflite
