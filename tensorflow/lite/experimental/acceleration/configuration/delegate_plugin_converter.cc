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
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_plugin_converter.h"

#include <functional>

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/shims/c/common.h"

namespace tflite {
namespace delegates {

using ::tflite_shims::delegates::DelegatePluginInterface;
using ::tflite_shims::delegates::TfLiteOpaqueDelegatePtr;

// This class implements the C++ DelegatePluginInterface using
// the equivalent C API, which is the TfLiteDelegatePlugin struct.
class DelegatePluginViaCApi : public DelegatePluginInterface {
 public:
  explicit DelegatePluginViaCApi(const TfLiteOpaqueDelegatePlugin& plugin_c_api,
                                 const ::tflite::TFLiteSettings& settings)
      : plugin_c_api_(plugin_c_api), tflite_settings_(settings) {}
  TfLiteOpaqueDelegatePtr Create() override {
    return TfLiteOpaqueDelegatePtr(plugin_c_api_.create(&tflite_settings_),
                                   plugin_c_api_.destroy);
  }

  int GetDelegateErrno(TfLiteOpaqueDelegate* from_delegate) override {
    return plugin_c_api_.get_delegate_errno(from_delegate);
  }

 private:
  TfLiteOpaqueDelegatePlugin plugin_c_api_;
  const ::tflite::TFLiteSettings& tflite_settings_;
};

std::function<
    std::unique_ptr<DelegatePluginInterface>(const ::tflite::TFLiteSettings&)>
DelegatePluginConverter(const TfLiteOpaqueDelegatePlugin& plugin_c_api) {
  return [plugin_c_api](const ::tflite::TFLiteSettings& settings)
             -> std::unique_ptr<DelegatePluginInterface> {
    return absl::make_unique<DelegatePluginViaCApi>(plugin_c_api, settings);
  };
}

}  // namespace delegates
}  // namespace tflite
