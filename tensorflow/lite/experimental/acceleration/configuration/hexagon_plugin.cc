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
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"

#if defined(__ARM_ARCH)
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#endif

namespace tflite {
namespace delegates {
class HexagonPlugin : public DelegatePluginInterface {
 public:
  TfLiteDelegatePtr Create() override {
#if defined(__ARM_ARCH)
    TfLiteHexagonInit();
    auto* delegate_ptr = TfLiteHexagonDelegateCreate(&options_);
    TfLiteDelegatePtr delegate(delegate_ptr, [](TfLiteDelegate* delegate) {
      TfLiteHexagonDelegateDelete(delegate);
      TfLiteHexagonTearDown();
    });
    return delegate;
#else   // !defined(__ARM_ARCH)
    return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
#endif  // defined(__ARM_ARCH)
  }
  int GetDelegateErrno(TfLiteDelegate* /* from_delegate */) override {
    return 0;
  }
  static std::unique_ptr<HexagonPlugin> New(
      const TFLiteSettings& tflite_settings) {
    return std::make_unique<HexagonPlugin>(tflite_settings);
  }
  explicit HexagonPlugin(const TFLiteSettings& tflite_settings) {
    const HexagonSettings* settings = tflite_settings.hexagon_settings();
#if defined(__ARM_ARCH)
    options_ = TfLiteHexagonDelegateOptions({0});
    if (settings) {
      options_.debug_level = settings->debug_level();
      options_.powersave_level = settings->powersave_level();
      options_.print_graph_profile = settings->print_graph_profile();
      options_.print_graph_debug = settings->print_graph_debug();
      if (tflite_settings.max_delegated_partitions() >= 0) {
        options_.max_delegated_partitions =
            tflite_settings.max_delegated_partitions();
      }
    }
#else
    (void)settings;
#endif
  }

 private:
#if defined(__ARM_ARCH)
  TfLiteHexagonDelegateOptions options_;
#endif
};

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(HexagonPlugin, HexagonPlugin::New);

}  // namespace delegates
}  // namespace tflite
