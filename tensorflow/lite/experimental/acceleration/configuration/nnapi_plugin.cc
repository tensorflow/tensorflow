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
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"

namespace tflite {
namespace delegates {

inline tflite::StatefulNnApiDelegate::Options::ExecutionPreference
ConvertExecutionPrefence(
    NNAPIExecutionPreference from_compatibility_preference) {
  using TflitePreference =
      tflite::StatefulNnApiDelegate::Options::ExecutionPreference;
  switch (from_compatibility_preference) {
    case NNAPIExecutionPreference_NNAPI_LOW_POWER:
      return TflitePreference::kLowPower;
    case NNAPIExecutionPreference_NNAPI_FAST_SINGLE_ANSWER:
      return TflitePreference::kFastSingleAnswer;
    case NNAPIExecutionPreference_NNAPI_SUSTAINED_SPEED:
      return TflitePreference::kSustainedSpeed;
    default:
      return TflitePreference::kUndefined;
  }
}

class NnapiPlugin : public DelegatePluginInterface {
 public:
  TfLiteDelegatePtr Create() override {
    auto nnapi_delegate =
        absl::make_unique<tflite::StatefulNnApiDelegate>(options_);
    return TfLiteDelegatePtr(
        nnapi_delegate.release(), [](TfLiteDelegate* delegate) {
          delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(delegate);
        });
  }
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override {
    auto nnapi_delegate =
        reinterpret_cast<tflite::StatefulNnApiDelegate*>(from_delegate);
    return nnapi_delegate->GetNnApiErrno();
  }
  static std::unique_ptr<NnapiPlugin> New(
      const TFLiteSettings& tflite_settings) {
    return absl::make_unique<NnapiPlugin>(tflite_settings);
  }
  explicit NnapiPlugin(const TFLiteSettings& tflite_settings) {
    const NNAPISettings* nnapi_settings = tflite_settings.nnapi_settings();
    if (!nnapi_settings) return;
    if (nnapi_settings->accelerator_name() &&
        nnapi_settings->accelerator_name()->Length() != 0) {
      accelerator_ = nnapi_settings->accelerator_name()->str();
      options_.accelerator_name = accelerator_.c_str();
    }
    if (nnapi_settings->cache_directory() &&
        nnapi_settings->cache_directory()->Length() != 0) {
      cache_dir_ = nnapi_settings->cache_directory()->str();
      options_.cache_dir = cache_dir_.c_str();
    }
    if (nnapi_settings->model_token() &&
        nnapi_settings->model_token()->Length() != 0) {
      model_token_ = nnapi_settings->model_token()->str();
      options_.model_token = model_token_.c_str();
    }
    options_.execution_preference =
        ConvertExecutionPrefence(nnapi_settings->execution_preference());
    options_.disallow_nnapi_cpu =
        !nnapi_settings->allow_nnapi_cpu_on_android_10_plus();
  }

 private:
  std::string accelerator_, cache_dir_, model_token_;
  tflite::StatefulNnApiDelegate::Options options_;
};

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(NnapiPlugin, NnapiPlugin::New);

}  // namespace delegates
}  // namespace tflite
