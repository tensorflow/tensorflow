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
#ifndef TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_NNAPI_PLUGIN_H_
#define TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_NNAPI_PLUGIN_H_

// This file provides the NNApiPlugin class, which implements the
// TFLite Delegate Plugin for the NNAPI Delegate.

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace delegates {

class NnapiPlugin : public DelegatePluginInterface {
 public:
  TfLiteDelegatePtr Create() override {
    std::unique_ptr<tflite::StatefulNnApiDelegate> nnapi_delegate = nullptr;
    if (!support_library_handle_) {
      nnapi_delegate =
          std::make_unique<tflite::StatefulNnApiDelegate>(options_);
    } else {
      auto nnapi_support_library_driver =
          reinterpret_cast<const NnApiSLDriverImplFL5*>(
              support_library_handle_);
      nnapi_delegate = std::make_unique<tflite::StatefulNnApiDelegate>(
          nnapi_support_library_driver, options_);
    }
    return TfLiteDelegatePtr(
        nnapi_delegate.release(), [](TfLiteDelegate* delegate) {
          delete static_cast<tflite::StatefulNnApiDelegate*>(delegate);
        });
  }
  int GetDelegateErrno(TfLiteDelegate* from_delegate) override {
    auto nnapi_delegate =
        static_cast<tflite::StatefulNnApiDelegate*>(from_delegate);
    return nnapi_delegate->GetNnApiErrno();
  }
  static std::unique_ptr<NnapiPlugin> New(
      const TFLiteSettings& tflite_settings) {
    return std::make_unique<NnapiPlugin>(tflite_settings);
  }
  explicit NnapiPlugin(const TFLiteSettings& tflite_settings) {
    const NNAPISettings* nnapi_settings = tflite_settings.nnapi_settings();
    if (!nnapi_settings) return;
    if (nnapi_settings->accelerator_name() &&
        nnapi_settings->accelerator_name()->Length() != 0) {
      accelerator_ = nnapi_settings->accelerator_name()->str();
      options_.accelerator_name = accelerator_.c_str();
    }

    SetCompilationCacheDir(tflite_settings);
    SetModelToken(tflite_settings);

    options_.execution_preference =
        ConvertExecutionPrefence(nnapi_settings->execution_preference());
    options_.disallow_nnapi_cpu =
        !nnapi_settings->allow_nnapi_cpu_on_android_10_plus();
    options_.execution_priority =
        ConvertExecutionPriority(nnapi_settings->execution_priority());
    options_.allow_fp16 = nnapi_settings->allow_fp16_precision_for_fp32();
    options_.use_burst_computation = nnapi_settings->use_burst_computation();
    if (tflite_settings.max_delegated_partitions() >= 0) {
      options_.max_number_delegated_partitions =
          tflite_settings.max_delegated_partitions();
    }
    support_library_handle_ = nnapi_settings->support_library_handle();
  }
  const tflite::StatefulNnApiDelegate::Options& Options() { return options_; }
  const int64_t GetSupportLibraryHandle() { return support_library_handle_; }

 private:
  void SetCompilationCacheDir(const TFLiteSettings& tflite_settings) {
    if (tflite_settings.compilation_caching_settings() &&
        tflite_settings.compilation_caching_settings()->cache_dir() &&
        tflite_settings.compilation_caching_settings()->cache_dir()->Length() !=
            0) {
      cache_dir_ =
          tflite_settings.compilation_caching_settings()->cache_dir()->str();
      options_.cache_dir = cache_dir_.c_str();
    } else if (tflite_settings.nnapi_settings() &&
               tflite_settings.nnapi_settings()->cache_directory() &&
               tflite_settings.nnapi_settings()->cache_directory()->Length() !=
                   0) {
      cache_dir_ = tflite_settings.nnapi_settings()->cache_directory()->str();
      options_.cache_dir = cache_dir_.c_str();
    }
  }

  void SetModelToken(const TFLiteSettings& tflite_settings) {
    if (tflite_settings.compilation_caching_settings() &&
        tflite_settings.compilation_caching_settings()->model_token() &&
        tflite_settings.compilation_caching_settings()
                ->model_token()
                ->Length() != 0) {
      model_token_ =
          tflite_settings.compilation_caching_settings()->model_token()->str();
      options_.model_token = model_token_.c_str();
    } else if (tflite_settings.nnapi_settings()->model_token() &&
               tflite_settings.nnapi_settings()->model_token()->Length() != 0) {
      model_token_ = tflite_settings.nnapi_settings()->model_token()->str();
      options_.model_token = model_token_.c_str();
    }
  }

  static inline tflite::StatefulNnApiDelegate::Options::ExecutionPreference
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

  static inline int ConvertExecutionPriority(
      NNAPIExecutionPriority from_compatibility_priority) {
    switch (from_compatibility_priority) {
      case NNAPIExecutionPriority_NNAPI_PRIORITY_LOW:
        return ANEURALNETWORKS_PRIORITY_LOW;
      case NNAPIExecutionPriority_NNAPI_PRIORITY_MEDIUM:
        return ANEURALNETWORKS_PRIORITY_MEDIUM;
      case NNAPIExecutionPriority_NNAPI_PRIORITY_HIGH:
        return ANEURALNETWORKS_PRIORITY_HIGH;
      default:
        return ANEURALNETWORKS_PRIORITY_DEFAULT;
    }
  }

  std::string accelerator_, cache_dir_, model_token_;
  tflite::StatefulNnApiDelegate::Options options_;
  int64_t support_library_handle_ = 0;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_ACCELERATION_CONFIGURATION_NNAPI_PLUGIN_H_
