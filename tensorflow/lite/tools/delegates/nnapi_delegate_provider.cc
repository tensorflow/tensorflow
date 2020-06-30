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
#include <string>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#if defined(__ANDROID__)
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif

namespace tflite {
namespace tools {

class NnapiDelegateProvider : public DelegateProvider {
 public:
  NnapiDelegateProvider() {
#if defined(__ANDROID__)
    default_params_.AddParam("use_nnapi", ToolParam::Create<bool>(false));
    default_params_.AddParam("nnapi_execution_preference",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("nnapi_execution_priority",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("nnapi_accelerator_name",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("disable_nnapi_cpu",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("nnapi_allow_fp16",
                             ToolParam::Create<bool>(false));
#endif
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;

  std::string GetName() const final { return "NNAPI"; }
};
REGISTER_DELEGATE_PROVIDER(NnapiDelegateProvider);

std::vector<Flag> NnapiDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {
#if defined(__ANDROID__)
    CreateFlag<bool>("use_nnapi", params, "use nnapi delegate api"),
    CreateFlag<std::string>("nnapi_execution_preference", params,
                            "execution preference for nnapi delegate. Should "
                            "be one of the following: fast_single_answer, "
                            "sustained_speed, low_power, undefined"),
    CreateFlag<std::string>("nnapi_execution_priority", params,
                            "The model execution priority in nnapi, and it "
                            "should be one of the following: default, low, "
                            "medium, high."),
    CreateFlag<std::string>(
        "nnapi_accelerator_name", params,
        "the name of the nnapi accelerator to use (requires Android Q+)"),
    CreateFlag<bool>("disable_nnapi_cpu", params,
                     "Disable the NNAPI CPU device"),
    CreateFlag<bool>("nnapi_allow_fp16", params,
                     "Allow fp32 computation to be run in fp16")
#endif
  };

  return flags;
}

void NnapiDelegateProvider::LogParams(const ToolParams& params) const {
#if defined(__ANDROID__)
  TFLITE_LOG(INFO) << "Use nnapi : [" << params.Get<bool>("use_nnapi") << "]";
  if (params.Get<bool>("use_nnapi")) {
    if (!params.Get<std::string>("nnapi_execution_preference").empty()) {
      TFLITE_LOG(INFO) << "nnapi execution preference: ["
                       << params.Get<std::string>("nnapi_execution_preference")
                       << "]";
    }
    if (!params.Get<std::string>("nnapi_execution_priority").empty()) {
      TFLITE_LOG(INFO) << "model execution priority in nnapi: ["
                       << params.Get<std::string>("nnapi_execution_priority")
                       << "]";
    }
    std::string log_string = "nnapi accelerator name: [" +
                             params.Get<std::string>("nnapi_accelerator_name") +
                             "]";
    std::string string_device_names_list = nnapi::GetStringDeviceNamesList();
    // Print available devices when possible
    if (!string_device_names_list.empty()) {
      log_string += " (Available: " + string_device_names_list + ")";
    }
    TFLITE_LOG(INFO) << log_string;
    if (params.Get<bool>("disable_nnapi_cpu")) {
      TFLITE_LOG(INFO) << "disable_nnapi_cpu: ["
                       << params.Get<bool>("disable_nnapi_cpu") << "]";
    }
    if (params.Get<bool>("nnapi_allow_fp16")) {
      TFLITE_LOG(INFO) << "Allow fp16 in NNAPI: ["
                       << params.Get<bool>("nnapi_allow_fp16") << "]";
    }
  }
#endif
}

TfLiteDelegatePtr NnapiDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});
#if defined(__ANDROID__)
  if (params.Get<bool>("use_nnapi")) {
    StatefulNnApiDelegate::Options options;
    std::string accelerator_name =
        params.Get<std::string>("nnapi_accelerator_name");
    if (!accelerator_name.empty()) {
      options.accelerator_name = accelerator_name.c_str();
    } else if (params.Get<bool>("disable_nnapi_cpu")) {
      options.disallow_nnapi_cpu = true;
    }

    if (params.Get<bool>("nnapi_allow_fp16")) {
      options.allow_fp16 = true;
    }

    std::string string_execution_preference =
        params.Get<std::string>("nnapi_execution_preference");
    // Only set execution preference if user explicitly passes one. Otherwise,
    // leave it as whatever NNAPI has as the default.
    if (!string_execution_preference.empty()) {
      tflite::StatefulNnApiDelegate::Options::ExecutionPreference
          execution_preference =
              tflite::StatefulNnApiDelegate::Options::kUndefined;
      if (string_execution_preference == "low_power") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kLowPower;
      } else if (string_execution_preference == "sustained_speed") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
      } else if (string_execution_preference == "fast_single_answer") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kFastSingleAnswer;
      } else if (string_execution_preference == "undefined") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kUndefined;
      } else {
        TFLITE_LOG(WARN) << "The provided value ("
                         << string_execution_preference
                         << ") is not a valid nnapi execution preference.";
      }
      options.execution_preference = execution_preference;
    }

    std::string string_execution_priority =
        params.Get<std::string>("nnapi_execution_priority");
    // Only set execution priority if user explicitly passes one. Otherwise,
    // leave it as whatever NNAPI has as the default.
    if (!string_execution_priority.empty()) {
      int execution_priority = 0;
      if (string_execution_priority == "default") {
        execution_priority = ANEURALNETWORKS_PRIORITY_DEFAULT;
      } else if (string_execution_priority == "low") {
        execution_priority = ANEURALNETWORKS_PRIORITY_LOW;
      } else if (string_execution_priority == "medium") {
        execution_priority = ANEURALNETWORKS_PRIORITY_MEDIUM;
      } else if (string_execution_priority == "high") {
        execution_priority = ANEURALNETWORKS_PRIORITY_HIGH;
      } else {
        TFLITE_LOG(WARN) << "The provided value (" << string_execution_priority
                         << ") is not a valid nnapi execution priority.";
      }
      options.execution_priority = execution_priority;
    }

    int max_delegated_partitions = params.Get<int>("max_delegated_partitions");
    if (max_delegated_partitions > 0) {
      options.max_number_delegated_partitions = max_delegated_partitions;
    }
    delegate = evaluation::CreateNNAPIDelegate(options);
    if (!delegate.get()) {
      TFLITE_LOG(WARN) << "NNAPI acceleration is unsupported on this platform.";
    }
  } else if (!params.Get<std::string>("nnapi_accelerator_name").empty()) {
    TFLITE_LOG(WARN)
        << "`--use_nnapi=true` must be set for the provided NNAPI accelerator ("
        << params.Get<std::string>("nnapi_accelerator_name") << ") to be used.";
  } else if (!params.Get<std::string>("nnapi_execution_preference").empty()) {
    TFLITE_LOG(WARN) << "`--use_nnapi=true` must be set for the provided NNAPI "
                        "execution preference ("
                     << params.Get<std::string>("nnapi_execution_preference")
                     << ") to be used.";
  }
#endif
  return delegate;
}

}  // namespace tools
}  // namespace tflite
