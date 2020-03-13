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

#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/delegate_provider.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#if defined(__ANDROID__)
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif

namespace tflite {
namespace benchmark {

class NnapiDelegateProvider : public DelegateProvider {
 public:
  std::vector<Flag> CreateFlags(BenchmarkParams* params) const final;

  void AddParams(BenchmarkParams* params) const final;

  void LogParams(const BenchmarkParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(
      const BenchmarkParams& params) const final;

  std::string GetName() const final { return "NNAPI"; }
};
REGISTER_DELEGATE_PROVIDER(NnapiDelegateProvider);

std::vector<Flag> NnapiDelegateProvider::CreateFlags(
    BenchmarkParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_nnapi", params, "use nnapi delegate api"),
      CreateFlag<std::string>("nnapi_execution_preference", params,
                              "execution preference for nnapi delegate. Should "
                              "be one of the following: fast_single_answer, "
                              "sustained_speed, low_power, undefined"),
      CreateFlag<std::string>(
          "nnapi_accelerator_name", params,
          "the name of the nnapi accelerator to use (requires Android Q+)"),
      CreateFlag<bool>("disable_nnapi_cpu", params,
                       "Disable the NNAPI CPU device")};

  return flags;
}

void NnapiDelegateProvider::AddParams(BenchmarkParams* params) const {
  params->AddParam("use_nnapi", BenchmarkParam::Create<bool>(false));
  params->AddParam("nnapi_execution_preference",
                   BenchmarkParam::Create<std::string>(""));
  params->AddParam("nnapi_accelerator_name",
                   BenchmarkParam::Create<std::string>(""));
  params->AddParam("disable_nnapi_cpu", BenchmarkParam::Create<bool>(false));
}

void NnapiDelegateProvider::LogParams(const BenchmarkParams& params) const {
#if defined(__ANDROID__)
  TFLITE_LOG(INFO) << "Use nnapi : [" << params.Get<bool>("use_nnapi") << "]";
  if (params.Get<bool>("use_nnapi")) {
    if (!params.Get<std::string>("nnapi_execution_preference").empty()) {
      TFLITE_LOG(INFO) << "nnapi execution preference: ["
                       << params.Get<std::string>("nnapi_execution_preference")
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
  }
#endif
}

TfLiteDelegatePtr NnapiDelegateProvider::CreateTfLiteDelegate(
    const BenchmarkParams& params) const {
  TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});
  if (params.Get<bool>("use_nnapi")) {
    StatefulNnApiDelegate::Options options;
    std::string accelerator_name =
        params.Get<std::string>("nnapi_accelerator_name");
    if (!accelerator_name.empty()) {
      options.accelerator_name = accelerator_name.c_str();
    } else if (params.Get<bool>("disable_nnapi_cpu")) {
      options.disallow_nnapi_cpu = true;
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

  return delegate;
}

}  // namespace benchmark
}  // namespace tflite
