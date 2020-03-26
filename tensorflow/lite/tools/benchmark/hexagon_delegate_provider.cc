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

#if (defined(ANDROID) || defined(__ANDROID__)) && \
    (defined(__arm__) || defined(__aarch64__))
#define TFLITE_ENABLE_HEXAGON
#endif

namespace tflite {
namespace benchmark {

class HexagonDelegateProvider : public DelegateProvider {
 public:
  std::vector<Flag> CreateFlags(BenchmarkParams* params) const final;

  void AddParams(BenchmarkParams* params) const final;

  void LogParams(const BenchmarkParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(
      const BenchmarkParams& params) const final;

  std::string GetName() const final { return "Hexagon"; }
};
REGISTER_DELEGATE_PROVIDER(HexagonDelegateProvider);

std::vector<Flag> HexagonDelegateProvider::CreateFlags(
    BenchmarkParams* params) const {
#if defined(TFLITE_ENABLE_HEXAGON)
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_hexagon", params, "Use Hexagon delegate"),
      CreateFlag<std::string>(
          "hexagon_lib_path", params,
          "The library path for the underlying Hexagon libraries."),
      CreateFlag<bool>("hexagon_profiling", params,
                       "Enables Hexagon profiling")};
  return flags;
#else
  return {};
#endif
}

void HexagonDelegateProvider::AddParams(BenchmarkParams* params) const {
#if defined(TFLITE_ENABLE_HEXAGON)
  params->AddParam("use_hexagon", BenchmarkParam::Create<bool>(false));
  params->AddParam("hexagon_lib_path",
                   BenchmarkParam::Create<std::string>("/data/local/tmp"));
  params->AddParam("hexagon_profiling", BenchmarkParam::Create<bool>(false));
#endif
}

void HexagonDelegateProvider::LogParams(const BenchmarkParams& params) const {
#if defined(TFLITE_ENABLE_HEXAGON)
  TFLITE_LOG(INFO) << "Use Hexagon : [" << params.Get<bool>("use_hexagon")
                   << "]";
  TFLITE_LOG(INFO) << "Hexagon lib path : ["
                   << params.Get<std::string>("hexagon_lib_path") << "]";
  TFLITE_LOG(INFO) << "Hexagon Profiling : ["
                   << params.Get<bool>("hexagon_profiling") << "]";
#endif
}

TfLiteDelegatePtr HexagonDelegateProvider::CreateTfLiteDelegate(
    const BenchmarkParams& params) const {
  TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});
#if defined(TFLITE_ENABLE_HEXAGON)
  if (params.Get<bool>("use_hexagon")) {
    delegate = evaluation::CreateHexagonDelegate(
        params.Get<std::string>("hexagon_lib_path"),
        params.Get<bool>("hexagon_profiling"));

    if (!delegate.get()) {
      TFLITE_LOG(WARN)
          << "Could not create Hexagon delegate: platform may not support "
             "delegate or required libraries are missing";
    }
  }
#endif
  return delegate;
}

}  // namespace benchmark
}  // namespace tflite
