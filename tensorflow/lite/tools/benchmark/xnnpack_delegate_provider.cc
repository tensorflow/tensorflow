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

namespace tflite {
namespace benchmark {

class XnnpackDelegateProvider : public DelegateProvider {
 public:
  std::vector<Flag> CreateFlags(BenchmarkParams* params) const final;

  void AddParams(BenchmarkParams* params) const final;

  void LogParams(const BenchmarkParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(
      const BenchmarkParams& params) const final;

  std::string GetName() const final { return "XNNPACK"; }
};
REGISTER_DELEGATE_PROVIDER(XnnpackDelegateProvider);

std::vector<Flag> XnnpackDelegateProvider::CreateFlags(
    BenchmarkParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_xnnpack", params, "use XNNPack")};
  return flags;
}

void XnnpackDelegateProvider::AddParams(BenchmarkParams* params) const {
  params->AddParam("use_xnnpack", BenchmarkParam::Create<bool>(false));
}

void XnnpackDelegateProvider::LogParams(const BenchmarkParams& params) const {
  TFLITE_LOG(INFO) << "Use xnnpack : [" << params.Get<bool>("use_xnnpack")
                   << "]";
}

TfLiteDelegatePtr XnnpackDelegateProvider::CreateTfLiteDelegate(
    const BenchmarkParams& params) const {
  if (params.Get<bool>("use_xnnpack")) {
    return evaluation::CreateXNNPACKDelegate(
        params.Get<int32_t>("num_threads"));
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

}  // namespace benchmark
}  // namespace tflite
