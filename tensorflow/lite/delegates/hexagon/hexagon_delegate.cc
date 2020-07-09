/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate_kernel.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_implementation.h"
#include "tensorflow/lite/delegates/hexagon/utils.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
// Should be > 0. > 16 causes problems.
constexpr int kMaxHexagonGraphs = 4;
constexpr int kMaxMaxHexagonGraphs = 16;
constexpr int kMinNodesPerHexagonGraph = 2;

class HexagonDelegate : public SimpleDelegateInterface {
 public:
  explicit HexagonDelegate(const TfLiteHexagonDelegateOptions* params)
      : params_(params != nullptr ? *params
                                  : TfLiteHexagonDelegateOptions({0})) {
    if (params_.max_delegated_partitions <= 0) {
      params_.max_delegated_partitions = kMaxHexagonGraphs;
    } else if (params_.max_delegated_partitions > kMaxMaxHexagonGraphs) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "Hexagon delegate: cannot have this many %d partitions, "
                      "and will cap to at most %d partitions.\n",
                      params_.max_delegated_partitions, kMaxMaxHexagonGraphs);
      params_.max_delegated_partitions = kMaxMaxHexagonGraphs;
    }
    if (params_.min_nodes_per_partition <= 0) {
      params_.min_nodes_per_partition = kMinNodesPerHexagonGraph;
    }
  }

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    return IsNodeSupportedByHexagon(registration, node, context);
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override { return "TfLiteHexagonDelegate"; }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<HexagonDelegateKernel>(params_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    auto options = SimpleDelegateInterface::Options();
    options.max_delegated_partitions = params_.max_delegated_partitions;
    options.min_nodes_per_partition = params_.min_nodes_per_partition;
    return options;
  }

 private:
  TfLiteHexagonDelegateOptions params_;
};

}  // namespace
}  // namespace tflite

TfLiteDelegate* TfLiteHexagonDelegateCreate(
    const TfLiteHexagonDelegateOptions* options) {
  // return tflite::CreateDelegate(options);
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::make_unique<tflite::HexagonDelegate>(options));
}

TfLiteHexagonDelegateOptions TfLiteHexagonDelegateOptionsDefault() {
  TfLiteHexagonDelegateOptions result{0};
  return result;
}

void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

void TfLiteHexagonInit() { tflite::HexagonDelegateKernel::InitState(); }

void TfLiteHexagonInitWithPath(const char* lib_directory_path) {
  if (lib_directory_path != nullptr) {
    std::string env_var_value = lib_directory_path;
    env_var_value += ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", env_var_value.c_str(), 1 /* overwrite */);
  }
  tflite::HexagonDelegateKernel::InitState();
}
void TfLiteHexagonTearDown() { tflite::HexagonDelegateKernel::Teardown(); }
