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

  bool VerifyDelegate() {
    auto* hexagon_nn = HexagonNNImplementation();
    if (hexagon_nn == nullptr) {
      return false;
    }
    if (hexagon_nn->hexagon_nn_version != nullptr &&
        hexagon_nn->hexagon_nn_hexagon_interface_version) {
      int hexagon_nn_version = -1;
      int hexagon_interface_version =
          hexagon_nn->hexagon_nn_hexagon_interface_version();
      if (hexagon_nn->hexagon_nn_version(&hexagon_nn_version) != 0) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "Failed to fetch Hexagon NN version. This might be "
                        "because you're using incompatible versions of "
                        "libhexagon_interface and libhexagon_nn_skel. "
                        "You must use compatible versions. "
                        "Refer to Tensorflow Lite Hexagon Delegate Guide.");
        return false;
      }
      if (hexagon_nn_version != hexagon_interface_version) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_WARNING,
            "Incompatible versions between interface library and "
            "libhexagon_skel %d vs %d. You must use compatible versions. "
            "Refer to Tensorflow Lite Hexagon Delegate Guide.",
            hexagon_interface_version, hexagon_nn_version);
        return false;
      }
    }
    return hexagon_nn->hexagon_nn_is_device_supported &&
           hexagon_nn->hexagon_nn_is_device_supported();
  }

 private:
  TfLiteHexagonDelegateOptions params_;
};

}  // namespace
}  // namespace tflite

TfLiteDelegate* TfLiteHexagonDelegateCreate(
    const TfLiteHexagonDelegateOptions* options) {
  auto hexagon_delegate_interface =
      std::make_unique<tflite::HexagonDelegate>(options);
  if (!hexagon_delegate_interface->VerifyDelegate()) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Hexagon Delegate is not supported.\n");
    return nullptr;
  }
  auto* initialized_delegate =
      tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
          std::move(hexagon_delegate_interface));
  if (options->enable_dynamic_batch_size) {
    initialized_delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  }
  return initialized_delegate;
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
