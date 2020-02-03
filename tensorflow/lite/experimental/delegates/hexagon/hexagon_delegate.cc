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
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate_kernel.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_implementation.h"
#include "tensorflow/lite/experimental/delegates/hexagon/utils.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
// Should be > 0. > 16 causes problems.
constexpr int kMaxHexagonGraphs = 4;

TfLiteRegistration GetHexagonKernelRegistration() {
  // This is the registration for the Delegate Node that gets added to
  // the TFLite graph instead of the subGraph it replaces it.
  // It is treated as a an OP node. But in our case
  // Init will initialize the delegate
  // Invoke will run the delegate graph.
  // Prepare for prearing the delegate.
  // Free for any cleaning needed by the delegate.
  TfLiteRegistration kernel_registration;
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = "TfLiteHexagonDelegate";
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<HexagonDelegateKernel*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    auto hexagon_kernel = absl::make_unique<HexagonDelegateKernel>();
    if (hexagon_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return hexagon_kernel.release();
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    HexagonDelegateKernel* kernel =
        reinterpret_cast<HexagonDelegateKernel*>(node->user_data);
    if (!kernel) {
      context->ReportError(context, "Hexagon Kernel was not initialized");
      return kTfLiteError;
    }
    return kernel->Invoke(context, node);
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      context->ReportError(context, "Hexagon Kernel was not initialized");
      return kTfLiteError;
    }
    HexagonDelegateKernel* kernel =
        reinterpret_cast<HexagonDelegateKernel*>(node->user_data);
    return kernel->Prepare(context, node);
  };

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Reserve 1 element, since we need first element to be size, will be updated
  // later.
  std::vector<int> supported_nodes(1);
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));
  TfLiteNode* node;
  TfLiteRegistration* registration;

  // Rudimentary mechanism to check how many Hexagon graphs we initialize.
  int num_components = 1;
  int last_index = -1;
  for (int node_index : TfLiteIntArrayView(plan)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (IsNodeSupportedByHexagon(registration, node, context)) {
      // If there is a 'break' in node indices, a new subgraph (and therefore, a
      // new Hexagon graph) will be created.
      if (last_index != -1 && node_index != last_index + 1) {
        if (num_components == kMaxHexagonGraphs) {
          break;
        }
        ++num_components;
      }
      supported_nodes.push_back(node_index);
      last_index = node_index;
    }
  }
  // Set first element to the number of nodes to replace.
  supported_nodes[0] = supported_nodes.size() - 1;
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "Hexagon delegate: %d nodes delegated out of %d nodes.\n",
                  supported_nodes[0], plan->size);
  TfLiteRegistration hexagon_kernel_registration =
      GetHexagonKernelRegistration();

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, hexagon_kernel_registration,
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), delegate);
}

class HexagonDelegate : public TfLiteDelegate {
 public:
  explicit HexagonDelegate(const TfLiteHexagonDelegateOptions* params)
      : params_(params != nullptr ? *params : TfLiteHexagonDelegateOptions()) {}

  TfLiteHexagonDelegateOptions* params() { return &params_; }

  bool VerifyDelegate() {
    auto* hexagon_nn = HexagonNNImplementation();
    if (hexagon_nn == nullptr) {
      return false;
    }
    return hexagon_nn->hexagon_nn_is_device_supported &&
           hexagon_nn->hexagon_nn_is_device_supported();
  }

 private:
  TfLiteHexagonDelegateOptions params_;
};

TfLiteDelegate* CreateDelegate(const TfLiteHexagonDelegateOptions* params) {
  TfLiteDelegate* delegate = new HexagonDelegate(params);
  if (!static_cast<HexagonDelegate*>(delegate)->VerifyDelegate()) {
    delete delegate;
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Hexagon Delegate is not supported.\n");
    return nullptr;
  }

  delegate->data_ = static_cast<HexagonDelegate*>(delegate)->params();
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->Prepare = &DelegatePrepare;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for Hexagon.");

  return delegate;
}

}  // namespace
}  // namespace tflite

TfLiteDelegate* TfLiteHexagonDelegateCreate(
    const TfLiteHexagonDelegateOptions* options) {
  return tflite::CreateDelegate(options);
}

void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate) { delete delegate; }

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
