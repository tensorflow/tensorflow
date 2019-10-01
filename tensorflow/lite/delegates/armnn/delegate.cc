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
#include "tensorflow/lite/delegates/armnn/delegate.h"

#include <string>
#include <vector>

#include "backendsCommon/BackendRegistry.hpp"
#include "backendsCommon/IBackendInternal.hpp"

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/minimal_logging.h"

#include "tensorflow/lite/delegates/armnn/delegate_kernel.h"

namespace tflite {
ArmNNDelegate::ArmNNDelegate() : ArmNNDelegate(Options()) {}

ArmNNDelegate::ArmNNDelegate(Options options)
    : TfLiteDelegate(TfLiteDelegateCreate()), delegate_data_() {
  if (options.backend_name) {
    delegate_data_.backend_name = options.backend_name;
  }
  delegate_data_.enable_tuning = options.enable_tuning;
  delegate_data_.enable_profiling = options.enable_profiling;
  delegate_data_.enable_logging = options.enable_logging;

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for ArmNN.");

  Prepare = DoPrepare;
  CopyFromBufferHandle = DoCopyFromBufferHandle;
  CopyToBufferHandle = DoCopyToBufferHandle;
  FreeBufferHandle = DoFreeBufferHandle;

  data_ = &delegate_data_;
}

const ArmNNDelegate::Options ArmNNDelegate::GetOptions(
    TfLiteDelegate* delegate) {
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  ArmNNDelegate::Options options;
  options.backend_name = delegate_data->backend_name.empty()
                             ? nullptr
                             : delegate_data->backend_name.c_str();
  options.enable_tuning = delegate_data->enable_tuning;
  options.enable_profiling = delegate_data->enable_profiling;
  options.enable_logging = delegate_data->enable_logging;

  return options;
}

TfLiteStatus ArmNNDelegate::DoPrepare(TfLiteContext* context,
                                      TfLiteDelegate* delegate) {
  // Allocate one element in vector already since TensorFlow Lite uses
  // the first value as the number of nodes. The actual value will be set
  // later, after the vector has been filled.
  std::vector<int> supported_nodes(1);
  // We don't care about all nodes_, we only care about ones in the
  // current plan.
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  // Get backends and check if requested backend is present
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  const auto backend_ids = armnn::BackendRegistryInstance().GetBackendIds();
  if (backend_ids.count(delegate_data->backend_name) == 0) {
    return kTfLiteOk;
  }

  // Instantiate backend and extract layer support validator interface
  auto backend_factory =
      armnn::BackendRegistryInstance().GetFactory(delegate_data->backend_name);
  auto backend = backend_factory();
  auto backend_support = backend->GetLayerSupport();
  std::vector<std::string> failures;

  // Check for every node if it is supported
  for (int node_index : TfLiteIntArrayView(plan)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (delegate::arm::ArmNNDelegateKernel::Validate(
            context, registration->builtin_code, registration->version, node,
            backend_support.get(), &failures)) {
      supported_nodes.push_back(node_index);
    }
  }
  // First element in vector must be the number of actual nodes.
  supported_nodes[0] = supported_nodes.size() - 1;

  // If there are no delegated nodes, short-circuit node replacement.
  if (!supported_nodes[0]) {
    return kTfLiteOk;
  }

  // ArmNN Node Delegate Registration
  static const TfLiteRegistration armnn_delegate_kernel = {
      .init = [](TfLiteContext* context, const char* buffer,
                 size_t length) -> void* {
        const TfLiteDelegateParams* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        delegate::arm::ArmNNDelegateKernel* kernel_state =
            new delegate::arm::ArmNNDelegateKernel;
        kernel_state->Init(context, params);
        return kernel_state;
      },

      .free = [](TfLiteContext* context, void* buffer) -> void {
        delete reinterpret_cast<delegate::arm::ArmNNDelegateKernel*>(buffer);
      },

      .prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        delegate::arm::ArmNNDelegateKernel* state =
            reinterpret_cast<delegate::arm::ArmNNDelegateKernel*>(
                node->user_data);
        return state->Prepare(context, node);
      },

      .invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        delegate::arm::ArmNNDelegateKernel* state =
            reinterpret_cast<delegate::arm::ArmNNDelegateKernel*>(
                node->user_data);
        return state->Invoke(context, node);
      },

      .profiling_string = nullptr,
      .builtin_code = kTfLiteBuiltinDelegate,
      .custom_name = "TfLiteArmNnDelegate",
      .version = 1,
  };

  // Request TFLite to partition the graph and make kernels
  // for each independent node sub set a new armnn_delegate_kernel.
  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, armnn_delegate_kernel,
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), delegate);
}

TfLiteStatus ArmNNDelegate::DoCopyFromBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle, TfLiteTensor* tensor) {
  return kTfLiteError;
}

TfLiteStatus ArmNNDelegate::DoCopyToBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle, TfLiteTensor* tensor) {
  return kTfLiteError;
}

void ArmNNDelegate::DoFreeBufferHandle(TfLiteContext* context,
                                       TfLiteDelegate* delegate,
                                       TfLiteBufferHandle* handle) {}
}  // namespace tflite
