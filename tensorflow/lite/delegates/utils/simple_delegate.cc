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
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
TfLiteRegistration GetDelegateKernelRegistration(
    SimpleDelegateInterface* delegate) {
  TfLiteRegistration kernel_registration;
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = delegate->name();
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<SimpleDelegateKernelInterface*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    if (params == nullptr) {
      TF_LITE_KERNEL_LOG(context, "NULL TfLiteDelegateParams passed.");
      return nullptr;
    }
    auto* delegate =
        reinterpret_cast<SimpleDelegateInterface*>(params->delegate->data_);
    std::unique_ptr<SimpleDelegateKernelInterface> delegate_kernel(
        delegate->CreateDelegateKernelInterface());
    if (delegate_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return delegate_kernel.release();
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Delegate kernel was not initialized");
      return kTfLiteError;
    }
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    return delegate_kernel->Prepare(context, node);
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    TFLITE_DCHECK(delegate_kernel != nullptr);
    return delegate_kernel->Invoke(context, node);
  };

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context,
                             TfLiteDelegate* base_delegate) {
  auto* delegate =
      reinterpret_cast<SimpleDelegateInterface*>(base_delegate->data_);
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    return delegate->IsNodeSupportedByDelegate(registration, node, context);
  };
  // TODO(b/149484598): Update to have method that gets all supported nodes.
  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  const auto delegate_partitions = helper.GetFirstNLargestPartitions();

  // To avoid creating a new TfLiteIntArray and free it later, we reserve one
  // element to represent TfLiteIntArray.size which is the 1st element of
  // TfLiteIntArray C struct.
  std::vector<int> supported_nodes(1);
  for (const auto partition : delegate_partitions) {
    auto* nodes = partition->nodes_to_replace;
    supported_nodes.insert(supported_nodes.end(), nodes->data,
                           nodes->data + nodes->size);
  }
  // Set first element to the number of nodes to replace.
  supported_nodes[0] = supported_nodes.size() - 1;

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "%s delegate: %d nodes delegated out of %d nodes with "
                  "%d partitions.\n",
                  delegate->name(), supported_nodes[0],
                  helper.num_total_nodes(), delegate_partitions.size());
  TfLiteRegistration delegate_kernel_registration =
      GetDelegateKernelRegistration(delegate);

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, delegate_kernel_registration,
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), base_delegate);
}
}  // namespace

TfLiteDelegate* TfLiteDelegateFactory::CreateSimpleDelegate(
    std::unique_ptr<SimpleDelegateInterface> simple_delegate) {
  if (simple_delegate == nullptr) {
    return nullptr;
  }
  auto delegate = new TfLiteDelegate();
  delegate->Prepare = &DelegatePrepare;
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;
  delegate->data_ = simple_delegate.release();
  return delegate;
}

void TfLiteDelegateFactory::DeleteSimpleDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  SimpleDelegateInterface* simple_delegate =
      reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
  delete simple_delegate;
  delete delegate;
}

}  // namespace tflite
