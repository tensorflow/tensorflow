/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace {
TfLiteRegistrationExternal* GetDelegateKernelRegistration(
    SimpleOpaqueDelegateInterface* delegate) {
  TfLiteRegistrationExternal* kernel_registration =
      TfLiteRegistrationExternalCreate(kTfLiteBuiltinDelegate, delegate->Name(),
                                       /*version=*/1);

  TfLiteRegistrationExternalSetFree(
      kernel_registration,
      [](TfLiteOpaqueContext* context, void* buffer) -> void {
        delete reinterpret_cast<SimpleOpaqueDelegateInterface*>(buffer);
      });

  TfLiteRegistrationExternalSetInit(
      kernel_registration,
      [](TfLiteOpaqueContext* context, const char* buffer,
         size_t length) -> void* {
        const TfLiteOpaqueDelegateParams* params =
            reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
        if (params == nullptr) {
          return nullptr;
        }
        auto* delegate_data = reinterpret_cast<SimpleOpaqueDelegateInterface*>(
            params->delegate_data);
        std::unique_ptr<SimpleOpaqueDelegateKernelInterface> delegate_kernel(
            delegate_data->CreateDelegateKernelInterface());
        if (delegate_kernel->Init(context, params) != kTfLiteOk) {
          return nullptr;
        }
        return delegate_kernel.release();
      });
  TfLiteRegistrationExternalSetPrepare(
      kernel_registration,
      [](TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        SimpleOpaqueDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleOpaqueDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        return delegate_kernel->Prepare(context, opaque_node);
      });
  TfLiteRegistrationExternalSetInvoke(
      kernel_registration,
      [](TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        SimpleOpaqueDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleOpaqueDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        TFLITE_DCHECK(delegate_kernel != nullptr);
        return delegate_kernel->Eval(context, opaque_node);
      });

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteOpaqueContext* opaque_context,
                             TfLiteOpaqueDelegate* opaque_delegate,
                             void* data) {
  auto* simple_opaque_delegate =
      reinterpret_cast<SimpleOpaqueDelegateInterface*>(data);
  TF_LITE_ENSURE_STATUS(simple_opaque_delegate->Initialize(opaque_context));

  std::vector<int> supported_nodes;
  TfLiteIntArray* execution_plan;
  TF_LITE_ENSURE_STATUS(
      TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
  std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)> plan(
      TfLiteIntArrayCopy(execution_plan), TfLiteIntArrayFree);

  for (int i = 0; i < plan->size; ++i) {
    const int node_id = plan->data[i];

    TfLiteOpaqueNode* opaque_node;
    TfLiteRegistrationExternal* registration_external;
    TfLiteOpaqueContextGetNodeAndRegistration(
        opaque_context, node_id, &opaque_node, &registration_external);

    if (simple_opaque_delegate->IsNodeSupportedByDelegate(
            registration_external, opaque_node, opaque_context)) {
      supported_nodes.push_back(node_id);
    }
  }

  TfLiteRegistrationExternal* delegate_kernel_registration =
      GetDelegateKernelRegistration(simple_opaque_delegate);

  return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
      opaque_context, delegate_kernel_registration,
      BuildTfLiteIntArray(supported_nodes).get(), opaque_delegate);
}
}  // namespace

TfLiteOpaqueDelegate* TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
    std::unique_ptr<SimpleOpaqueDelegateInterface> simple_delegate,
    int64_t flags) {
  if (simple_delegate == nullptr) {
    return {};
  }

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.Prepare = &DelegatePrepare;
  opaque_delegate_builder.flags = flags;
  opaque_delegate_builder.data = simple_delegate.release();

  return TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
}

void TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(
    TfLiteOpaqueDelegate* opaque_delegate) {
  if (!opaque_delegate) return;
  auto* simple_delegate = reinterpret_cast<SimpleOpaqueDelegateInterface*>(
      TfLiteOpaqueDelegateGetData(opaque_delegate));
  delete simple_delegate;
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

}  // namespace tflite
