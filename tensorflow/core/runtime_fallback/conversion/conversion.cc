/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements conversion function between RuntimeFallback and
// KernelFallback.

#include "tensorflow/core/runtime_fallback/conversion/conversion.h"

#include <utility>

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_utils.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

static RuntimeFallbackTensor ConvertKernelFallbackToRuntimeFallbackTensor(
    const KernelFallbackTensor &tensor, const tfrt::Device &src,
    const tfrt::Device &dst, const tfrt::ExecutionContext &exec_ctx) {
  assert(&src == &dst);
  auto optional_eager_resource =
      exec_ctx.resource_context()
          ->GetResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);
  assert(optional_eager_resource.has_value());
  auto expected_eager_context =
      optional_eager_resource.value()->GetTFEagerContext();
  assert(expected_eager_context);
  Device *d;
  Status s =
      expected_eager_context.get()->FindDeviceFromName(src.name().data(), &d);
  assert(s.ok());
  Tensor t(*tensor.GetTensor());
  OwnedTensorHandle tensor_handle{tensorflow::TensorHandle::CreateLocalHandle(
      std::move(t),
      /*d=*/t.dtype() == DT_RESOURCE ? expected_eager_context.get()->HostCPU()
                                     : d,
      /*op_device=*/d,
      /*resource_device=*/t.dtype() == DT_RESOURCE ? d : nullptr,
      expected_eager_context.get())};
  return RuntimeFallbackTensor(tensor.shape(), tensor.dtype(),
                               std::move(tensor_handle));
}

static tfrt::AsyncValueRef<KernelFallbackTensor>
ConvertRuntimeFallbackToKernelFallbackTensor(
    const RuntimeFallbackTensor &tensor, const tfrt::Device &src,
    const tfrt::Device &dst, const tfrt::ExecutionContext &exec_ctx) {
  const tensorflow::Tensor *tf_tensor;
  Status s = tensor.GetTensorHandle()->Tensor(&tf_tensor);
  if (!s.ok()) {
    return tfrt::MakeErrorAsyncValueRef(s.message());
  }
  auto src_knfb_tensor =
      KernelFallbackTensor(tensor.shape(), tensor.dtype(), *tf_tensor);
  return TransferTensorToDevice(exec_ctx, src_knfb_tensor, src, dst);
}

void RegisterRuntimeFallbackTensorToKernelFallbackConversionFn(
    tfrt::TensorConversionFnRegistry *registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertKernelFallbackToRuntimeFallbackTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertRuntimeFallbackToKernelFallbackTensor));
}

}  // namespace tfd
}  // namespace tensorflow
