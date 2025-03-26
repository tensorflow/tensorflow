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
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"

#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

tfrt::AsyncValueRef<KernelFallbackTensor> TransferTensorToDevice(
    const tfrt::ExecutionContext& exec_ctx, const KernelFallbackTensor& tensor,
    const tfrt::Device& src_device, const tfrt::Device& dst_device) {
  const tensorflow::Tensor& src = *tensor.GetTensor();

  auto expected_src = GetTfDevice(exec_ctx, src_device);
  if (!expected_src) {
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(expected_src.takeError()));
  }
  auto expected_dst = GetTfDevice(exec_ctx, dst_device);
  if (!expected_dst) {
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(expected_dst.takeError()));
  }
  tensorflow::Device* srcd = expected_src.get();
  tensorflow::Device* dstd = expected_dst.get();

  return TransferTensorToDevice<KernelFallbackTensor>(exec_ctx, src, srcd,
                                                      dstd);
}

llvm::Expected<Device*> GetTfDevice(const tfrt::ExecutionContext& exec_ctx,
                                    const tfrt::Device& device) {
  auto eager_context_expected =
      exec_ctx.resource_context()
          ->GetOrCreateResource<tfd::EagerContextResource>(
              tfd::kEagerContextResourceName)
          ->GetTFEagerContext();
  if (!eager_context_expected) {
    return eager_context_expected.takeError();
  }
  Device* tf_device;
  absl::Status s = eager_context_expected.get()->FindDeviceFromName(
      device.name().data(), &tf_device);
  if (!s.ok()) {
    return tfrt::MakeStringError(s.message());
  }
  return tf_device;
}

}  // namespace tfd
}  // namespace tensorflow
