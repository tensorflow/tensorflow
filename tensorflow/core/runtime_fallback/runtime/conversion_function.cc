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

// This file implements conversion function between TFRuntimeFallback and Host
// tensors.

#include "tensorflow/core/runtime_fallback/runtime/conversion_function.h"

#include <utility>

#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_kernels.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_utils.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

tfrt::Expected<tfrt::DenseHostTensor>
ConvertRuntimeFallbackTensorToDenseHostTensor(
    const RuntimeFallbackTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  tensorflow::Status status;
  // Resolve ensures Tensor is on host CPU.
  OwnedAbstractTensorInterface tensor_interface{
      tensor.GetTensorHandle()->Resolve(&status)};
  if (!status.ok())
    return tfrt::MakeStringError("error resolving TensorHandle: ",
                                 status.error_message());

  void *data = tensor_interface->Data();
  size_t size = tensor_interface->ByteSize();
  // `tensor_interface` holds a reference on underlying Tensorflow buffer and is
  // held alive by HostBuffer deallocator lambda capture (a
  // llvm::unique_function), and it gets released when HostBuffer deallocator is
  // called and destroyed.
  auto host_buffer = tfrt::HostBuffer::CreateFromExternal(
      data, size,
      [tensor_interface = std::move(tensor_interface)](void *, size_t) {});
  // Assume HostBuffer::CreateFromExternal never fails.
  return tfrt::DenseHostTensor(tensor.metadata(), std::move(host_buffer));
}

static tfrt::AsyncValueRef<tfrt::StringHostTensor>
ConvertRuntimeFallbackTensorToStringHostTensor(
    const RuntimeFallbackTensor &tensor, const tfrt::Device &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host_ctx = exec_ctx.host();
  tensorflow::Status status;
  // Resolve ensures Tensor is on host CPU.
  OwnedAbstractTensorInterface tensor_interface{
      tensor.GetTensorHandle()->Resolve(&status)};
  if (!status.ok())
    return tfrt::MakeErrorAsyncValueRef(

        tfrt::StrCat("error resolving TensorHandle: ", status.error_message()));

  assert(tensor_interface->Type() == DT_STRING);

  // TODO(tfrt-devs): Consider a more efficient way to pass string
  // tensors between TFRT and TF.
  auto string_host_tensor =
      CopyTfStringTensorToStringHostTensor(tensor_interface.get(), host_ctx);
  if (!string_host_tensor)
    return tfrt::MakeErrorAsyncValueRef(

        tfrt::StrCat(
            "error converting TF string tensor to tfrt::StringHostTensor: ",
            string_host_tensor.takeError()));
  return tfrt::MakeAvailableAsyncValueRef<tfrt::StringHostTensor>(
      std::move(*string_host_tensor));
}

static tfrt::AsyncValueRef<RuntimeFallbackTensor>
ConvertScalarHostTensorToRuntimeFallbackTensor(
    const tfrt::AnyScalarHostTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();

  // The memory copy here is necessary since current TensorFlow doesn't support
  // packed TFRT representations like ScalarHostTensor.
  auto optional_dht =
      tfrt::CopyScalarHostTensorToDenseHostTensor(tensor, exec_ctx);
  if (!optional_dht)
    return tfrt::MakeErrorAsyncValueRef(
        "error copying ScalarHostTensor to DenseHostTensor");

  return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
      CopyRefDHTToRuntimeFallbackTensor(optional_dht.value(), host));
}

static tfrt::AsyncValueRef<RuntimeFallbackTensor>
ConvertDenseHostTensorToRuntimeFallbackTensor(
    const tfrt::DenseHostTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();

  // CopyRef and transfer one HostBuffer reference to RuntimeFallbackTensor.
  return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
      CopyRefDHTToRuntimeFallbackTensor(tensor, host));
}

static tfrt::AsyncValueRef<RuntimeFallbackTensor>
ConvertStringHostTensorToRuntimeFallbackTensor(
    const tfrt::StringHostTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();

  return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
      CopySHTToRuntimeFallbackTensor(tensor, host));
}

static tfrt::Expected<RuntimeFallbackTensor>
TransferRuntimeFallbackToAnotherDevice(const RuntimeFallbackTensor &tensor,
                                       const tfrt::Device &src,
                                       const tfrt::Device &dst,
                                       const tfrt::ExecutionContext &exec_ctx) {
  auto eager_context_resource =
      exec_ctx.resource_context()
          ->GetResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);
  if (!eager_context_resource.has_value())
    return tfrt::MakeStringError(
        "Cannot get EagerContext from ExecutionContext.");
  auto expected_eager_context =
      eager_context_resource.value()->GetTFEagerContext();
  if (!expected_eager_context) return expected_eager_context.takeError();
  auto *eager_context = expected_eager_context.get();

  auto *th = tensor.GetTensorHandle();
  Device *tf_device;
  Status s = eager_context->FindDeviceFromName(dst.name().data(), &tf_device);
  if (!s.ok()) return tfrt::MakeStringError(s.error_message());

  auto *host = exec_ctx.host();

  TensorHandle *result_th;

  s = EagerCopyToDevice(th, eager_context, &eager_context->Executor(),
                        tf_device,
                        /*mirror=*/false, &result_th);
  if (!s.ok()) return tfrt::MakeStringError(s.error_message());
  return CreateRuntimeFallbackTensorFromTfTensorHandle(
      OwnedTensorHandle(result_th), host);
}

void RegisterTFRuntimeFallbackTensorToHostConversionFn(
    tfrt::TensorConversionFnRegistry *registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertRuntimeFallbackTensorToDenseHostTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertRuntimeFallbackTensorToStringHostTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertScalarHostTensorToRuntimeFallbackTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertDenseHostTensorToRuntimeFallbackTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertStringHostTensorToRuntimeFallbackTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(TransferRuntimeFallbackToAnotherDevice));
}

}  // namespace tfd
}  // namespace tensorflow
