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

#include "tensorflow/core/runtime_fallback/kernel/conversion/conversion.h"

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

#include <cstdint>
#include <utility>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_utils.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
using tfrt::DenseHostTensor;

static tfrt::AsyncValueRef<tfrt::StringHostTensor>
ConvertKernelFallbackTensorToStringHostTensor(
    const KernelFallbackTensor& tensor, const tfrt::Device& src,
    const tfrt::CpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto dst_knfb_tensor = TransferTensorToDevice(exec_ctx, tensor, src, dst);
  auto dst_knfb_tensor_ptr = dst_knfb_tensor.AsPtr();
  auto result = tfrt::MakeUnconstructedAsyncValueRef<tfrt::StringHostTensor>();
  dst_knfb_tensor_ptr.AndThen([dst_knfb_tensor = std::move(dst_knfb_tensor),
                               result = result.CopyRef(), exec_ctx]() {
    if (dst_knfb_tensor.IsError()) {
      result.SetError(dst_knfb_tensor.GetError());
      return;
    }
    auto* host = exec_ctx.host();
    assert(!IsUnsupported(dst_knfb_tensor->metadata().dtype) &&
           "Unsupported dtype");
    const auto* tf_tensor = dst_knfb_tensor->GetTensor();
    assert(tf_tensor->dtype() == DT_STRING && "dtype is not DT_STRING");

    auto sht = tfrt::StringHostTensor::CreateUninitialized(
        tfd::GetTensorMetadata(*tf_tensor), host);
    const int64_t num_elems = tf_tensor->NumElements();
    const tensorflow::tstring* tstrings =
        reinterpret_cast<const tensorflow::tstring*>(tf_tensor->data());

    auto strings = sht->strings();
    for (int i = 0; i < num_elems; ++i) {
      strings[i] = tstrings[i];
    }

    result.emplace(std::move(*sht));
  });
  return result;
}

static tfrt::AsyncValueRef<KernelFallbackTensor>
ConvertStringHostTensorToKernelFallbackTensor(
    const tfrt::StringHostTensor& tensor, const tfrt::CpuDevice& src,
    const tfrt::Device& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto tf_tensor = CopyShtToTfTensor(tensor);
  auto src_knfb_tensor =
      KernelFallbackTensor(tensor.shape(), tensor.dtype(), tf_tensor);
  return TransferTensorToDevice(exec_ctx, src_knfb_tensor, src, dst);
}

static tfrt::AsyncValueRef<tfrt::DenseHostTensor>
ConvertKernelFallbackTensorToDenseHostTensor(
    const KernelFallbackTensor& tensor, const tfrt::Device& src,
    const tfrt::CpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto dst_knfb_tensor = TransferTensorToDevice(exec_ctx, tensor, src, dst);
  auto dst_knfb_tensor_ptr = dst_knfb_tensor.AsPtr();
  auto result = tfrt::MakeUnconstructedAsyncValueRef<tfrt::DenseHostTensor>();

  dst_knfb_tensor_ptr.AndThen([dst_knfb_tensor = std::move(dst_knfb_tensor),
                               result = result.CopyRef(), exec_ctx]() {
    if (dst_knfb_tensor.IsError()) {
      result.SetError(dst_knfb_tensor.GetError());
      return;
    }
    const auto* tf_tensor = dst_knfb_tensor->GetTensor();
    void* data = tf_tensor->data();
    size_t size = tf_tensor->AllocatedBytes();
    tfrt::RCReference<tfrt::HostBuffer> host_buffer =
        tfrt::HostBuffer::CreateFromExternal(
            data, size, [tensor = std::move(*tf_tensor)](void*, size_t) {});
    // Assume HostBuffer::CreateFromExternal never fails.
    result.emplace(dst_knfb_tensor->metadata(), std::move(host_buffer));
  });
  return result;
}

static tfrt::AsyncValueRef<KernelFallbackTensor>
ConvertDenseHostTensorToKernelFallbackTensor(
    const tfrt::DenseHostTensor& tensor, const tfrt::CpuDevice& src,
    const tfrt::Device& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto tf_tensor =
      MoveHostBufferToTfTensor(tensor.buffer(), tensor.dtype(), tensor.shape());
  KernelFallbackTensor src_knfb_tensor(tensor.shape(), tensor.dtype(),
                                       tf_tensor);
  return TransferTensorToDevice(exec_ctx, src_knfb_tensor, src, dst);
}

static tfrt::AsyncValueRef<KernelFallbackTensor> TransferKernelFallback(
    const KernelFallbackTensor& tensor, const tfrt::Device& src,
    const tfrt::Device& dst, const tfrt::ExecutionContext& exec_ctx) {
  return TransferTensorToDevice(exec_ctx, tensor, src, dst);
}

void RegisterKernelFallbackTensorConversionFn(
    tfrt::TensorConversionFnRegistry* registry) {
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertKernelFallbackTensorToDenseHostTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertStringHostTensorToKernelFallbackTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertKernelFallbackTensorToStringHostTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertDenseHostTensorToKernelFallbackTensor));
  registry->AddTensorConversionFn(TFRT_CONVERSION(TransferKernelFallback));
}

}  // namespace tfd
}  // namespace tensorflow
