// Copyright 2021 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements the XLIR kernels.

#include "llvm/Support/Error.h"
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/ccl_wrapper.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

// Common place for all collective thunks to source nccl/rccl headers.
// Also, all the RunNcclCollective() functions for various thunks should
// use XLA_ENABLE_XCCL to guard use NCCL/RCCL usage (and not use GOOGLE_XCCL).
#if GOOGLE_XCCL
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define XLA_ENABLE_XCCL 1
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // GOOGLE_XCCL

#if XLA_ENABLE_XCCL
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#error "Neither CUDA nor ROCm enabled but NCCL/RCCL enabled"
#endif

// Also include this file required by all collective thunks.
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"

#endif  // XLA_ENABLE_XCCL

namespace xla {
namespace gpu {

tfrt::AsyncValueRef<tfrt::gpu::GpuCclHandle> CclCreate(
    tfrt::Argument<tfrt::gpu::GpuContext> context,
    const tfrt::ExecutionContext& exec_ctx) {
  auto* xccl_ctx = exec_ctx.request_ctx()->GetDataIfExists<XcclContext>();
  if (!xccl_ctx) {
    return tfrt::MakeErrorAsyncValueRef("Failed to get XcclContext");
  }

  auto current = tfrt::gpu::wrapper::CtxSetCurrent(context->get());
  if (!current) {
    return tfrt::MakeErrorAsyncValueRef(llvm::toString(current.takeError()));
  }

  int device_ordinal;
  if (auto error = cudaGetDevice(&device_ordinal)) {
    return tfrt::MakeErrorAsyncValueRef("Failed cudaGetDevice.");
  }

  ncclComm_t comm = xccl_ctx->clique.GetCommForDeviceOrdinal(device_ordinal);
  xccl_ctx->ccl_handle =
      tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuCclHandle>(
          context.ValueRef(),
          tfrt::gpu::wrapper::OwningCclComm({comm, current->platform()}));
  return xccl_ctx->ccl_handle.CopyRef();
}

void RegisterXlirKernels(tfrt::KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("xlir.ccl.create", TFRT_KERNEL(CclCreate));
}

namespace kernels {

TFRT_STATIC_KERNEL_REGISTRATION(RegisterXlirKernels);

}  // namespace kernels
}  // namespace gpu
}  // namespace xla
