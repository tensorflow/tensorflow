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
#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/gpu/kernels/kernels_detail.h"  // from @tf_runtime
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

static llvm::Expected<tfrt::gpu::GpuModule> ModuleLoad(
    tfrt::Argument<tfrt::gpu::GpuContext> context,
    const tfrt::ExecutionContext& exec_ctx) {
  const GpuModuleData* gpu_module_data =
      exec_ctx.request_ctx()->GetDataIfExists<GpuModuleData>();

  if (gpu_module_data == nullptr) {
    return tfrt::MakeStringError(
        "No GpuModuleData resource found in the request context.");
  }
  llvm::StringRef blob = gpu_module_data->blob;

  if (blob.empty() || blob.back() != 0)
    return tfrt::MakeStringError("blob must be null-terminated");

  auto current = tfrt::gpu::wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();

  auto module = tfrt::gpu::wrapper::ModuleLoadData(*current, blob.data());
  if (!module) return module.takeError();

  // Resolve constants.
  for (const auto& constant : gpu_module_data->constants) {
    if (constant.content.empty()) continue;

    auto global = tfrt::gpu::wrapper::ModuleGetGlobal(
        module->get(), constant.symbol_name.data());
    if (!global) return global.takeError();

    const void* constant_content =
        static_cast<const void*>(constant.content.data());
    tfrt::gpu::GpuPointer constant_content_ptr(
        const_cast<void*>(constant_content), current->platform());

    if (auto error = tfrt::gpu::wrapper::MemcpyAsync(
            *current, global->base, constant_content_ptr, global->size_bytes,
            tfrt::gpu::wrapper::Stream(nullptr, current->platform()))) {
      return error;
    }
  }
  return tfrt::gpu::GpuModule(context.ValueRef(), std::move(*module));
}

#if XLA_ENABLE_XCCL
static tfrt::AsyncValueRef<tfrt::gpu::GpuCclHandle> CclCreate(
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
  auto ccl_comm = tfrt::gpu::wrapper::CclComm(comm, current->platform());

  xccl_ctx->ccl_handle =
      tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuCclHandle>(
          context.ValueRef(),
          tfrt::gpu::wrapper::OwningCclComm({comm, current->platform()}));
  return xccl_ctx->ccl_handle.CopyRef();
}

static tfrt::AsyncValueRef<tfrt::Chain> CclCollectivePermute(
    tfrt::Argument<tfrt::gpu::GpuCclHandle> handle,
    tfrt::Argument<tfrt::gpu::GpuBuffer> input,
    tfrt::Argument<tfrt::gpu::GpuBuffer> output,
    // Needs to be sorted alphabetically by attribute name!
    tfrt::Attribute<int32_t> data_type,
    const tfrt::ExecutionContext& exec_ctx) {
  auto* xccl_ctx = exec_ctx.request_ctx()->GetDataIfExists<XcclContext>();
  if (!xccl_ctx) {
    return tfrt::MakeErrorAsyncValueRef("Failed to get XcclContext");
  }
  const absl::optional<int64_t>& source_peer =
      xccl_ctx->collective_permute_source_target.source_peer;
  const absl::optional<int64_t>& target_peer =
      xccl_ctx->collective_permute_source_target.target_peer;

  auto type = static_cast<ncclDataType_t>(*data_type);
  auto width = tfrt::gpu::wrapper::GetCclDataTypeSizeBytes(type);
  if (!width)
    return tfrt::MakeErrorAsyncValueRef(llvm::toString(width.takeError()));
  assert(*width != 0);

  if (target_peer) {
    handle->AddCallback(
        [input = input.ValueRef(), count = input->size() / *width, type,
         peer = *target_peer](tfrt::gpu::wrapper::CurrentContext current,
                              tfrt::gpu::wrapper::Stream stream,
                              tfrt::gpu::wrapper::CclComm comm) -> llvm::Error {
          return tfrt::gpu::wrapper::CclSend(current, input->pointer(), count,
                                             type, peer, comm, stream);
        });
  }

  if (source_peer) {
    handle->AddCallback(
        [output = output.ValueRef(), count = output->size() / *width, type,
         peer = *source_peer](tfrt::gpu::wrapper::CurrentContext current,
                              tfrt::gpu::wrapper::Stream stream,
                              tfrt::gpu::wrapper::CclComm comm) -> llvm::Error {
          return tfrt::gpu::wrapper::CclRecv(current, output->pointer(), count,
                                             type, peer, comm, stream);
        });
  } else {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    handle->AddCallback(
        [output = output.ValueRef(), count = output->size() / *width, type,
         peer = *source_peer](tfrt::gpu::wrapper::CurrentContext current,
                              tfrt::gpu::wrapper::Stream stream,
                              tfrt::gpu::wrapper::CclComm comm) -> llvm::Error {
          return tfrt::gpu::wrapper::MemsetD8Async(current, output->pointer(),
                                                   0, output->size(), stream);
        });
  }

  return tfrt::MakeAvailableAsyncValueRef<tfrt::Chain>();
}
#endif  // XLA_ENABLE_XCCL

static llvm::Error CustomCall(
    const tfrt::gpu::GpuStream& stream, tfrt::Chain chain,
    tfrt::RemainingArguments buffers,
    // Needs to be sorted alphabetically by attribute name!
    tfrt::ArrayAttr args_to_target_args, tfrt::StringAttribute opaque,
    tfrt::ArrayAttr results_to_target_results,
    tfrt::Attribute<int64_t> target_args_count,
    tfrt::Attribute<int64_t> target_results_count,
    const tfrt::ExecutionContext& exec_ctx) {
  auto* custom_call_ctx =
      exec_ctx.request_ctx()->GetDataIfExists<CustomCallContext>();
  if (!custom_call_ctx) {
    return tfrt::MakeStringError("Failed to get CustomCallContext.");
  }

  auto current = tfrt::gpu::wrapper::CtxSetCurrent(stream.context()->get());
  if (!current) {
    return tfrt::MakeStringError(llvm::toString(current.takeError()));
  }

  const int64_t total_target_params_count =
      *target_args_count + *target_results_count;
  llvm::SmallVector<void*, 16> target_buffers;
  target_buffers.reserve(total_target_params_count);

  if (args_to_target_args.GetNumElements() == 0 &&
      results_to_target_results.GetNumElements() == 0) {
    assert(buffers.size() == total_target_params_count);
    for (const auto& arg : buffers.values()) {
      assert(arg->IsType<tfrt::gpu::GpuBuffer>());
      auto pointer = arg->get<tfrt::gpu::GpuBuffer>().pointer();
      target_buffers.push_back(pointer.raw(current->platform()));
    }
  } else {
    std::fill_n(std::back_inserter(target_buffers), total_target_params_count,
                nullptr);
    tfrt::AsyncValue* buffer_ptr = buffers[0];
    for (auto target_index : args_to_target_args.GetValue<int64_t>())
      target_buffers[target_index] = buffer_ptr++;
    for (auto target_index : results_to_target_results.GetValue<int64_t>())
      target_buffers[*target_args_count + target_index] = buffer_ptr++;
  }

  XlaCustomCallStatus custom_call_status;
  custom_call_ctx->call_target(stream.get(), target_buffers.data(),
                               opaque.get().data(), opaque.get().size(),
                               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return tfrt::MakeStringError(absl::StrCat("CustomCall failed: ", *message));
  }
  return llvm::Error::success();
}

static void RegisterXlirKernels(tfrt::KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("xlir.module.load", TFRT_KERNEL(ModuleLoad));
  kernel_reg->AddKernel("xlir.custom_call",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CustomCall));
#if XLA_ENABLE_XCCL
  kernel_reg->AddKernel("xlir.ccl.create", TFRT_KERNEL(CclCreate));
  kernel_reg->AddKernel("xlir.ccl.collective_permute",
                        TFRT_KERNEL(CclCollectivePermute));
#endif  // XLA_ENABLE_XCCL
}

TFRT_STATIC_KERNEL_REGISTRATION(RegisterXlirKernels);

}  // namespace gpu
}  // namespace xla
