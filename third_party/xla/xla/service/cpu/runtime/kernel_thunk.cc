/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime/kernel_thunk.h"

#define EIGEN_USE_THREADS

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "llvm/ADT/SmallVector.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<KernelThunk>> KernelThunk::Create(
    Info info, absl::Span<const BufferAllocation::Slice> arguments_buffers,
    absl::Span<const BufferAllocation::Slice> results_buffers,
    std::string kernel_name, se::ThreadDim thread_dim,
    std::optional<uint64_t> min_alignment) {
  if (min_alignment.has_value() && !absl::has_single_bit(*min_alignment)) {
    return Internal("Host kernel %s minimum alignment %d is not a power of 2",
                    info.op_name, *min_alignment);
  }

  return absl::WrapUnique(
      new KernelThunk(std::move(info), arguments_buffers, results_buffers,
                      std::move(kernel_name), thread_dim, min_alignment));
}

KernelThunk::KernelThunk(
    Info info, absl::Span<const BufferAllocation::Slice> arguments_buffers,
    absl::Span<const BufferAllocation::Slice> results_buffers,
    std::string kernel_name, se::ThreadDim thread_dim,
    std::optional<uint64_t> min_alignment)
    : Thunk(Kind::kKernel, std::move(info)),
      arguments_buffers_(arguments_buffers.begin(), arguments_buffers.end()),
      results_buffers_(results_buffers.begin(), results_buffers.end()),
      num_kernel_args_(arguments_buffers.size() + results_buffers.size()),
      kernel_name_(std::move(kernel_name)),
      thread_dim_(thread_dim),
      min_alignment_(min_alignment),
      call_once_(thread_dim_ == se::ThreadDim()),
      kernel_ptr_(nullptr) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> KernelThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  VLOG(3) << absl::StreamFormat(
      "Launch host kernel %s with %d arguments buffers and %d results buffers: "
      "#threads=%s",
      kernel_name_, arguments_buffers_.size(), results_buffers_.size(),
      thread_dim_.ToString());

  // We use `llvm::SmallVector` instead of `absl::InlinedVector` because
  // it allows to resize a vector without zero-initializing storage.
  llvm::SmallVector<SE_HOST_KernelArg, 8> kernel_args;
  kernel_args.resize_for_overwrite(num_kernel_args_);

  SE_HOST_KernelArg* kernel_args_ptr = kernel_args.data();
  const BufferAllocations* allocations = params.buffer_allocations;

  for (BufferAllocation::Slice& buffer : arguments_buffers_) {
    if constexpr (ShouldCheckBufferSlices()) {
      TF_ASSIGN_OR_RETURN(auto mem, allocations->GetDeviceAddress(buffer));
      *kernel_args_ptr++ = SE_HOST_KernelArg{mem.opaque(), mem.size()};
    } else {
      auto mem = allocations->GetDeviceAddressUnchecked(buffer);
      *kernel_args_ptr++ = SE_HOST_KernelArg{mem.opaque(), mem.size()};
    }
  }

  for (BufferAllocation::Slice& buffer : results_buffers_) {
    if constexpr (ShouldCheckBufferSlices()) {
      TF_ASSIGN_OR_RETURN(auto mem, allocations->GetDeviceAddress(buffer));
      *kernel_args_ptr++ = SE_HOST_KernelArg{mem.opaque(), mem.size()};
    } else {
      auto mem = allocations->GetDeviceAddressUnchecked(buffer);
      *kernel_args_ptr++ = SE_HOST_KernelArg{mem.opaque(), mem.size()};
    }
  }

  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(3))) {
    VlogKernelArgs(kernel_args);
  }

  // Сheck that all resolved buffers are properly aligned.
  if constexpr (ShouldCheckBufferSlices()) {
    TF_RETURN_IF_ERROR(CheckBufferAlignment(kernel_args));
  }

  // TODO(ezhulenev): Kernel ptr should be loaded as a part of Thunk
  // initialization stage.
  se::host::HostKernel* kernel = kernel_ptr_.load();

  // Because thunks are owned by a parent CpuExecutable, we can safely assume
  // that kernel pointer will not change after we find it the first time.
  if (ABSL_PREDICT_FALSE(kernel == nullptr)) {
    TF_ASSIGN_OR_RETURN(SE_HOST_Kernel * kernel_fn,
                        params.host_kernels->Find(kernel_name_));

    absl::MutexLock lock(&mutex_);
    kernel_.emplace(num_kernel_args_, kernel_fn, nullptr);
    kernel_ptr_.store(kernel = &kernel_.value());
  }

  // Use a fast path if kernel called just once.
  if (ABSL_PREDICT_TRUE(call_once_)) {
    TF_RETURN_IF_ERROR(kernel->CallOnce(kernel_args));
    return OkExecuteEvent();
  }

  // If intra-op thread pool is not nullptr, we launch HostKernel in async mode
  // by scheduling tasks into it. HostKernel launch completion will
  // automatically signal KernelThunk execute completion.
  if (ABSL_PREDICT_TRUE(params.intra_op_threadpool)) {
    return kernel->Launch(
        thread_dim_, kernel_args, [&params](se::host::HostKernel::Task task) {
          params.intra_op_threadpool->getPool()->Schedule(std::move(task));
        });
  }

  TF_RETURN_IF_ERROR(kernel->Launch(thread_dim_, kernel_args));
  return OkExecuteEvent();
}

absl::Status KernelThunk::CheckBufferAlignment(
    absl::Span<const SE_HOST_KernelArg> kernel_args) {
  if (min_alignment_.has_value()) {
    for (int64_t i = 0; i < num_kernel_args_; ++i) {
      auto ptr = reinterpret_cast<uintptr_t>(kernel_args[i].data);
      if (ABSL_PREDICT_FALSE((ptr & (*min_alignment_ - 1)) != 0)) {
        return Internal(
            "Host kernel %s buffer argument #%d (%p) is not aligned to a "
            "required minimum alignment of %d bytes",
            info().op_name, i, kernel_args[i].data, *min_alignment_);
      }
    }
  }
  return absl::OkStatus();
}

void KernelThunk::VlogKernelArgs(
    absl::Span<const SE_HOST_KernelArg> kernel_args) {
  for (int64_t i = 0; i < arguments_buffers_.size(); ++i) {
    VLOG(3) << absl::StreamFormat("  arg #%d: %s (%p)", i,
                                  arguments_buffers_[i].ToString(),
                                  kernel_args[i].data);
  }
  for (int64_t i = 0; i < results_buffers_.size(); ++i) {
    VLOG(3) << absl::StreamFormat(
        "  res #%d: %s (%p)", i, results_buffers_[i].ToString(),
        kernel_args[arguments_buffers_.size() + i].data);
  }
}

KernelThunk::BufferUses KernelThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const BufferAllocation::Slice& buffer : arguments_buffers_) {
    buffer_uses.emplace_back(buffer, BufferUse::kRead);
  }
  for (const BufferAllocation::Slice& buffer : results_buffers_) {
    buffer_uses.emplace_back(buffer, BufferUse::kWrite);
  }
  return buffer_uses;
}

}  // namespace xla::cpu
