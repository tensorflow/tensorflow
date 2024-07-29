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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/task.h"
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
    std::optional<int64_t> min_alignment) {
  return absl::WrapUnique(
      new KernelThunk(std::move(info), arguments_buffers, results_buffers,
                      std::move(kernel_name), thread_dim, min_alignment));
}

KernelThunk::KernelThunk(
    Info info, absl::Span<const BufferAllocation::Slice> arguments_buffers,
    absl::Span<const BufferAllocation::Slice> results_buffers,
    std::string kernel_name, se::ThreadDim thread_dim,
    std::optional<int64_t> min_alignment)
    : Thunk(Kind::kKernel, std::move(info)),
      arguments_buffers_(arguments_buffers.begin(), arguments_buffers.end()),
      results_buffers_(results_buffers.begin(), results_buffers.end()),
      kernel_name_(std::move(kernel_name)),
      thread_dim_(thread_dim),
      min_alignment_(min_alignment),
      kernel_ptr_(nullptr) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> KernelThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  VLOG(3) << absl::StreamFormat(
      "Launch host kernel %s with %d arguments buffers and %d results buffers: "
      "#threads=%s",
      kernel_name_, arguments_buffers_.size(), results_buffers_.size(),
      thread_dim_.ToString());

  absl::InlinedVector<se::DeviceMemoryBase, 8> buffers_data;
  buffers_data.reserve(arguments_buffers_.size() + results_buffers_.size());

  int64_t arg_num = 0;
  for (BufferAllocation::Slice& buffer : arguments_buffers_) {
    TF_ASSIGN_OR_RETURN(buffers_data.emplace_back(),
                        params.buffer_allocations->GetDeviceAddress(buffer));
    VLOG(3) << absl::StreamFormat("  arg #%d: %s (%p)", arg_num++,
                                  buffer.ToString(),
                                  buffers_data.back().opaque());
  }

  int64_t res_num = 0;
  for (BufferAllocation::Slice& buffer : results_buffers_) {
    TF_ASSIGN_OR_RETURN(buffers_data.emplace_back(),
                        params.buffer_allocations->GetDeviceAddress(buffer));
    VLOG(3) << absl::StreamFormat("  res #%d: %s (%p)", res_num++,
                                  buffer.ToString(),
                                  buffers_data.back().opaque());
  }

  // Check that all buffers are aligned to the minimum alignment. We codegen
  // with the assumption that all buffers are aligned, and if they are not, we
  // will crash with a segmentation fault, or worse, produce incorrect results.
  if (min_alignment_.has_value()) {
    for (int64_t i = 0; i < buffers_data.size(); ++i) {
      se::DeviceMemoryBase& data = buffers_data[i];
      if (reinterpret_cast<uintptr_t>(data.opaque()) % *min_alignment_ != 0) {
        return Internal(
            "Host kernel %s buffer argument #%d (%p) is not aligned to a "
            "required minimum alignment of %d bytes",
            info().op_name, i, data.opaque(), *min_alignment_);
      }
    }
  }

  // TODO(ezhulenev): Kernel ptr should be loaded as a part of Thunk
  // initialization stage.
  SE_HOST_Kernel* kernel_ptr = kernel_ptr_.load();

  // Because thunks are owned by a parent CpuExecutable, we can safely assume
  // that kernel pointer will not change after we find it the first time.
  if (kernel_ptr == nullptr) {
    TF_ASSIGN_OR_RETURN(kernel_ptr, params.host_kernels->Find(kernel_name_));
    kernel_ptr_.store(kernel_ptr);
  }

  se::host::HostKernel kernel(buffers_data.size(), kernel_ptr, nullptr);

  // If intra-op thread pool is not nullptr, we launch HostKernel in async mode
  // by scheduling tasks into it. HostKernel launch completion will
  // automatically signal KernelThunk execute completion.
  if (params.intra_op_threadpool) {
    return kernel.Launch(thread_dim_, buffers_data,
                         [&params](se::host::HostKernel::Task task) {
                           params.intra_op_threadpool->getPool()->Schedule(
                               ToCopyableTask(std::move(task)));
                         });
  }

  TF_RETURN_IF_ERROR(kernel.Launch(thread_dim_, buffers_data));
  return OkExecuteEvent();
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
