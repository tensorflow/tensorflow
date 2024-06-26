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

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
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
      kernel_name_(std::move(kernel_name)),
      thread_dim_(thread_dim),
      min_alignment_(min_alignment),
      use_task_runner_(thread_dim != se::ThreadDim()),
      kernel_ptr_(nullptr) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> KernelThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  VLOG(3) << absl::StreamFormat(
      "Launch host kernel %s with %d arguments buffers and %d results buffers: "
      "#threads=%s",
      kernel_name_, arguments_buffers_.size(), results_buffers_.size(),
      thread_dim_.ToString());

  int64_t num_args = arguments_buffers_.size() + results_buffers_.size();
  absl::InlinedVector<SE_HOST_KernelArg, 8> kernel_args(num_args);

  // We initialize `kernel_args` array using pointer to the first argument,
  // because individual elements access adds up measurable overhead, and this
  // code is on the critical path.
  SE_HOST_KernelArg* kernel_args_ptr = kernel_args.data();
  int64_t kernel_arg_idx = 0;

  int64_t arg_num = 0;
  for (BufferAllocation::Slice& buffer : arguments_buffers_) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase arg_data,
                        params.buffer_allocations->GetDeviceAddress(buffer));
    VLOG(3) << absl::StreamFormat("  arg #%d: %s (%p)", arg_num++,
                                  buffer.ToString(), arg_data.opaque());
    kernel_args_ptr[kernel_arg_idx++] =
        SE_HOST_KernelArg{arg_data.opaque(), arg_data.size()};
  }

  int64_t res_num = 0;
  for (BufferAllocation::Slice& buffer : results_buffers_) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase result_data,
                        params.buffer_allocations->GetDeviceAddress(buffer));
    VLOG(3) << absl::StreamFormat("  res #%d: %s (%p)", res_num++,
                                  buffer.ToString(), result_data.opaque());
    kernel_args_ptr[kernel_arg_idx++] =
        SE_HOST_KernelArg{result_data.opaque(), result_data.size()};
  }

  // Check that all buffers are aligned to the minimum alignment. We codegen
  // with the assumption that all buffers are aligned, and if they are not, we
  // will crash with a segmentation fault, or worse, produce incorrect results.
  if (min_alignment_.has_value()) {
    for (int64_t i = 0; i < num_args; ++i) {
      auto ptr = reinterpret_cast<uintptr_t>(kernel_args_ptr[i].data);
      if (ABSL_PREDICT_FALSE((ptr & (*min_alignment_ - 1)) != 0)) {
        return Internal(
            "Host kernel %s buffer argument #%d (%p) is not aligned to a "
            "required minimum alignment of %d bytes",
            info().op_name, i, kernel_args_ptr[i].data, *min_alignment_);
      }
    }
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
    kernel_.emplace(num_args, kernel_fn, nullptr);
    kernel_ptr_.store(kernel = &kernel_.value());
  }

  // If intra-op thread pool is not nullptr, we launch HostKernel in async mode
  // by scheduling tasks into it. HostKernel launch completion will
  // automatically signal KernelThunk execute completion.
  if (ABSL_PREDICT_FALSE(params.intra_op_threadpool && use_task_runner_)) {
    return kernel->Launch(thread_dim_, kernel_args,
                          [&params](se::host::HostKernel::Task task) {
                            params.intra_op_threadpool->getPool()->Schedule(
                                ToCopyableTask(std::move(task)));
                          });
  }

  TF_RETURN_IF_ERROR(kernel->Launch(thread_dim_, kernel_args));
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
