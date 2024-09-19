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

#ifndef XLA_BACKENDS_CPU_RUNTIME_KERNEL_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_KERNEL_THUNK_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/host/host_kernel.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Forward declare thunk defined below.
class KernelThunk;

namespace internal {

// If the number of kernel parameters (arguments and results) is unknown at
// compile time, we use this value to indicate that the parameter is dynamic.
inline constexpr int64_t kDynamicKernelParameter = -1;

// A base template for a KernelThunk that can be specialized for a statically
// known number of arguments and results. We go extra mile here to optimize
// host kernel dispatching on the hot execution path to minimize the XLA runtime
// overheads for the smallest HLO modules.
template <int64_t num_arguments = kDynamicKernelParameter,
          int64_t num_results = kDynamicKernelParameter>
class KernelThunk : public Thunk {
 public:
  BufferUses buffer_uses() const final;

 protected:
  tsl::AsyncValueRef<ExecuteEvent> ExecuteInternal(const ExecuteParams& params);

 private:
  friend class ::xla::cpu::KernelThunk;

  static constexpr bool IsDynamic(size_t n) {
    return n == kDynamicKernelParameter;
  }

  static constexpr size_t Size(int64_t size) {
    return std::max<size_t>(size, 0);
  }

  // If we know the number of arguments and results at compile time, we use
  // std::array with a fixed size, which allows compiler to automatically unroll
  // all the loops on a hot path.

  using ArgumentsBuffers = std::conditional_t<
      IsDynamic(num_arguments), std::vector<BufferAllocation::Slice>,
      std::array<BufferAllocation::Slice, Size(num_arguments)>>;

  using ResultsBuffers = std::conditional_t<
      IsDynamic(num_results), std::vector<BufferAllocation::Slice>,
      std::array<BufferAllocation::Slice, Size(num_results)>>;

  using KernelArgs = std::conditional_t<
      IsDynamic(num_arguments) || IsDynamic(num_results),
      absl::InlinedVector<SE_HOST_KernelArg, 8>,
      std::array<SE_HOST_KernelArg, Size(num_arguments + num_results)>>;

  KernelThunk(Info info,
              absl::Span<const BufferAllocation::Slice> arguments_buffers,
              absl::Span<const BufferAllocation::Slice> results_buffers,
              absl::flat_hash_set<int64_t> invariant_arguments,
              std::string kernel_name, se::ThreadDim thread_dim,
              std::optional<uint64_t> min_alignment);

  absl::Status CheckInvariantBuffersMemory(const KernelArgs& kernel_args) const;

  ArgumentsBuffers arguments_buffers_;
  ResultsBuffers results_buffers_;

  // A set of invariant arguments (their indices).
  absl::flat_hash_set<int64_t> invariant_arguments_;

  size_t num_kernel_args_;

  std::string kernel_name_;
  se::ThreadDim thread_dim_;
  std::optional<uint64_t> min_alignment_;

  // If `true`, host kernel will be called just once for a logical thread dim
  // (1,1,1). This is a fast path for small host kernels that have just one
  // logical thread dim.
  bool call_once_;

  // Lazily loaded host kernel corresponding to `kernel_name_`.
  absl::Mutex mutex_;
  std::optional<se::host::HostKernel> kernel_ ABSL_GUARDED_BY(mutex_);
  std::atomic<se::host::HostKernel*> kernel_ptr_;  // pointer to `kernel_`

  // Pre-initialized kernel arguments that are updated with memory addresses
  // before the kernel launch.
  KernelArgs kernel_args_;
};

}  // namespace internal

// Kernel thunk specialization for a small kernel with a statically known number
// of arguments and results.
template <int64_t num_arguments, int64_t num_results>
class SmallKernelThunk final
    : public internal::KernelThunk<num_arguments, num_results> {
  using Base = internal::KernelThunk<num_arguments, num_results>;

 public:
  using Base::Base;

  tsl::AsyncValueRef<Thunk::ExecuteEvent> Execute(
      const Thunk::ExecuteParams& params) final;
};

// Kernel thunk specialization for dynamic number of arguments and results.
class KernelThunk final : public internal::KernelThunk<> {
  using Base = internal::KernelThunk<>;

 public:
  using Base::Base;

  static absl::StatusOr<std::unique_ptr<Thunk>> Create(
      Thunk::Info info,
      absl::Span<const BufferAllocation::Slice> arguments_buffers,
      absl::Span<const BufferAllocation::Slice> results_buffers,
      std::string kernel_name, se::ThreadDim thread_dim,
      absl::flat_hash_set<int64_t> invariant_arguments,
      std::optional<uint64_t> min_alignment = std::nullopt);

  tsl::AsyncValueRef<Thunk::ExecuteEvent> Execute(
      const Thunk::ExecuteParams& params) final;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_KERNEL_THUNK_H_
