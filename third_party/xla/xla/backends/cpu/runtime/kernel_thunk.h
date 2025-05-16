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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/runtime/workgroup_dim.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Forward declare thunk defined below.
class KernelThunk;

// Base class for kernel thunks required for serialization.
// A base class is needed so that we can serialize KernelThunk and
// SmallKernelThunk via the same interface.
class KernelThunkBase : public Thunk {
 public:
  virtual ~KernelThunkBase() = default;  // NOLINT: clang-tidy complains that
                                         // `override` should be used here.
  KernelThunkBase(Kind kind, Info info) : Thunk(kind, std::move(info)) {}

  virtual absl::string_view kernel_name() const = 0;
  virtual const WorkgroupDim& workgroup_dim() const = 0;
  virtual const std::optional<uint64_t>& min_alignment() const = 0;

  virtual absl::Span<const BufferAllocation::Slice> arguments_buffers()
      const = 0;

  virtual absl::Span<const BufferAllocation::Slice> results_buffers() const = 0;

  virtual const absl::flat_hash_set<int64_t>& invariant_arguments() const = 0;
};

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
class KernelThunk : public KernelThunkBase {
 public:
  BufferUses buffer_uses() const final;

  absl::string_view kernel_name() const final { return kernel_name_; }
  const WorkgroupDim& workgroup_dim() const final { return workgroup_dim_; }
  const std::optional<uint64_t>& min_alignment() const final {
    return min_alignment_;
  }

  absl::Span<const BufferAllocation::Slice> arguments_buffers() const final {
    return absl::MakeSpan(arguments_buffers_);
  }

  absl::Span<const BufferAllocation::Slice> results_buffers() const final {
    return absl::MakeSpan(results_buffers_);
  }

  const absl::flat_hash_set<int64_t>& invariant_arguments() const final {
    return invariant_arguments_;
  }

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
      absl::InlinedVector<XLA_CPU_KernelArg, 8>,
      std::array<XLA_CPU_KernelArg, Size(num_arguments + num_results)>>;

  KernelThunk(Info info,
              absl::Span<const BufferAllocation::Slice> arguments_buffers,
              absl::Span<const BufferAllocation::Slice> results_buffers,
              absl::flat_hash_set<int64_t> invariant_arguments,
              std::string kernel_name, WorkgroupDim workgroup_dim,
              std::optional<uint64_t> min_alignment);

  absl::Status CheckInvariantBuffersMemory(const KernelArgs& kernel_args) const;

  ArgumentsBuffers arguments_buffers_;
  ResultsBuffers results_buffers_;

  // A set of invariant arguments (their indices).
  absl::flat_hash_set<int64_t> invariant_arguments_;

  size_t num_kernel_args_;

  std::string kernel_name_;
  WorkgroupDim workgroup_dim_;
  std::optional<uint64_t> min_alignment_;

  // If `true`, host kernel will be called just once for a workgroup id
  // (0, 0, 0). This is a fast path for small host kernels that have just one
  // workgroup.
  bool call_once_;

  // Lazily loaded host kernel corresponding to `kernel_name_`.
  absl::once_flag kernel_init_flag_;
  absl::StatusOr<Kernel> kernel_;

  // Pre-initialized kernel arguments that are updated with memory addresses
  // before the kernel launch. Align `KernelArgs` to 64 bytes to allow aligned
  // moves on a hot path.
  alignas(64) KernelArgs kernel_args_;
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
      std::string kernel_name, WorkgroupDim workgroup_dim,
      absl::flat_hash_set<int64_t> invariant_arguments,
      std::optional<uint64_t> min_alignment = std::nullopt);

  static absl::StatusOr<std::unique_ptr<Thunk>> Create(
      Thunk::Info info, const KernelSpec& kernel_spec,
      std::optional<uint64_t> min_alignment);

  tsl::AsyncValueRef<Thunk::ExecuteEvent> Execute(
      const Thunk::ExecuteParams& params) final;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_KERNEL_THUNK_H_
