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

#include "xla/backends/cpu/runtime/kernel_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace internal {

// Checks that all buffers are aligned to the minimum alignment. We codegen
// with the assumption that all buffers are aligned, and if they are not, we
// will crash with a segmentation fault, or worse, produce incorrect results.
static absl::Status CheckBufferAlignment(
    const Thunk::Info& info, uint64_t min_alignment,
    absl::Span<const XLA_CPU_KernelArg> kernel_args) {
  if (min_alignment == 0) return absl::OkStatus();

  for (int64_t i = 0; i < kernel_args.size(); ++i) {
    auto ptr = reinterpret_cast<uintptr_t>(kernel_args[i].data);
    if (ABSL_PREDICT_FALSE((ptr & (min_alignment - 1)) != 0)) {
      return Internal(
          "Host kernel %s buffer argument #%d (%p) is not aligned to a "
          "required minimum alignment of %d bytes",
          info.op_name, i, kernel_args[i].data, min_alignment);
    }
  }

  return absl::OkStatus();
}

// VLOGs kernel arguments resolved from the buffer allocations.
static void VlogKernelArgs(
    absl::Span<const BufferAllocation::Slice> arguments_buffers,
    absl::Span<const BufferAllocation::Slice> results_buffers,
    absl::Span<const XLA_CPU_KernelArg> kernel_args) {
  for (int64_t i = 0; i < arguments_buffers.size(); ++i) {
    VLOG(3) << absl::StreamFormat("  arg #%d: %s (%p)", i,
                                  arguments_buffers[i].ToString(),
                                  kernel_args[i].data);
  }
  for (int64_t i = 0; i < results_buffers.size(); ++i) {
    VLOG(3) << absl::StreamFormat(
        "  res #%d: %s (%p)", i, results_buffers[i].ToString(),
        kernel_args[arguments_buffers.size() + i].data);
  }
}

// Returns kernel buffer uses for a given arguments and results buffers.
static Thunk::BufferUses KernelBufferUses(
    absl::Span<const BufferAllocation::Slice> arguments_buffers,
    absl::Span<const BufferAllocation::Slice> results_buffers) {
  Thunk::BufferUses buffer_uses;
  for (const BufferAllocation::Slice& buffer : arguments_buffers) {
    buffer_uses.emplace_back(buffer, BufferUse::kRead);
  }
  for (const BufferAllocation::Slice& buffer : results_buffers) {
    buffer_uses.emplace_back(buffer, BufferUse::kWrite);
  }
  return buffer_uses;
}

template <int64_t num_arguments, int64_t num_results>
KernelThunk<num_arguments, num_results>::KernelThunk(
    Info info, absl::Span<const BufferAllocation::Slice> arguments_buffers,
    absl::Span<const BufferAllocation::Slice> results_buffers,
    absl::flat_hash_set<int64_t> invariant_arguments, std::string kernel_name,
    NumWorkGroups num_workgroups, std::optional<uint64_t> min_alignment)
    : KernelThunkBase(Kind::kKernel, std::move(info)),
      invariant_arguments_(std::move(invariant_arguments)),
      num_kernel_args_(arguments_buffers.size() + results_buffers.size()),
      kernel_name_(std::move(kernel_name)),
      num_workgroups_(num_workgroups),
      min_alignment_(min_alignment),
      call_once_(num_workgroups_ == NumWorkGroups()) {
  // Resize storage for arguments and results buffers if it is dynamic.
  if constexpr (IsDynamic(num_arguments)) {
    arguments_buffers_.resize(arguments_buffers.size());
  }
  if constexpr (IsDynamic(num_results)) {
    results_buffers_.resize(results_buffers.size());
  }

  // Copy buffers from the arguments and results.
  for (size_t i = 0; i < arguments_buffers.size(); ++i) {
    arguments_buffers_[i] = arguments_buffers[i];
  }
  for (size_t i = 0; i < results_buffers.size(); ++i) {
    results_buffers_[i] = results_buffers[i];
  }

  // Resize storage for kernel arguments if it is dynamic.
  if constexpr (IsDynamic(num_arguments) || IsDynamic(num_results)) {
    kernel_args_.resize(num_kernel_args_);
  }

  // Initialize kernel arguments with null pointers and known buffer sizes.
  // We'll use them as a template to resolve buffer addresses at run time.
  for (size_t i = 0; i < arguments_buffers.size(); ++i) {
    kernel_args_[i] = XLA_CPU_KernelArg{
        nullptr, static_cast<size_t>(arguments_buffers_[i].size())};
  }
  for (size_t i = 0; i < results_buffers.size(); ++i) {
    kernel_args_[arguments_buffers_.size() + i] = XLA_CPU_KernelArg{
        nullptr, static_cast<size_t>(results_buffers_[i].size())};
  }
}

template <int64_t num_arguments, int64_t num_results>
ABSL_ATTRIBUTE_ALWAYS_INLINE tsl::AsyncValueRef<Thunk::ExecuteEvent>
KernelThunk<num_arguments, num_results>::ExecuteInternal(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat(
      "Launch host kernel %s with %d arguments and %d results: #workgroups=%v",
      kernel_name_, arguments_buffers_.size(), results_buffers_.size(),
      num_workgroups_);

  alignas(64) KernelArgs kernel_args = kernel_args_;
  XLA_CPU_KernelArg* kernel_args_ptr = kernel_args.data();

  const BufferAllocations* allocations = params.buffer_allocations;

  for (const BufferAllocation::Slice& buffer : arguments_buffers_) {
    if constexpr (ShouldCheckBufferSlices()) {
      TF_ASSIGN_OR_RETURN(auto mem, allocations->GetDeviceAddress(buffer));
      kernel_args_ptr++->data = mem.opaque();
    } else {
      auto mem = allocations->GetDeviceAddressUnchecked(buffer);
      kernel_args_ptr++->data = mem.opaque();
    }
  }

  for (const BufferAllocation::Slice& buffer : results_buffers_) {
    if constexpr (ShouldCheckBufferSlices()) {
      TF_ASSIGN_OR_RETURN(auto mem, allocations->GetDeviceAddress(buffer));
      kernel_args_ptr++->data = mem.opaque();
    } else {
      auto mem = allocations->GetDeviceAddressUnchecked(buffer);
      kernel_args_ptr++->data = mem.opaque();
    }
  }

  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(3))) {
    VlogKernelArgs(arguments_buffers_, results_buffers_, kernel_args);
  }

  // Сheck that all resolved buffers are properly aligned, and that invariant
  // property holds.
  if constexpr (ShouldCheckBufferSlices()) {
    // TODO(abanas): Check also for overlapping buffers.
    TF_RETURN_IF_ERROR(
        CheckBufferAlignment(info(), min_alignment_.value_or(0), kernel_args));
    TF_RETURN_IF_ERROR(CheckInvariantBuffersMemory(kernel_args));
  }

  // TODO(ezhulenev): Kernel ptr should be loaded as a part of Thunk
  // initialization stage.
  absl::call_once(kernel_init_flag_, [&] {
    // Because thunks are owned by a parent CpuExecutable, we can safely assume
    // that kernel pointer will not change after we find it the first time.
    absl::StatusOr<FunctionLibrary::Kernel*> kernel_fn =
        params.function_library->ResolveFunction<FunctionLibrary::Kernel>(
            kernel_name_);

    if (kernel_fn.ok()) {
      kernel_.emplace(num_kernel_args_, *kernel_fn);
    } else {
      kernel_ = std::move(kernel_fn.status());
    }
  });

  TF_RETURN_IF_ERROR(kernel_.status());
  Kernel* kernel = &kernel_.value();

  // Use a fast path if kernel called just once.
  if (ABSL_PREDICT_TRUE(call_once_)) {
    TF_RETURN_IF_ERROR(kernel->CallOnce(kernel_args));
    return OkExecuteEvent();
  }

  // If intra-op thread pool is not nullptr, we launch HostKernel in async mode
  // by scheduling tasks into it. HostKernel launch completion will
  // automatically signal KernelThunk execute completion.
  if (ABSL_PREDICT_TRUE(params.intra_op_threadpool)) {
    return kernel->Launch(num_workgroups_, kernel_args,
                          params.intra_op_threadpool);
  }

  TF_RETURN_IF_ERROR(kernel->Launch(num_workgroups_, kernel_args));
  return OkExecuteEvent();
}

// Check if memory overlaps with any of the elements in the container.
static bool Contains(absl::Span<const XLA_CPU_KernelArg> container,
                     const XLA_CPU_KernelArg& memory) {
  return absl::c_any_of(container, [&memory](const XLA_CPU_KernelArg& element) {
    const std::byte* element_data = static_cast<std::byte*>(element.data);
    const std::byte* memory_data = static_cast<std::byte*>(memory.data);
    return (element.data < memory_data + memory.size) &&
           (element_data + element.size > memory.data);
  });
}

template <int64_t num_arguments, int64_t num_results>
absl::Status
KernelThunk<num_arguments, num_results>::CheckInvariantBuffersMemory(
    const KernelArgs& kernel_args) const {
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(10))) {
    VLOG(10) << "Verify invariant buffers: ";
    for (auto index : invariant_arguments_) {
      VLOG(10) << absl::StreamFormat("  invariant arg id: %d", index);
    }
  }

  auto arguments = absl::Span<const XLA_CPU_KernelArg>(
      kernel_args.data(), arguments_buffers_.size());
  auto results = absl::Span<const XLA_CPU_KernelArg>(
      kernel_args.data() + arguments_buffers_.size(), results_buffers_.size());

  // Verify all argument buffers.
  for (int64_t i = 0; i < arguments.size(); ++i) {
    const XLA_CPU_KernelArg& argument = arguments[i];
    if (invariant_arguments_.contains(i)) {
      // This argument should be read only, i.e. not one of the results.
      if (Contains(results, argument)) {
        return Internal("Argument marked as invariant aliases with a result");
      }
    } else {
      // For completeness, we check that a read write buffer is one of the
      // results.
      if (!Contains(results, argument)) {
        return Internal(
            "Argument not marked as invariant but doesn't alias with any "
            "results");
      }
    }
  }

  return absl::OkStatus();
}

template <int64_t num_arguments, int64_t num_results>
Thunk::BufferUses KernelThunk<num_arguments, num_results>::buffer_uses() const {
  return KernelBufferUses(arguments_buffers_, results_buffers_);
}

}  // namespace internal

tsl::AsyncValueRef<Thunk::ExecuteEvent> KernelThunk::Execute(
    const Thunk::ExecuteParams& params) {
  return Base::ExecuteInternal(params);
}

template <int64_t num_arguments, int64_t num_results>
tsl::AsyncValueRef<Thunk::ExecuteEvent>
SmallKernelThunk<num_arguments, num_results>::Execute(
    const Thunk::ExecuteParams& params) {
  return Base::ExecuteInternal(params);
}

absl::StatusOr<std::unique_ptr<Thunk>> KernelThunk::Create(
    Thunk::Info info,
    absl::Span<const BufferAllocation::Slice> arguments_buffers,
    absl::Span<const BufferAllocation::Slice> results_buffers,
    std::string kernel_name, NumWorkGroups num_workgroups,
    absl::flat_hash_set<int64_t> invariant_arguments,
    std::optional<uint64_t> min_alignment) {
  if (min_alignment.has_value() && !absl::has_single_bit(*min_alignment)) {
    return Internal("Host kernel %s minimum alignment %d is not a power of 2",
                    info.op_name, *min_alignment);
  }

  auto small_kernel_thunk = [&](auto num_arguments, auto num_results) {
    return absl::WrapUnique(
        new SmallKernelThunk<num_arguments(), num_results()>(
            std::move(info), arguments_buffers, results_buffers,
            std::move(invariant_arguments), std::move(kernel_name),
            num_workgroups, min_alignment));
  };

  static constexpr auto _0 = std::integral_constant<size_t, 0>{};
  static constexpr auto _1 = std::integral_constant<size_t, 1>{};
  static constexpr auto _2 = std::integral_constant<size_t, 2>{};
  static constexpr auto _3 = std::integral_constant<size_t, 3>{};
  static constexpr auto _4 = std::integral_constant<size_t, 4>{};
  static constexpr auto _5 = std::integral_constant<size_t, 5>{};
  static constexpr auto _6 = std::integral_constant<size_t, 6>{};
  static constexpr auto _7 = std::integral_constant<size_t, 7>{};
  static constexpr auto _8 = std::integral_constant<size_t, 8>{};
  static constexpr auto _9 = std::integral_constant<size_t, 9>{};
  static constexpr auto _10 = std::integral_constant<size_t, 10>{};
  static constexpr auto _11 = std::integral_constant<size_t, 11>{};
  static constexpr auto _12 = std::integral_constant<size_t, 12>{};

  std::pair<size_t, size_t> params(arguments_buffers.size(),
                                   results_buffers.size());

  // Return SmallKernelThunk specializations for the most common cases.
  // NOLINTBEGIN
  if (params == std::make_pair(_0(), _1())) return small_kernel_thunk(_0, _1);
  if (params == std::make_pair(_1(), _1())) return small_kernel_thunk(_1, _1);
  if (params == std::make_pair(_2(), _1())) return small_kernel_thunk(_2, _1);
  if (params == std::make_pair(_3(), _1())) return small_kernel_thunk(_3, _1);
  if (params == std::make_pair(_4(), _1())) return small_kernel_thunk(_4, _1);
  if (params == std::make_pair(_5(), _1())) return small_kernel_thunk(_5, _1);
  if (params == std::make_pair(_6(), _1())) return small_kernel_thunk(_6, _1);
  if (params == std::make_pair(_7(), _1())) return small_kernel_thunk(_7, _1);
  if (params == std::make_pair(_8(), _1())) return small_kernel_thunk(_8, _1);
  if (params == std::make_pair(_9(), _1())) return small_kernel_thunk(_9, _1);
  if (params == std::make_pair(_10(), _1())) return small_kernel_thunk(_10, _1);
  if (params == std::make_pair(_11(), _1())) return small_kernel_thunk(_11, _1);
  if (params == std::make_pair(_12(), _1())) return small_kernel_thunk(_12, _1);
  // NOLINTEND

  // Return a generic KernelThunk for dynamic numbers of arguments and results.
  return absl::WrapUnique(
      new KernelThunk(std::move(info), arguments_buffers, results_buffers,
                      std::move(invariant_arguments), std::move(kernel_name),
                      num_workgroups, min_alignment));
}

absl::StatusOr<std::unique_ptr<Thunk>> KernelThunk::Create(
    Thunk::Info info, const KernelSpec& kernel_spec,
    std::optional<uint64_t> min_alignment) {
  // TODO(ezhulenev): Migrate KernelSpec to use NumWorkGroups.
  NumWorkGroups num_workgroups{kernel_spec.thread_dim().x,
                               kernel_spec.thread_dim().y,
                               kernel_spec.thread_dim().z};
  return Create(std::move(info), kernel_spec.argument_buffers(),
                kernel_spec.result_buffers(), kernel_spec.name(),
                num_workgroups, kernel_spec.invariant_arguments(),
                min_alignment);
}

}  // namespace xla::cpu
