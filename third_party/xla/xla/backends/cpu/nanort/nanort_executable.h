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

#ifndef XLA_BACKENDS_CPU_NANORT_NANORT_EXECUTABLE_H_
#define XLA_BACKENDS_CPU_NANORT_NANORT_EXECUTABLE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/runtime/thread_pool_task_runner.h"
#include "xla/ffi/execution_context.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "tsl/platform/mem.h"

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

class NanoRtExecutable {
 public:
  // Creates a new instance of the NanoRtExecutable from compatible XLA
  // executable.
  static absl::StatusOr<std::unique_ptr<NanoRtExecutable>> Create(
      std::unique_ptr<Executable> executable);

  // NanoRtExecutable can be asynchronous and return unavailable async value
  // that becomes available after the execution is complete. It is the caller's
  // responsibility to make sure that arguments, results and temp buffers are
  // alive during execution.
  using ExecuteEvent = tsl::Chain;

  class ExecuteOptions {
   public:
    ExecuteOptions()
        : intra_op_thread_pool_(nullptr),
          task_runner_(nullptr),
          local_device_id_(0),
          global_device_id_(0),
          device_assignment_(nullptr),
          launch_id_(0),
          ffi_context_(nullptr) {}
    // Sets the thread pool device on which to run Eigen subcomputations.
    //
    // This field must be set for XLA:CPU models that call Eigen routines, but
    // may be null otherwise.  Routines that use this field should always CHECK
    // (or TF_RET_CHECK) that it's not null before dereferencing it, so that
    // users get a clean crash rather than a segfault.
    //
    // Does not take ownership.
    ExecuteOptions& set_intra_op_thread_pool(
        const Eigen::ThreadPoolDevice* intra_op_thread_pool);

    ExecuteOptions& set_ffi_context(const ffi::ExecutionContext* ffi_context);

    ExecuteOptions& set_launch_id(int32_t launch_id);

    ExecuteOptions& set_local_device_id(LocalDeviceId local_device_id);
    ExecuteOptions& set_global_device_id(GlobalDeviceId global_device_id);

    ExecuteOptions& set_device_assignment(DeviceAssignment* device_assignment);

    const Eigen::ThreadPoolDevice* intra_op_thread_pool() const;
    ThreadPoolTaskRunner* task_runner() const;

    LocalDeviceId local_device_id() const { return local_device_id_; }
    GlobalDeviceId global_device_id() const { return global_device_id_; }
    DeviceAssignment* device_assignment() const { return device_assignment_; }
    int32_t launch_id() const { return launch_id_; }
    const ffi::ExecutionContext* ffi_context() const { return ffi_context_; }

   private:
    const Eigen::ThreadPoolDevice* intra_op_thread_pool_;
    std::unique_ptr<ThreadPoolTaskRunner> task_runner_;

    LocalDeviceId local_device_id_;
    GlobalDeviceId global_device_id_;
    DeviceAssignment* device_assignment_;

    // If non-zero, identifies this execution as part of a potentially
    // multi-device launch. This can be used to detect scheduling errors, e.g.
    // if multi-host programs are launched in different orders on different
    // hosts, the launch IDs may be used by the runtime to detect the mismatch.
    int32_t launch_id_;
    const ffi::ExecutionContext* ffi_context_;
  };

  // A non-owning read-only view into the XLA executable's argument buffer.
  class Argument {
   public:
    template <typename T>
    Argument(const T* data, int64_t size);

    inline Argument(const void* data, int64_t size);

    template <typename T>
    explicit Argument(absl::Span<const T> data);

    absl::Span<const std::byte> data() const { return data_; }

   private:
    absl::Span<const std::byte> data_;
  };

  // A non-owning writable view into the XLA executable's result buffer.
  class Result {
   public:
    template <typename T>
    Result(T* data, int64_t size);

    inline Result(void* data, int64_t size);

    template <typename T>
    explicit Result(absl::Span<T> data);

    absl::Span<std::byte> data() const { return data_; }

   private:
    absl::Span<std::byte> data_;
  };

  // A non-owning writable view into the XLA executable's temporary buffer (a
  // buffer that is used by the executable to store intermediate results).
  using PreallocatedTemp = absl::Span<std::byte>;

  // An owning writable byte buffer that can be used as a temporary buffer.
  template <size_t n>
  class ManagedTemp {
   public:
    explicit ManagedTemp(size_t size) : data_(size) {}

    ManagedTemp(const ManagedTemp&) = delete;
    ManagedTemp& operator=(const ManagedTemp&) = delete;

    PreallocatedTemp data() { return absl::MakeSpan(data_); }

   private:
    friend class NanoRtExecutable;
    using Allocator = tsl::port::AlignedAllocator<std::byte, Align()>;
    alignas(Align()) absl::FixedArray<std::byte, n, Allocator> data_;
  };

  tsl::AsyncValueRef<ExecuteEvent> Execute(absl::Span<const Argument> arguments,
                                           absl::Span<const Result> results,
                                           PreallocatedTemp temp = {},
                                           const ExecuteOptions& options = {});

  template <size_t n>
  tsl::AsyncValueRef<ExecuteEvent> Execute(absl::Span<const Argument> arguments,
                                           absl::Span<const Result> results,
                                           ManagedTemp<n>& temp,
                                           const ExecuteOptions& options = {}) {
    return Execute(arguments, results, temp.data(), std::move(options));
  }

  // Returns the size of the temp buffer required to run the executable.
  size_t temp_buffer_size() const;

 private:
  NanoRtExecutable(std::unique_ptr<Executable> executable,
                   std::vector<size_t> allocation_sizes,
                   std::vector<size_t> argument_to_allocation_index,
                   std::vector<size_t> result_to_allocation_index,
                   std::optional<size_t> temp_allocation_index);

  std::unique_ptr<Executable> executable_;
  std::vector<size_t> allocation_sizes_;

  // A mapping from the argument/result index to the index of the corresponding
  // allocation (defined by the executable's buffer assignment).
  std::vector<size_t> argument_to_allocation_index_;
  std::vector<size_t> result_to_allocation_index_;

  // Index of the temp allocation.
  std::optional<size_t> temp_allocation_index_;
};

template <typename T>
NanoRtExecutable::Argument::Argument(const T* data, int64_t size)
    : data_(reinterpret_cast<const std::byte*>(data), size * sizeof(T)) {}

NanoRtExecutable::Argument::Argument(const void* data, int64_t size)
    : data_(reinterpret_cast<const std::byte*>(data), size) {}

template <typename T>
NanoRtExecutable::Argument::Argument(absl::Span<const T> data)
    : Argument(data.data(), data.size()) {}

template <typename T>
NanoRtExecutable::Result::Result(T* data, int64_t size)
    : data_(reinterpret_cast<std::byte*>(data), size * sizeof(T)) {}

NanoRtExecutable::Result::Result(void* data, int64_t size)
    : data_(reinterpret_cast<std::byte*>(data), size) {}

template <typename T>
NanoRtExecutable::Result::Result(absl::Span<T> data)
    : Result(data.data(), data.size()) {}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_NANORT_NANORT_EXECUTABLE_H_
