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
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/service/executable.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/mem.h"

namespace xla::cpu {

class NanoRtExecutable {
 public:
  // Creates a new instance of the NanoRtExecutable from compatible XLA
  // executable.
  static absl::StatusOr<std::unique_ptr<NanoRtExecutable>> Create(
      std::unique_ptr<Executable> executable,
      std::shared_ptr<tsl::thread::ThreadPool> thread_pool);

  // NanoRtExecutable can be asynchronous and return unavailable async value
  // that becomes available after the execution is complete. It is the caller's
  // responsibility to make sure that arguments, results and temp buffers are
  // alive during execution.
  using ExecuteEvent = tsl::Chain;

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
                                           PreallocatedTemp temp = {});

  template <size_t n>
  tsl::AsyncValueRef<ExecuteEvent> Execute(absl::Span<const Argument> arguments,
                                           absl::Span<const Result> results,
                                           ManagedTemp<n>& temp) {
    return Execute(arguments, results, temp.data());
  }

  // Returns the size of the temp buffer required to run the executable.
  size_t temp_buffer_size() const;

 private:
  NanoRtExecutable(std::unique_ptr<Executable> executable,
                   std::shared_ptr<tsl::thread::ThreadPool> thread_pool,
                   std::vector<size_t> allocation_sizes,
                   std::vector<size_t> argument_to_allocation_index,
                   std::vector<size_t> result_to_allocation_index,
                   std::optional<size_t> temp_allocation_index);

  std::unique_ptr<Executable> executable_;
  std::shared_ptr<tsl::thread::ThreadPool> thread_pool_;

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
