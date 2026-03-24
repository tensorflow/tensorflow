/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_METADATA_H_
#define XLA_STREAM_EXECUTOR_KERNEL_METADATA_H_

#include <cstdint>
#include <optional>

namespace stream_executor {

// KernelMetadata holds runtime-queryable attributes of a loaded kernel, such as
// registers allocated, shared memory used, etc.
// Not all platforms support reporting of all information, so each accessor
// returns false if the associated field is not populated in the underlying
// platform.
class KernelMetadata {
 public:
  KernelMetadata() = default;

  // Returns the number of registers used per thread executing this kernel.
  std::optional<int64_t> registers_per_thread() const {
    return registers_per_thread_;
  }

  // Returns the amount of [static] shared memory used per block executing this
  // kernel. Note that dynamic shared memory allocations are not (and can not)
  // be reported here (since they're not specified until kernel launch time).
  std::optional<int64_t> shared_memory_bytes() const {
    return shared_memory_bytes_;
  }

  void set_registers_per_thread(int registers_per_thread) {
    registers_per_thread_ = registers_per_thread;
  }
  void set_shared_memory_bytes(int shared_memory_bytes) {
    shared_memory_bytes_ = shared_memory_bytes;
  }

 private:
  std::optional<int64_t> registers_per_thread_;
  std::optional<int64_t> shared_memory_bytes_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_METADATA_H_
