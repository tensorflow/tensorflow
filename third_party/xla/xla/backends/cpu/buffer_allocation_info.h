/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_H_
#define XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_H_

#include <cassert>
#include <cstdint>

namespace xla::cpu {

// `BufferAllocationInfo` stores information about buffer allocations required
// by an XLA:CPU executable at run time. It corresponds to a `BufferAllocation`
// in the `BufferAssignment` for a compiled XLA program.
//
// This class decouples buffer allocation info from the `BufferAllocation`
// class, which brings in a heavy dependency set, including protobuf dependency,
// and a `BufferAssignment` itself. We use this lightweight class in places that
// don't want to bring in these dependencies, e.g. in AOT compilation.
class BufferAllocationInfo {
 public:
  // If buffer allocation is an in-out parameter, we use `kParameter` kind and
  // set both entry parameter and result numbers.
  enum class Kind : uint64_t {
    kConstant = 0,
    kTemp = 1,
    kParameter = 2,
    kThreadLocal = 3
  };

  bool is_constant() const { return kind_ == Kind::kConstant; }

  bool is_entry_parameter() const { return entry_param_number_ >= 0; }

  int32_t entry_parameter_number() const {
    assert(is_entry_parameter());  // WARNING: do not replace with DCHECK
    return entry_param_number_;
  }

  bool is_result() const { return result_number_ >= 0; }

  int32_t result_number() const {
    assert(is_result());  // WARNING: do not replace with DCHECK
    return result_number_;
  }

  // Returns true if this buffer is temporary scratch space required by the XLA
  // computations. These are always allocated by the runtime.
  bool is_temp() const { return kind_ == Kind::kTemp; }

  // Returns true if this buffer is allocated on the C stack or into registers.
  // These buffers are never allocated by the runtime.
  bool is_thread_local() const { return kind_ == Kind::kThreadLocal; }

  Kind kind() const { return kind_; }
  uint64_t size() const { return size_; }

  bool operator==(const BufferAllocationInfo& buffer_info) const {
    return kind_ == buffer_info.kind_ && size_ == buffer_info.size_ &&
           entry_param_number_ == buffer_info.entry_param_number_ &&
           result_number_ == buffer_info.result_number_;
  }

  static BufferAllocationInfo Temp(uint64_t size) {
    return BufferAllocationInfo(Kind::kTemp, size);
  }

  static BufferAllocationInfo Constant(uint64_t size) {
    return BufferAllocationInfo(Kind::kConstant, size);
  }

  static BufferAllocationInfo EntryParameter(uint64_t size,
                                             int32_t entry_param_number) {
    return BufferAllocationInfo(Kind::kParameter, size, entry_param_number);
  }

  static BufferAllocationInfo InOutParameter(uint64_t size,
                                             int32_t entry_param_number,
                                             int32_t result_number) {
    return BufferAllocationInfo(Kind::kParameter, size, entry_param_number,
                                result_number);
  }

  static BufferAllocationInfo Result(uint64_t size, int32_t result_number) {
    return BufferAllocationInfo(Kind::kParameter, size, -1, result_number);
  }

  static BufferAllocationInfo ThreadLocal(uint64_t size) {
    return BufferAllocationInfo(Kind::kThreadLocal, size);
  }

 private:
  BufferAllocationInfo(Kind kind, uint64_t size,
                       int32_t entry_param_number = -1,
                       int32_t result_number = -1)
      : kind_(kind),
        size_(size),
        entry_param_number_(entry_param_number),
        result_number_(result_number) {}

  Kind kind_;
  uint64_t size_;
  int32_t entry_param_number_ = -1;
  int32_t result_number_ = -1;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_BUFFER_ALLOCATION_INFO_H_
