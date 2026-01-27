/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_ENCODED_BUFFER_ALLOCATION_INFO_H_
#define TENSORFLOW_COMPILER_TF2XLA_ENCODED_BUFFER_ALLOCATION_INFO_H_

#include <cstdint>

#include "xla/backends/cpu/buffer_allocation_info.h"

namespace xla {
namespace cpu {

// Encoded version of `BufferAllocationInfo`, which can be used to reconstruct
// the `BufferAllocationInfo` later. It's used in the AOT compiler, to
// represent buffer allocation info as a lightweight struct.
struct EncodedBufferAllocationInfo {
  EncodedBufferAllocationInfo(uint64_t packed_kind_and_size,
                              uint32_t entry_param_number,
                              uint32_t result_number)
      : packed_kind_and_size(packed_kind_and_size),
        entry_param_number(entry_param_number),
        result_number(result_number) {}

  // Encodes BufferAllocationInfo into the struct that can be used to
  // reconstruct the BufferAllocationInfo later using the constructor. We need
  // this because we use BufferAllocationInfo in places where using protocol
  // buffers would negatively impact binary size.
  explicit EncodedBufferAllocationInfo(
      const BufferAllocationInfo& buffer_info) {
    packed_kind_and_size = Pack(buffer_info.kind(), buffer_info.size());
    entry_param_number = buffer_info.is_entry_parameter()
                             ? buffer_info.entry_parameter_number()
                             : -1;
    result_number = buffer_info.is_result() ? buffer_info.result_number() : -1;
  }

  explicit operator BufferAllocationInfo() const {
    auto kind = UnpackKind(packed_kind_and_size);
    auto size = UnpackSize(packed_kind_and_size);
    int32_t entry_param_number = static_cast<int32_t>(this->entry_param_number);
    int32_t result_number = static_cast<int32_t>(this->result_number);

    switch (kind) {
      case BufferAllocationInfo::Kind::kConstant:
        return BufferAllocationInfo::Constant(size);
      case BufferAllocationInfo::Kind::kTemp:
        return BufferAllocationInfo::Temp(size);
      case BufferAllocationInfo::Kind::kParameter:
        if (entry_param_number >= 0 && result_number >= 0) {
          return BufferAllocationInfo::InOutParameter(size, entry_param_number,
                                                      result_number);
        }
        if (entry_param_number >= 0) {
          return BufferAllocationInfo::EntryParameter(size, entry_param_number);
        }
        return BufferAllocationInfo::Result(size, result_number);
      case BufferAllocationInfo::Kind::kThreadLocal:
        return BufferAllocationInfo::ThreadLocal(size);
    }
  }

  static uint64_t Pack(BufferAllocationInfo::Kind kind, uint64_t size) {
    return (static_cast<uint64_t>(size) << 2) | static_cast<uint64_t>(kind);
  }

  static constexpr BufferAllocationInfo::Kind UnpackKind(uint64_t packed) {
    return static_cast<BufferAllocationInfo::Kind>((packed << 62) >> 62);
  }

  static constexpr uint64_t UnpackSize(uint64_t packed) { return packed >> 2; }

  uint64_t packed_kind_and_size = 0;
  uint32_t entry_param_number = -1;
  uint32_t result_number = -1;
};
}  // namespace cpu

// TODO(ezhulenev): This is a temporary hack to keep `tfcompile` code working.
namespace cpu_function_runtime {
using BufferInfo = ::xla::cpu::BufferAllocationInfo;
using EncodedBufferInfo = ::xla::cpu::EncodedBufferAllocationInfo;
}  // namespace cpu_function_runtime

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_TF2XLA_ENCODED_BUFFER_ALLOCATION_INFO_H_
