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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SDC_BUFFER_ID_H_
#define XLA_BACKENDS_GPU_RUNTIME_SDC_BUFFER_ID_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/thunk_id.h"

namespace xla::gpu {

// An ID that identifies a buffer within a program. It's a combination of the
// thunk ID and the buffer index within the thunk.
//
// A single buffer can be referred to by multiple SdcBufferIds, when it's being
// used in different thunks.
class SdcBufferId {
 public:
  SdcBufferId() = default;

  // Creates a SdcBufferId that represents the `buffer_idx`-th buffer of a thunk
  // with `thunk_info`.
  //
  // Returns an error if `buffer_idx` is too large to be represented in a
  // SdcBufferId.
  static absl::StatusOr<SdcBufferId> Create(ThunkId thunk_id,
                                            size_t buffer_idx) {
    if (buffer_idx >= (1 << kBitsReservedForBufferIndex)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Buffer index (%u) is too large to be represented in a SdcBufferId "
          "(max = %u)",
          buffer_idx, (1 << kBitsReservedForBufferIndex) - 1));
    }

    const uint32_t value = (static_cast<uint32_t>(thunk_id.value())
                            << kBitsReservedForBufferIndex) |
                           static_cast<uint32_t>(buffer_idx);
    return SdcBufferId(value);
  }

  ThunkId thunk_id() const {
    return ThunkId(value_ >> kBitsReservedForBufferIndex);
  }
  size_t buffer_idx() const {
    return value_ & ((1 << kBitsReservedForBufferIndex) - 1);
  }

  // Raw numeric value of the ID, for use in SdcLogEntry::entry_id.
  uint32_t value() const { return value_; }

  bool operator==(const SdcBufferId& other) const {
    return value_ == other.value_;
  }
  bool operator!=(const SdcBufferId& other) const { return !(*this == other); }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SdcBufferId& buffer_id) {
    absl::Format(&sink, "{thunk_id: %u, buffer_idx: %u}",
                 buffer_id.thunk_id().value(), buffer_id.buffer_idx());
  }

  template <typename H>
  friend H AbslHashValue(H h, const SdcBufferId& buffer_id) {
    return H::combine(std::move(h), buffer_id.value_);
  }

 private:
  // Out of 32 bits available in SDC entry id, reserve that much for the
  // buffer index. This limits us to:
  // - 2^kBitsReservedForBufferIndex max buffers per thunk
  // - 2^(32-kBitsReservedForBufferIndex) max thunks
  // Which hopefully is enough.
  static constexpr size_t kBitsReservedForBufferIndex = 8;

  explicit SdcBufferId(uint32_t value) : value_(value) {}

  uint32_t value_ = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SDC_BUFFER_ID_H_
