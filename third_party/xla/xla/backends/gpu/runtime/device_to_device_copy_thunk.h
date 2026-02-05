/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_DEVICE_TO_DEVICE_COPY_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_DEVICE_TO_DEVICE_COPY_THUNK_H_

#include <cstdint>
#include <memory>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"

namespace xla {
namespace gpu {

// A thunk that copies data from a device buffer to another device buffer.
class DeviceToDeviceCopyThunk : public Thunk {
 public:
  // Constructs a CopyThunk that copies host data from `source_buffer` to the
  // device buffer `destination_buffer`.
  DeviceToDeviceCopyThunk(ThunkInfo thunk_info,
                          const ShapedSlice& source_buffer,
                          const ShapedSlice& destination_buffer,
                          int64_t mem_size);

  DeviceToDeviceCopyThunk(const DeviceToDeviceCopyThunk&) = delete;
  DeviceToDeviceCopyThunk& operator=(const DeviceToDeviceCopyThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const ShapedSlice& source() const { return source_buffer_; }
  const ShapedSlice& destination() const { return destination_buffer_; }
  int64_t size_bytes() const { return mem_size_; }

  BufferUses buffer_uses() const override {
    return {
        BufferUse::Read(source_buffer_.slice, source_buffer_.shape),
        BufferUse::Write(destination_buffer_.slice, destination_buffer_.shape),
    };
  }

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<DeviceToDeviceCopyThunk>> FromProto(
      ThunkInfo thunk_info, const DeviceToDeviceCopyThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  friend bool operator==(const DeviceToDeviceCopyThunk& lhs,
                         const DeviceToDeviceCopyThunk& rhs) {
    if (lhs.size_bytes() != rhs.size_bytes()) {
      return false;
    }
    return std::tie(lhs.source_buffer_, lhs.destination_buffer_) ==
           std::tie(rhs.source_buffer_, rhs.destination_buffer_);
  }

  friend bool operator!=(const DeviceToDeviceCopyThunk& lhs,
                         const DeviceToDeviceCopyThunk& rhs) {
    return !(lhs == rhs);
  }

 private:
  const ShapedSlice source_buffer_;
  const ShapedSlice destination_buffer_;
  const int64_t mem_size_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_DEVICE_TO_DEVICE_COPY_THUNK_H_
