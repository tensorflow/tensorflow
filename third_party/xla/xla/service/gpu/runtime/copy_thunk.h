/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_COPY_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_COPY_THUNK_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

// A thunk that copies data from a device buffer to another device buffer.
class DeviceToDeviceCopyThunk : public Thunk {
 public:
  // Constructs a CopyThunk that copies host data from `source_buffer` to the
  // device buffer `destination_buffer`. `mem_size` is the size of the data in
  // bytes.
  DeviceToDeviceCopyThunk(ThunkInfo thunk_info,
                          const BufferAllocation::Slice& source_buffer,
                          const BufferAllocation::Slice& destination_buffer,
                          uint64_t mem_size);

  DeviceToDeviceCopyThunk(const DeviceToDeviceCopyThunk&) = delete;
  DeviceToDeviceCopyThunk& operator=(const DeviceToDeviceCopyThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  void ClearCompileTimeInfo() override { Thunk::ClearCompileTimeInfo(); }

  const BufferAllocation::Slice& source() const { return source_buffer_; }
  const BufferAllocation::Slice& destination() const {
    return destination_buffer_;
  }
  uint64_t size_bytes() const { return mem_size_; }

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  const uint64_t mem_size_;
};

class DeviceToHostCopyThunk : public DeviceToDeviceCopyThunk {
 public:
  DeviceToHostCopyThunk(ThunkInfo thunk_info,
                        const BufferAllocation::Slice& source_buffer,
                        const BufferAllocation::Slice& destination_buffer,
                        uint64_t mem_size);
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
};

class HostToDeviceCopyThunk : public DeviceToDeviceCopyThunk {
 public:
  HostToDeviceCopyThunk(ThunkInfo thunk_info,
                        const BufferAllocation::Slice& source_buffer,
                        const BufferAllocation::Slice& destination_buffer,
                        uint64_t mem_size);
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_COPY_THUNK_H_
