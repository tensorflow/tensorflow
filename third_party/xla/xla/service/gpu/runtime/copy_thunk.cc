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

#include "xla/service/gpu/runtime/copy_thunk.h"

#include <cstdint>

#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

DeviceToDeviceCopyThunk::DeviceToDeviceCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size)
    : Thunk(Kind::kCopy, thunk_info),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {}

absl::Status DeviceToDeviceCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_);
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_);
  VLOG(3) << "Memcpy D2D of size " << mem_size_ << " from "
          << source_data.opaque() << " to " << destination_data.opaque();
  return params.stream->Memcpy(&destination_data, source_data, mem_size_);
}

DeviceToHostCopyThunk::DeviceToHostCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size)
    : DeviceToDeviceCopyThunk(thunk_info, source_buffer, destination_buffer,
                              mem_size) {}

absl::Status DeviceToHostCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination());
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source());
  void* cpu_dst = destination_data.opaque();
  VLOG(3) << "Memcpy D2H for memory offload from " << source_data.opaque()
          << " to " << cpu_dst;
  return params.stream->Memcpy(cpu_dst, source_data, size_bytes());
}

HostToDeviceCopyThunk::HostToDeviceCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size)
    : DeviceToDeviceCopyThunk(thunk_info, source_buffer, destination_buffer,
                              mem_size) {}

absl::Status HostToDeviceCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination());
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source());
  void* cpu_src = source_data.opaque();
  VLOG(3) << "Memcpy H2D for memory offload from " << cpu_src << " to "
          << destination_data.opaque();
  return params.stream->Memcpy(&destination_data, cpu_src, size_bytes());
}

}  // namespace gpu
}  // namespace xla
