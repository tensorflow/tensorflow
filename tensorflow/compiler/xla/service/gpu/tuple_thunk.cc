/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/tuple_thunk.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

Status TupleThunk::ExecuteOnStream(const BufferAllocations& buffer_allocations,
                                   se::Stream* stream,
                                   HloExecutionProfiler* profiler) {
  auto size = tuple_element_buffers_.size();
  auto tuple_element_buffer_addresses = absl::make_unique<void*[]>(size);
  for (int i = 0; i != size; ++i) {
    tuple_element_buffer_addresses[i] =
        buffer_allocations.GetDeviceAddress(tuple_element_buffers_[i]).opaque();
  }
  se::DeviceMemory<void*> dest_buffer_address(
      buffer_allocations.GetDeviceAddress(dest_buffer_));

  auto host_size = size * sizeof(void*);
  auto op_profiler = profiler->MakeScopedInstructionProfiler(hlo_instruction());
  if (!stream
           ->ThenMemcpy(&dest_buffer_address,
                        tuple_element_buffer_addresses.get(), host_size)
           .ok()) {
    return InternalError(
        "Unable to launch MemcpyH2D from %p to %p with size %lu",
        tuple_element_buffer_addresses.get(), dest_buffer_address.opaque(),
        host_size);
  }
  // Free the tuple address buffer when memcpy is done.
  auto* buffers_raw = tuple_element_buffer_addresses.release();
  if (!stream->ThenDoHostCallback([buffers_raw] { delete[] buffers_raw; })
           .ok()) {
    delete[] buffers_raw;
    return InternalError("Unable to enqueue host callback!");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
