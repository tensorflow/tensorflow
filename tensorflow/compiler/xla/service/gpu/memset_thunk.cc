/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

namespace se = ::perftools::gputools;

Status MemzeroThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  se::DeviceMemoryBase dest_data = buffer_allocations.GetDeviceAddress(dest_);
  stream->ThenMemZero(&dest_data, dest_data.size());
  return Status::OK();
}

Status Memset32BitValueThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  se::DeviceMemoryBase dest_data = buffer_allocations.GetDeviceAddress(dest_);
  stream->ThenMemset32(&dest_data, value_, dest_data.size());
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
